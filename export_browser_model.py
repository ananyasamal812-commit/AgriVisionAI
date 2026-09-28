"""Export the serving checkpoint with its original Grad-CAM target, and verify parity."""
import hashlib
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from src.models.model_factory import ModelFactory
from src.inference.images import read_rgb

ROOT = Path(__file__).resolve().parent


class BrowserModel(torch.nn.Module):
    """Analytic Grad-CAM through the 1x1-convolution/BN/SiLU/GAP/linear head.

    Target is blocks[-1], exactly as in src/explainability/gradcam.py.
    No approximation to the derivative is needed for this EfficientNet head.
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        assert backbone.conv_head.kernel_size == (1, 1)
        assert isinstance(backbone.bn2.act, torch.nn.SiLU)

    def forward(self, image):
        b = self.backbone
        activation = b.blocks(b.bn1(b.conv_stem(image)))
        raw = b.conv_head(activation)
        bn = b.bn2
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        normalized = (raw - bn.running_mean[None, :, None, None]) * scale[None, :, None, None] + bn.bias[None, :, None, None]
        features = F.silu(normalized)
        logits = b.classifier(features.mean(dim=(2, 3)))
        weights = b.classifier.weight[logits.argmax(dim=1)]
        sigmoid = normalized.sigmoid()
        gradient = weights[:, :, None, None] * sigmoid * (1 + normalized * (1 - sigmoid))
        gradient = gradient * scale[None, :, None, None] / (features.shape[2] * features.shape[3])
        gradient = F.conv2d(gradient, b.conv_head.weight.transpose(0, 1))
        cam = (gradient.mean(dim=(2, 3), keepdim=True) * activation).sum(dim=1).relu()
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-7)
        return logits, cam


def main():
    torch.set_num_threads(2)
    public = ROOT / 'frontend/public'
    target = public / 'model'
    fixtures = ROOT / 'outputs/browser-validation'
    target.mkdir(parents=True, exist_ok=True)
    fixtures.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ROOT / 'checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('smoke_test_only'):
        raise ValueError('Refusing to export a smoke-test checkpoint')
    classes = json.loads((ROOT / 'configs/classes.json').read_text())
    image_size = checkpoint.get('image_size', 128)
    wrapped = ModelFactory.create('efficientnet_b0', len(classes), pretrained=False)
    wrapped.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    wrapped.eval()
    wrapped.requires_grad_(False)
    browser = BrowserModel(wrapped.model).eval()
    path = target / 'agrivision.onnx'
    torch.onnx.export(browser, torch.zeros(1, 3, image_size, image_size), str(path),
                      input_names=['image'], output_names=['logits', 'heatmap'],
                      opset_version=17, dynamo=False, external_data=False)
    onnx.checker.check_model(str(path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    samples = sorted(p for p in (ROOT / 'sample_images').rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})[:3]
    for folder in sorted((ROOT / 'datasets/processed/plantvillage/test').glob('*'))[:12]:
        if folder.is_dir():
            sample = next((p for p in sorted(folder.iterdir()) if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}), None)
            if sample is not None:
                samples.append(sample)
    if not samples:
        raise ValueError('No real images available for export validation')
    records = []
    for index, sample in enumerate(samples):
        rgb = read_rgb(sample)
        resized = cv2.resize(rgb, (image_size, image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        normalized = (resized - np.array([.485, .456, .406], dtype=np.float32) * 255) * (1 / (np.array([.229, .224, .225], dtype=np.float32) * 255))
        tensor = torch.from_numpy(np.ascontiguousarray(normalized.transpose(2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            reference = wrapped(tensor).numpy()
            analytic_logits, analytic_cam = browser(tensor)
        outputs = session.run(None, {'image': tensor.numpy()})
        np.testing.assert_allclose(outputs[0], reference, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(outputs[1], analytic_cam.numpy(), atol=2e-4, rtol=2e-4)
        # Check the analytic heatmap against an independent autograd derivative.
        b = wrapped.model
        activation = b.blocks(b.bn1(b.conv_stem(tensor))).detach().requires_grad_(True)
        prediction = b.classifier(b.bn2(b.conv_head(activation)).mean(dim=(2, 3)))
        prediction[0, prediction.argmax(dim=1)[0]].backward()
        expected = (activation.grad.mean(dim=(2, 3), keepdim=True) * activation).sum(dim=1).relu()
        expected = expected - expected.amin(dim=(1, 2), keepdim=True)
        expected = expected / (expected.amax(dim=(1, 2), keepdim=True) + 1e-7)
        np.testing.assert_allclose(analytic_cam.numpy(), expected.detach().numpy(), atol=2e-4, rtol=2e-4)
        tensor.numpy().astype('<f4').tofile(fixtures / f'input-{index}.bin')
        rgba = cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA)
        rgba.tofile(fixtures / f'rgba-{index}.bin')
        record = {'index': index, 'width': rgb.shape[1], 'height': rgb.shape[0],
                  'class_index': int(reference.argmax()), 'logits': reference[0].tolist(),
                  'heatmap': outputs[1].reshape(-1).tolist(), 'heatmap_shape': list(outputs[1].shape),
                  'max_logit_error': float(np.abs(outputs[0] - reference).max())}
        records.append(record)
        print('Validated image', index + 1, 'max logit error', record['max_logit_error'], flush=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {'classes': classes, 'image_size': image_size, 'model_file': 'agrivision.onnx',
                'sha256': digest, 'model_bytes': path.stat().st_size, 'mean': [.485, .456, .406],
                'std': [.229, .224, .225], 'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                'heatmap_method': 'Grad-CAM at final EfficientNet block; analytic head derivative',
                'disease_info': json.loads((ROOT / 'data/disease_info.json').read_text())}
    (target / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
    (fixtures / 'reference.json').write_text(json.dumps(records), encoding='utf-8')
    (fixtures / 'export-validation.json').write_text(json.dumps({'images': len(records),
        'max_logit_error': max(r['max_logit_error'] for r in records), 'heatmap_autograd_parity': True,
        'model_bytes': path.stat().st_size, 'sha256': digest}, indent=2), encoding='utf-8')
    (public / 'fonts').mkdir(exist_ok=True)
    for name in ['NotoSansDevanagari-Full.ttf', 'OFL.txt']:
        shutil.copyfile(ROOT / 'assets/fonts' / name, public / 'fonts' / name)
    print('Browser model exported and verified:', path, flush=True)


if __name__ == '__main__':
    main()
