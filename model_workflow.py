"""Reproducible candidate training and evaluation, without overwriting the serving model."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class LeafDataset:
    def __init__(self, root, classes, transform, limit=None, seed=42, excluded=()):
        self.transform = transform
        self.samples = []
        actual = sorted(p.name for p in root.iterdir() if p.is_dir())
        if actual != classes:
            raise ValueError(f"Class folders in {root} do not match the supplied class mapping")
        rng = random.Random(seed)
        for label, name in enumerate(classes):
            paths = sorted(p for p in (root / name).rglob("*") if p.suffix.lower() in EXTENSIONS and p.resolve() not in excluded)
            if not paths:
                raise ValueError(f"No images for {name}")
            rng.shuffle(paths)
            self.samples.extend((p, label) for p in (paths[:limit] if limit else paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from src.inference.images import read_rgb
        path, label = self.samples[index]
        return self.transform(image=read_rgb(path))["image"], label


def score(model, loader, device, classes):
    import torch
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    model.eval()
    targets, predictions, top3 = [], [], 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            ranked = logits.topk(3, dim=1).indices.cpu()
            targets.extend(labels.tolist())
            predictions.extend(ranked[:, 0].tolist())
            top3 += (ranked == labels[:, None]).any(dim=1).sum().item()
    return {"samples": len(targets), "accuracy": accuracy_score(targets, predictions),
            "top3_accuracy": top3 / len(targets),
            "per_class": classification_report(targets, predictions, labels=list(range(len(classes))),
                                               target_names=classes, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(classes)))).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate", "audit"])
    parser.add_argument("--data", default="datasets/processed/plantvillage")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--classes", default="configs/classes.json")
    parser.add_argument("--transfer-backbone", action="store_true", help="Training only: replace the classifier for a new class mapping")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the new classifier; useful for an initial warm-up")
    parser.add_argument("--output", default="outputs/model-review")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-per-class", type=int, default=None, help="For smoke tests only; omit for a complete run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-report", default="outputs/data-audit/data-audit.json")
    args = parser.parse_args()
    classes = json.loads((ROOT / args.classes).read_text(encoding="utf-8"))
    if args.transfer_backbone and args.mode != "train":
        raise ValueError("Backbone transfer is only allowed when training a candidate")
    data = (ROOT / args.data).resolve()
    output = (ROOT / args.output).resolve()
    if output == ROOT / "checkpoints" or (output / "candidate.pth") == (ROOT / args.checkpoint).resolve():
        raise ValueError("Choose an isolated output directory, not the serving checkpoint directory")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (ROOT / args.checkpoint).resolve()
    if args.mode == "audit":
        seen = {}
        overlaps = []
        counts = {}
        for split in ("train", "valid", "test"):
            counts[split] = 0
            for path in sorted((data / split).rglob("*")):
                if path.suffix.lower() not in EXTENSIONS:
                    continue
                counts[split] += 1
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                previous = seen.get(digest)
                if previous and previous[0] != split:
                    overlaps.append([previous[1], str(path.relative_to(data))])
                else:
                    seen[digest] = (split, str(path.relative_to(data)))
            print(split, counts[split], flush=True)
        report = {"data": str(data), "counts": counts, "cross_split_exact_duplicates": len(overlaps), "examples": overlaps,
                  "note": "Exact file hashes only; near duplicates and field generalization are not assessed."}
        (output / "data-audit.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "examples"}), flush=True)
        return
    audit_path = ROOT / args.audit_report
    if not audit_path.is_file():
        raise ValueError("Run the audit command first, then pass its data-audit.json with --audit-report.")
    audit = json.loads(audit_path.read_text())
    if audit.get("data") != str(data):
        raise ValueError("Rerun the data audit for this dataset before training or evaluating.")
    excluded = {(data / pair[1]).resolve() for pair in audit["examples"]}
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from src.dataloader.transforms import get_valid_transforms
    from src.models.model_factory import ModelFactory
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    source_classes = checkpoint.get("classes", json.loads((ROOT / "configs/classes.json").read_text()))
    if source_classes != classes and not args.transfer_backbone:
        raise ValueError("Checkpoint class mapping mismatch")
    image_size = checkpoint.get("image_size", 128)
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelFactory.create(model_name, len(classes), pretrained=False).to(device)
    state = checkpoint["model_state_dict"]
    if args.transfer_backbone:
        state = {key: value for key, value in state.items() if not key.startswith("model.classifier.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) != {"model.classifier.weight", "model.classifier.bias"} or unexpected:
            raise ValueError("Backbone transfer requires a compatible EfficientNet checkpoint")
    else:
        model.load_state_dict(state)
    if args.freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("model.classifier.")
    del checkpoint, state
    metadata = {"checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                "data": str(data), "seed": args.seed, "max_per_class": args.max_per_class,
                "image_size": image_size, "model_name": model_name,
                "classes": classes, "transfer_backbone": args.transfer_backbone,
                "freeze_backbone": args.freeze_backbone,
                "excluded_exact_duplicates": len(excluded),
                "scope": "Dataset benchmark, not internet/field accuracy. Known exact overlaps excluded; near duplicates remain unaudited."}
    if args.mode == "evaluate":
        dataset = LeafDataset(data / args.split, classes, get_valid_transforms(image_size), args.max_per_class, args.seed, excluded)
        report = {**metadata, "split": args.split, **score(model, DataLoader(dataset, batch_size=args.batch_size), device, classes)}
        (output / "evaluation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: report[k] for k in ["split", "samples", "accuracy", "top3_accuracy", "scope"]}), flush=True)
        return
    transform = A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.65, 1.0)),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.Rotate(limit=25, p=0.5),
        A.RandomBrightnessContrast(p=0.4), A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.ImageCompression(quality_range=(60, 100), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])
    train = LeafDataset(data / "train", classes, transform, args.max_per_class, args.seed)
    valid = LeafDataset(data / "valid", classes, get_valid_transforms(image_size), args.max_per_class, args.seed, excluded)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    valid_loader = DataLoader(valid, batch_size=args.batch_size)
    baseline = score(model, valid_loader, device, classes)["accuracy"]
    print(json.dumps({"phase": "baseline", "validation_accuracy": baseline, "train_samples": len(train), "valid_samples": len(valid), "device": str(device)}), flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3 if args.freeze_backbone else 1e-4, weight_decay=1e-4)
    history, best = [], -1
    for epoch in range(args.epochs):
        started = time.monotonic()
        model.train()
        if args.freeze_backbone:
            model.eval()
            model.model.classifier.train()
        loss_total = 0
        for step, (images, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images.to(device)), labels.to(device))
            loss.backward(); optimizer.step()
            loss_total += loss.item() * labels.size(0)
            if step % 100 == 0:
                progress = {"epoch": epoch + 1, "batch": step, "batches": len(train_loader), "elapsed_seconds": round(time.monotonic() - started)}
                (output / "progress.json").write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
        accuracy = score(model, valid_loader, device, classes)["accuracy"]
        row = {"epoch": epoch + 1, "train_loss": loss_total / len(train), "validation_accuracy": accuracy}
        history.append(row); print(json.dumps(row), flush=True)
        if accuracy > best:
            best = accuracy
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "classes": classes, "image_size": image_size, "model_name": model_name,
                        "epoch": epoch + 1, "accuracy": accuracy, "seed": args.seed,
                        "smoke_test_only": args.max_per_class is not None}, output / "candidate.pth")
    (output / "training.json").write_text(json.dumps({**metadata, "train_samples": len(train), "valid_samples": len(valid),
        "baseline_validation_accuracy": baseline, "best_candidate_validation_accuracy": best, "history": history,
        "promotion": "Not deployed. Smoke runs are not evidence for model promotion. Validate on curated field data before replacement."}, indent=2))


if __name__ == "__main__":
    main()
