import json
from pathlib import Path
from uuid import uuid4
import yaml

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.models.model_factory import ModelFactory
from src.dataloader.transforms import get_valid_transforms
from src.explainability.gradcam import GradCAMGenerator
from src.inference.images import read_rgb


class Predictor:

    def __init__(
        self,
        checkpoint_path="checkpoints/best_model.pth",
        image_size=None,
        num_classes=38,
        class_file="configs/classes.json",
        model_name="efficientnet_b0",
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.root = Path(__file__).resolve().parents[2]
        if image_size is None:
            with open(self.root / "configs/config.yaml", encoding="utf-8") as config_file:
                image_size = yaml.safe_load(config_file)["dataset"]["image_size"]
        self.output_dir = self.root / "outputs" / "heatmaps"
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------- Load Class Names ---------------- #

        with open(self.root / class_file, "r", encoding="utf-8") as f:
            self.classes = json.load(f)

        # ---------------- Load Disease Knowledge Base ---------------- #

        disease_file = self.root / "data/disease_info.json"

        if disease_file.exists():

            with open(disease_file, "r") as f:
                self.disease_db = json.load(f)

        else:

            print("Warning: data/disease_info.json not found.")

            self.disease_db = {}

        # ---------------- Create Model ---------------- #

        self.model = ModelFactory.create(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=False,
        )

        checkpoint_path = self.root / checkpoint_path

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        if isinstance(checkpoint, dict) and checkpoint.get("smoke_test_only"):
            raise ValueError("A smoke-test checkpoint cannot be used for public predictions")
        if isinstance(checkpoint, dict) and checkpoint.get("classes") is not None:
            if checkpoint["classes"] != self.classes:
                raise ValueError("Checkpoint class order does not match configs/classes.json")
        if isinstance(checkpoint, dict) and checkpoint.get("image_size") is not None:
            image_size = checkpoint["image_size"]
        self.model_name = model_name
        self.image_size = image_size
        self.recorded_accuracy = checkpoint.get("accuracy") if isinstance(checkpoint, dict) else None
        self.recorded_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None

        if (
            isinstance(checkpoint, dict)
            and "model_state_dict" in checkpoint
        ):

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        else:

            self.model.load_state_dict(
                checkpoint
            )

        self.model.to(self.device)

        self.model.eval()

        # ---------------- GradCAM ---------------- #

        self.gradcam = GradCAMGenerator(
            self.model
        )

        # ---------------- Transform ---------------- #

        self.transform = get_valid_transforms(
            image_size=image_size
        )

    def preprocess(self, image_path):

        image = read_rgb(image_path)

        if image is None:

            raise ValueError(
                f"Unable to read image: {image_path}"
            )

        rgb_image = image.astype(
            np.float32
        ) / 255.0

        transformed = self.transform(
            image=image
        )

        tensor = transformed["image"]

        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device), rgb_image

    def predict(self, image_path):

        input_tensor, rgb_image = self.preprocess(
            image_path
        )

        with torch.inference_mode():

            outputs = self.model(
                input_tensor
            )

            probabilities = F.softmax(
                outputs,
                dim=1,
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1,
            )

        scores, indices = probabilities[0].topk(min(3, len(self.classes)))
        top_predictions = [
            {"class_index": index, "disease": self.classes[index], "confidence": round(score * 100, 2)}
            for score, index in zip(scores.tolist(), indices.tolist())
        ]
        class_index = prediction.item()
        warnings = []
        height, width = rgb_image.shape[:2]
        gray = cv2.cvtColor((rgb_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if min(height, width) < 128:
            warnings.append("This photo is small. A higher-resolution close-up may give a better result.")
        if gray.mean() < 35 or gray.mean() > 230:
            warnings.append("Lighting may hide leaf details. Try an evenly lit photo without glare.")
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 15:
            warnings.append("This photo may lack sharp detail. Try a focused close-up of one leaf.")
        uncertain = confidence.item() < 0.70 or (scores[0] - scores[1]).item() < 0.15
        if uncertain:
            warnings.append("The result is uncertain. Compare the possible matches and try another clear photo before taking action.")

        disease = self.classes[
            class_index
        ]

        info = self.disease_db.get(
            disease,
            {},
        )

        # ---------------- GradCAM ---------------- #

        gradcam_path = self.gradcam.generate(
            input_tensor=input_tensor,
            rgb_image=rgb_image,
            output_path=str(self.output_dir / f"{uuid4().hex}.png"),
        )

        return {
            "uncertain": uncertain,
            "warnings": warnings,
            "top_predictions": top_predictions,

            "class_index": class_index,

            "disease": disease,

            "confidence": round(
                confidence.item() * 100,
                2,
            ),

            "symptoms": info.get(
                "symptoms",
                "Not available",
            ),

            "cause": info.get(
                "cause",
                "Not available",
            ),

            "treatment": info.get(
                "treatment",
                "Not available",
            ),

            "prevention": info.get(
                "prevention",
                "Not available",
            ),

            "gradcam": Path(gradcam_path).relative_to(self.root).as_posix(),
        }
