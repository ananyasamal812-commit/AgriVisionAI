from pathlib import Path
import json

import torch
from tqdm import tqdm

from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.confusion_matrix import ConfusionMatrixGenerator


class Evaluator:

    def __init__(
        self,
        model,
        dataloader,
        class_names,
        device=None,
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)
        self.loader = dataloader
        self.class_names = class_names

        self.metrics = ClassificationMetrics()

    def evaluate(self):

        self.model.eval()

        self.metrics.reset()

        all_targets = []
        all_predictions = []

        with torch.no_grad():

            for images, labels in tqdm(self.loader):

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                self.metrics.update(outputs, labels)

                preds = outputs.argmax(dim=1)

                all_targets.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())

        results = self.metrics.compute()

        results["top5_accuracy"] = self.metrics.top_k_accuracy(5)

        Path("outputs").mkdir(exist_ok=True)

        with open("outputs/metrics.json", "w") as f:

            json.dump(results, f, indent=4)

        ConfusionMatrixGenerator(
            self.class_names
        ).generate(
            all_targets,
            all_predictions,
            normalize=False,
        )

        print("\n========== Evaluation Results ==========")

        for key, value in results.items():

            print(f"{key:15}: {value:.4f}")

        print("========================================")

        return results