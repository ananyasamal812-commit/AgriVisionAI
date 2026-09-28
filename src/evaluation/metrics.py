import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class ClassificationMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.targets = []
        self.predictions = []
        self.probabilities = []

    def update(self, outputs, labels):
        """
        outputs: model logits
        labels: ground truth labels
        """

        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        self.targets.extend(labels.cpu().numpy())
        self.predictions.extend(preds.cpu().numpy())
        self.probabilities.extend(probs.cpu().numpy())

    def compute(self):

        accuracy = accuracy_score(
            self.targets,
            self.predictions
        )

        precision = precision_score(
            self.targets,
            self.predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            self.targets,
            self.predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            self.targets,
            self.predictions,
            average="weighted",
            zero_division=0,
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def top_k_accuracy(self, k=5):

        probs = np.array(self.probabilities)
        labels = np.array(self.targets)

        topk = np.argsort(probs, axis=1)[:, -k:]

        correct = 0

        for i in range(len(labels)):
            if labels[i] in topk[i]:
                correct += 1

        return correct / len(labels)