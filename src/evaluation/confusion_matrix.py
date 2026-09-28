from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


class ConfusionMatrixGenerator:

    def __init__(self, class_names):

        self.class_names = class_names

    def generate(
        self,
        y_true,
        y_pred,
        normalize=False,
        save_path="outputs/confusion_matrix.png",
    ):

        cm = confusion_matrix(y_true, y_pred)

        if normalize:

            cm = cm.astype(np.float32)

            cm = cm / cm.sum(axis=1, keepdims=True)

            cm = np.nan_to_num(cm)

        fig, ax = plt.subplots(figsize=(18, 18))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names,
        )

        display.plot(
            cmap="Blues",
            xticks_rotation=90,
            ax=ax,
            colorbar=True,
        )

        plt.title("Crop Disease Confusion Matrix")

        Path("outputs").mkdir(exist_ok=True)

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Confusion matrix saved to {save_path}")