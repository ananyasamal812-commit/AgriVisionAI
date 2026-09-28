from pathlib import Path
import matplotlib.pyplot as plt


class TrainingPlots:

    def __init__(self, output_dir="outputs"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_loss(self, train_loss, valid_loss):

        plt.figure(figsize=(8, 5))

        plt.plot(train_loss, label="Train Loss", linewidth=2)
        plt.plot(valid_loss, label="Validation Loss", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(self.output_dir / "loss_curve.png", dpi=300)

        plt.close()

    def plot_accuracy(self, train_acc, valid_acc):

        plt.figure(figsize=(8, 5))

        plt.plot(train_acc, label="Train Accuracy", linewidth=2)
        plt.plot(valid_acc, label="Validation Accuracy", linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training & Validation Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(self.output_dir / "accuracy_curve.png", dpi=300)

        plt.close()

    def plot_learning_rate(self, lr_history):

        plt.figure(figsize=(8, 5))

        plt.plot(lr_history, linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(self.output_dir / "lr_curve.png", dpi=300)

        plt.close()

    def plot_all(
        self,
        train_loss,
        valid_loss,
        train_acc,
        valid_acc,
        lr_history,
    ):

        self.plot_loss(train_loss, valid_loss)

        self.plot_accuracy(train_acc, valid_acc)

        self.plot_learning_rate(lr_history)

        print(f"Plots saved to: {self.output_dir}")