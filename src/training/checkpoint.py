import os
import torch


class CheckpointManager:

    def __init__(self, save_dir="checkpoints"):

        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)

        self.best_path = os.path.join(
            save_dir,
            "best_model.pth",
        )

        self.last_path = os.path.join(
            save_dir,
            "last_model.pth",
        )

        self.best_accuracy = 0.0

    def save_best(
        self,
        model,
        optimizer,
        epoch,
        accuracy,
    ):

        if accuracy > self.best_accuracy:

            self.best_accuracy = accuracy

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": accuracy,
                },
                self.best_path,
            )

            print(f"Best model saved (Acc={accuracy:.4f})")

    def save_last(
        self,
        model,
        optimizer,
        epoch,
        accuracy,
    ):

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": accuracy,
            },
            self.last_path,
        )

    def load_last(
        self,
        model,
        optimizer,
        device,
    ):

        if not os.path.exists(self.last_path):

            print("No previous checkpoint found.")

            return 0

        checkpoint = torch.load(
            self.last_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        print(
            f"Resuming from Epoch {checkpoint['epoch']}"
        )

        return checkpoint["epoch"]