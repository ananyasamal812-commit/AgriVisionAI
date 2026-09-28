from pathlib import Path
import json

import torch


class CheckpointManager:

    def __init__(self, save_dir="checkpoints"):

        self.save_dir = Path(save_dir)

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_accuracy = 0.0

    # ---------------- Save Best ---------------- #

    def save_best(
        self,
        model,
        optimizer,
        epoch,
        accuracy,
    ):

        if accuracy <= self.best_accuracy:
            return

        self.best_accuracy = accuracy

        checkpoint = {
            "epoch": epoch,
            "accuracy": accuracy,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            self.save_dir / "best_model.pth",
        )

        with open(
            self.save_dir / "training_state.json",
            "w",
        ) as f:

            json.dump(
                {
                    "epoch": epoch,
                    "best_accuracy": accuracy,
                },
                f,
                indent=4,
            )

        print("\nBest model saved.")

    # ---------------- Save Last ---------------- #

    def save_last(
        self,
        model,
        optimizer,
        epoch,
        accuracy,
    ):

        checkpoint = {
            "epoch": epoch,
            "accuracy": accuracy,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            self.save_dir / "last_model.pth",
        )

    # ---------------- Load Best ---------------- #

    def load_best(
        self,
        model,
        device="cpu",
    ):

        path = self.save_dir / "best_model.pth"

        if not path.exists():

            print("No best checkpoint found.")

            return model

        checkpoint = torch.load(
            path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print("Best checkpoint loaded.")

        return model

    # ---------------- Resume Training ---------------- #

    def load_last(
        self,
        model,
        optimizer,
        device="cpu",
    ):

        path = self.save_dir / "last_model.pth"

        if not path.exists():

            print("No previous checkpoint found.")

            return 0

        checkpoint = torch.load(
            path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        print(
            f"Resuming from epoch {checkpoint['epoch']}"
        )

        return checkpoint["epoch"]