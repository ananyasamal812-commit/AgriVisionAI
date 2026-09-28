from pathlib import Path
import csv


class TrainingHistory:

    def __init__(self, save_dir="outputs"):

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        self.file = self.save_dir / "training_history.csv"

        # Create file with header if it doesn't exist
        if not self.file.exists():
            with open(self.file, "w", newline="") as f:

                writer = csv.writer(f)

                writer.writerow([
                    "Epoch",
                    "Train Loss",
                    "Train Accuracy",
                    "Valid Loss",
                    "Valid Accuracy",
                    "Learning Rate",
                ])

    def log(
        self,
        epoch,
        train_loss,
        train_acc,
        valid_loss,
        valid_acc,
        lr,
    ):

        with open(self.file, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                round(train_loss, 4),
                round(train_acc, 4),
                round(valid_loss, 4),
                round(valid_acc, 4),
                lr,
            ])