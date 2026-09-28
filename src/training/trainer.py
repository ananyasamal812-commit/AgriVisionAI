import torch

from src.utils.logger import Logger
from src.utils.checkpoint import CheckpointManager
from src.utils.tensorboard import TensorBoardLogger
from src.utils.history import TrainingHistory

from src.training.engine import Engine
from src.training.validator import Validator
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.training.losses import get_loss
from src.training.callbacks import EarlyStopping


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        epochs=10,
        lr=1e-3,
    ):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optimizer = get_optimizer(
            self.model,
            lr,
        )

        self.scheduler = get_scheduler(
            self.optimizer,
            epochs,
        )

        self.criterion = get_loss()

        self.engine = Engine(
            self.model,
            self.optimizer,
            self.criterion,
            self.device,
        )

        self.validator = Validator(
            self.model,
            self.criterion,
            self.device,
        )

        self.early = EarlyStopping()

        self.checkpoint = CheckpointManager()

        self.logger = Logger()

        self.tb = TensorBoardLogger()

        # Training History CSV
        self.history = TrainingHistory()

        self.epochs = epochs

        # Resume Training
        #self.start_epoch = self.checkpoint.load_last(
           # model=self.model,
            #optimizer=self.optimizer,
            #device=self.device,
        #)
        # Start training from scratch
                # Start training from scratch
                # Start training from scratch
        self.start_epoch = 0

    def fit(self):
        self.logger.info("Training Started")

        for epoch in range(self.start_epoch, self.epochs):

            # ---------------- Train ---------------- #

            train_loss, train_accuracy = self.engine.train_one_epoch(
                self.train_loader
            )

            # ---------------- Validation ---------------- #

            valid_loss, valid_accuracy = self.validator.validate(
                self.valid_loader
            )

            # ---------------- Scheduler ---------------- #

            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # ---------------- Console Logger ---------------- #

            self.logger.info("=" * 60)

            self.logger.info(
                f"Epoch {epoch + 1}/{self.epochs}"
            )

            self.logger.info(
                f"Train Loss      : {train_loss:.4f}"
            )

            self.logger.info(
                f"Train Accuracy  : {train_accuracy:.4f}"
            )

            self.logger.info(
                f"Valid Loss      : {valid_loss:.4f}"
            )

            self.logger.info(
                f"Valid Accuracy  : {valid_accuracy:.4f}"
            )

            self.logger.info(
                f"Learning Rate   : {current_lr:.6f}"
            )

            # ---------------- TensorBoard ---------------- #

            self.tb.log_scalar(
                "Loss/Train",
                train_loss,
                epoch + 1,
            )

            self.tb.log_scalar(
                "Loss/Validation",
                valid_loss,
                epoch + 1,
            )

            self.tb.log_scalar(
                "Accuracy/Train",
                train_accuracy,
                epoch + 1,
            )

            self.tb.log_scalar(
                "Accuracy/Validation",
                valid_accuracy,
                epoch + 1,
            )

            self.tb.log_scalar(
                "Learning Rate",
                current_lr,
                epoch + 1,
            )

            # ---------------- Training History ---------------- #

            self.history.log(
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_accuracy,
                valid_loss=valid_loss,
                valid_acc=valid_accuracy,
                lr=current_lr,
            )

            # ---------------- Checkpoints ---------------- #

            self.checkpoint.save_best(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                accuracy=valid_accuracy,
            )

            self.checkpoint.save_last(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                accuracy=valid_accuracy,
            )

            # ---------------- Early Stopping ---------------- #

            if self.early.step(valid_accuracy):

                self.logger.warning(
                    "Early stopping triggered."
                )

                break

        self.tb.close()

        self.logger.info("Training Finished")