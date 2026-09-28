from src.utils.config import Config
from src.dataloader.datamodule import CropDataModule
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer


def main():

    config = Config()

    data = CropDataModule(
        dataset_root=config.get("dataset", "root"),
        batch_size=config.get("training", "batch_size"),
        workers=config.get("training", "workers"),
        image_size=config.get("dataset", "image_size"),
    )

    model = ModelFactory.create(
        model_name=config.get("model", "name"),
        num_classes=config.get("dataset", "num_classes"),
        pretrained=config.get("model", "pretrained"),
    )

    trainer = Trainer(
        model=model,
        train_loader=data.train_loader(),
        valid_loader=data.valid_loader(),
        epochs=config.get("training", "epochs"),
        lr=config.get("training", "learning_rate"),
    )

    trainer.fit()


if __name__ == "__main__":
    main()