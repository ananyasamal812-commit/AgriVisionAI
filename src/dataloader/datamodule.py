import os
import torch
from torch.utils.data import DataLoader

from src.dataloader.dataset import CropDiseaseDataset
from src.dataloader.transforms import (
    get_train_transforms,
    get_valid_transforms,
)


class CropDataModule:

    def __init__(
        self,
        dataset_root,
        batch_size=32,
        workers=0,  # Use 0 on Windows during development
        image_size=224,
    ):

        self.batch_size = batch_size
        self.workers = workers

        self.pin_memory = torch.cuda.is_available()

        self.train_dataset = CropDiseaseDataset(
            os.path.join(dataset_root, "train"),
            get_train_transforms(image_size),
        )

        self.valid_dataset = CropDiseaseDataset(
            os.path.join(dataset_root, "valid"),
            get_valid_transforms(image_size),
        )

        self.test_dataset = CropDiseaseDataset(
            os.path.join(dataset_root, "test"),
            get_valid_transforms(image_size),
        )

    def train_loader(self):

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
        )

    def valid_loader(self):

        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
        )

    def test_loader(self):

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=self.pin_memory,
        )