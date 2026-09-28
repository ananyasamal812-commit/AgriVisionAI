from pathlib import Path

import cv2

from src.dataloader.transforms import get_valid_transforms


class ImageLoader:

    def __init__(self, image_size=224):

        self.transform = get_valid_transforms(image_size)

    def load(self, image_path):

        image = cv2.imread(str(Path(image_path)))

        if image is None:
            raise FileNotFoundError(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = self.transform(image=image)["image"]

        return image.unsqueeze(0)