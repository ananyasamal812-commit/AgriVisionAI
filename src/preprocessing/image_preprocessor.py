from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.preprocessing.resize import ImageResizer
from src.preprocessing.normalizer import ImageNormalizer
from src.preprocessing.transforms import ImageTransforms


class ImagePreprocessor:

    def __init__(
        self,
        input_root,
        output_root,
        image_size=224,
    ):

        self.input_root = Path(input_root)
        self.output_root = Path(output_root)

        self.resizer = ImageResizer(image_size)
        self.normalizer = ImageNormalizer()
        self.transforms = ImageTransforms()

        self.supported = {
            ".jpg",
            ".jpeg",
            ".png",
        }

    def process(self):

        images = list(self.input_root.rglob("*"))

        processed = 0
        skipped = 0
        failed = 0

        for image_path in tqdm(images):

            if image_path.suffix.lower() not in self.supported:

                skipped += 1
                continue

            relative = image_path.relative_to(
                self.input_root
            )

            save_path = self.output_root / relative

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            try:

                image = Image.open(
                    image_path
                ).convert("RGB")

                image = self.transforms.apply(
                    image
                )

                image = self.resizer.resize(
                    image
                )

                normalized = self.normalizer.normalize(
                    image
                )

                normalized = (
                    normalized * 255
                ).clip(
                    0,
                    255,
                ).astype(
                    np.uint8
                )

                Image.fromarray(
                    normalized
                ).save(
                    save_path,
                    quality=95,
                )

                processed += 1

            except Exception as e:

                failed += 1

                print(
                    f"Error processing {image_path}"
                )

                print(e)

        print()

        print("=" * 60)
        print("IMAGE PREPROCESSING COMPLETE")
        print("=" * 60)
        print(f"Processed Images : {processed}")
        print(f"Skipped Files    : {skipped}")
        print(f"Failed Images    : {failed}")
        print(f"Saved To         : {self.output_root}")
        print("=" * 60)