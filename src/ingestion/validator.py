"""
validator.py

Production Dataset Validator

Features
--------
✔ Detect corrupt images
✔ Detect empty files
✔ Detect unsupported extensions
✔ Validate image dimensions
✔ Generate validation report
✔ Generate dataset statistics

Author: Ananya Samal
Project: AgriVisionAI
"""

from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm import tqdm
import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


class DatasetValidator:

    def __init__(self, dataset_root):

        self.dataset_root = Path(dataset_root)

        self.records = []

        self.class_counter = Counter()

    def validate(self):

        splits = ["train", "valid", "test"]

        total_images = 0

        valid_images = 0

        invalid_images = 0

        for split in splits:

            split_path = self.dataset_root / split

            if not split_path.exists():

                print(f"{split} folder not found")

                continue

            class_folders = sorted(
                [x for x in split_path.iterdir() if x.is_dir()]
            )

            for class_folder in tqdm(
                class_folders,
                desc=f"Validating {split}"
            ):

                class_name = class_folder.name

                self.class_counter[class_name] += len(
                    list(class_folder.iterdir())
                )

                for image_path in class_folder.iterdir():

                    total_images += 1

                    status = "Valid"

                    reason = ""

                    width = 0

                    height = 0

                    try:

                        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:

                            status = "Unsupported"

                            reason = "Unsupported Extension"

                        elif image_path.stat().st_size == 0:

                            status = "Invalid"

                            reason = "Empty File"

                        else:

                            with Image.open(image_path) as img:

                                img.verify()

                            with Image.open(image_path) as img:

                                width, height = img.size

                            if width < 64 or height < 64:

                                status = "Invalid"

                                reason = "Resolution Too Small"

                    except Exception as e:

                        status = "Corrupted"

                        reason = str(e)

                    if status == "Valid":

                        valid_images += 1

                    else:

                        invalid_images += 1

                    self.records.append({

                        "split": split,

                        "class": class_name,

                        "image": str(image_path),

                        "width": width,

                        "height": height,

                        "status": status,

                        "reason": reason

                    })

        report = pd.DataFrame(self.records)

        output = Path(
            "datasets/metadata/validation_report.csv"
        )

        output.parent.mkdir(parents=True, exist_ok=True)

        report.to_csv(output, index=False)

        stats = pd.DataFrame({

            "class": list(self.class_counter.keys()),

            "images": list(self.class_counter.values())

        })

        stats.to_csv(

            "datasets/metadata/class_distribution.csv",

            index=False

        )

        print("\n" + "=" * 60)

        print("VALIDATION SUMMARY")

        print("=" * 60)

        print(f"Total Images   : {total_images}")

        print(f"Valid Images   : {valid_images}")

        print(f"Invalid Images : {invalid_images}")

        print(f"Classes        : {len(self.class_counter)}")

        print("=" * 60)

        print("\nReports saved:")

        print(output)

        print("datasets/metadata/class_distribution.csv")

        return report