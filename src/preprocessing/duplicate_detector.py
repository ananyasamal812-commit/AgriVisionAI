"""
duplicate_detector.py

Detect duplicate and near-duplicate images using perceptual hashing.

Author: Ananya Samal
Project: AgriVisionAI
"""

from pathlib import Path
from collections import defaultdict

import imagehash
import pandas as pd
from PIL import Image
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


class DuplicateDetector:

    def __init__(self, dataset_root):

        self.dataset_root = Path(dataset_root)

        self.hashes = {}

        self.duplicates = []

    def compute_hash(self, image_path):

        image = Image.open(image_path)

        return imagehash.phash(image)

    def scan(self):

        images = []

        for file in self.dataset_root.rglob("*"):

            if file.suffix.lower() in SUPPORTED_EXTENSIONS:

                images.append(file)

        print(f"\nScanning {len(images)} images...\n")

        for image in tqdm(images):

            try:

                current_hash = self.compute_hash(image)

                if current_hash in self.hashes:

                    self.duplicates.append({

                        "original": str(self.hashes[current_hash]),

                        "duplicate": str(image),

                        "hash": str(current_hash)

                    })

                else:

                    self.hashes[current_hash] = image

            except Exception as e:

                print(f"Skipped {image}: {e}")

        dataframe = pd.DataFrame(self.duplicates)

        output = Path("datasets/metadata/duplicates.csv")

        output.parent.mkdir(parents=True, exist_ok=True)

        dataframe.to_csv(output, index=False)

        print("\n" + "=" * 60)

        print("DUPLICATE DETECTION")

        print("=" * 60)

        print(f"Total Images      : {len(images)}")

        print(f"Duplicates Found  : {len(dataframe)}")

        print(f"Saved Report      : {output}")

        print("=" * 60)

        return dataframe

    def remove_duplicates(self):

        removed = 0

        for row in self.duplicates:

            duplicate = Path(row["duplicate"])

            if duplicate.exists():

                duplicate.unlink()

                removed += 1

        print(f"\nRemoved {removed} duplicate images.")