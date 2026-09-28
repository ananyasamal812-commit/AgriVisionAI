"""
scanner.py

Scans a dataset directory and collects statistics about image files.

Author: Ananya Samal
Project: AgriVisionAI
"""

from pathlib import Path
from collections import Counter
import logging
from typing import Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


class DatasetScanner:
    """
    Scans a dataset directory recursively.
    """

    def __init__(self, dataset_path: str):

        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset path not found: {self.dataset_path}"
            )

    def scan(self) -> Dict:
        """
        Scan dataset and collect statistics.
        """

        LOGGER.info("Scanning dataset...")

        image_files: List[Path] = []
        extension_counter = Counter()
        total_size = 0

        for file in self.dataset_path.rglob("*"):

            if not file.is_file():
                continue

            extension = file.suffix.lower()

            if extension not in SUPPORTED_EXTENSIONS:
                continue

            image_files.append(file)

            extension_counter[extension] += 1

            total_size += file.stat().st_size

        result = {
            "dataset_path": str(self.dataset_path),
            "total_images": len(image_files),
            "extensions": dict(extension_counter),
            "dataset_size_mb": round(total_size / (1024 * 1024), 2),
            "images": image_files
        }

        LOGGER.info("Scan complete.")

        return result

    def print_summary(self, result: Dict):

        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)

        print(f"Dataset Path : {result['dataset_path']}")
        print(f"Total Images : {result['total_images']}")
        print(f"Dataset Size : {result['dataset_size_mb']} MB")

        print("\nImage Types")

        for ext, count in result["extensions"].items():

            print(f"{ext:<8} {count}")

        print("=" * 60)