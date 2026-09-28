from pathlib import Path

from src.data.registry import DATASETS
from src.data.config import RAW_DIR


class DatasetDownloader:

    def __init__(self):

        self.raw_dir = RAW_DIR

    def show_sources(self):

        print("\nAvailable Dataset Sources\n")

        for key, value in DATASETS.items():

            print(f"{key}")

            print(f" Name : {value['name']}")

            print(f" Type : {value['download_type']}")

            print()


if __name__ == "__main__":

    downloader = DatasetDownloader()

    downloader.show_sources()