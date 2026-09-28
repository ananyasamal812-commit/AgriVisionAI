from config import *
from utils import create_directories

def initialize_dataset_structure():

    folders = [
        RAW_DIR,
        MERGED_DIR,
        PROCESSED_DIR,
        METADATA_DIR,
        CACHE_DIR,
        PLANTVILLAGE_DIR,
        KAGGLE_DIR,
        ROBOFLOW_DIR,
        MENDELEY_DIR,
        CUSTOM_DIR,
    ]

    create_directories(folders)

    print("\nDataset structure initialized successfully!")

if __name__ == "__main__":
    initialize_dataset_structure()