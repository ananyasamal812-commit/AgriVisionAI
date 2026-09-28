from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_DIR = PROJECT_ROOT / "datasets"

RAW_DIR = DATASETS_DIR / "raw"
MERGED_DIR = DATASETS_DIR / "merged"
PROCESSED_DIR = DATASETS_DIR / "processed"
METADATA_DIR = DATASETS_DIR / "metadata"
CACHE_DIR = DATASETS_DIR / "cache"

PLANTVILLAGE_DIR = RAW_DIR / "plantvillage"
KAGGLE_DIR = RAW_DIR / "kaggle"
ROBOFLOW_DIR = RAW_DIR / "roboflow"
MENDELEY_DIR = RAW_DIR / "mendeley"
CUSTOM_DIR = RAW_DIR / "custom"