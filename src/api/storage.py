import logging
import time
from pathlib import Path

OUTPUTS = Path(__file__).resolve().parents[2] / "outputs"
RETENTION_SECONDS = 24 * 60 * 60


def is_expired(path):
    try:
        return time.time() - path.stat().st_mtime > RETENTION_SECONDS
    except FileNotFoundError:
        return True


def cleanup_outputs():
    for folder, suffix in (("heatmaps", ".png"), ("reports", ".pdf")):
        for path in (OUTPUTS / folder).glob("*" + suffix):
            if is_expired(path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logging.warning("Could not expire output %s", path.name)
