import logging
from io import BytesIO
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Literal
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from src.api.dependencies import get_predictor
from fastapi.responses import FileResponse
from src.api.report import create_report
from src.inference.images import read_rgb
from src.api.storage import cleanup_outputs, is_expired

router = APIRouter()
prediction_lock = Lock()
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
UPLOADS = Path(__file__).resolve().parents[2] / "uploads"
ROOT = UPLOADS.parent
REPORTS = ROOT / "outputs" / "reports"

@router.post("/predict")
def predict(file: UploadFile = File(...), predictor=Depends(get_predictor)):
    """Validate an image and serialize inference because GradCAM uses model hooks."""
    image_path = None
    cleanup_outputs()
    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image must be 10 MB or smaller.")
        if not content:
            raise HTTPException(400, "The uploaded file is empty.")
        try:
            with Image.open(BytesIO(content)) as image:
                if image.format not in {"JPEG", "PNG", "WEBP"}:
                    raise HTTPException(415, "Please upload a JPEG, PNG, or WebP image.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise HTTPException(413, "Image must contain no more than 20 million pixels.")
                image.verify()
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "This file is not a valid image.") from exc
        UPLOADS.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=UPLOADS, suffix=".png", delete=False) as target:
            image_path = Path(target.name)
            target.write(content)
        normalized = Image.fromarray(read_rgb(image_path))
        normalized.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        normalized.save(image_path, format="PNG")
        if not prediction_lock.acquire(timeout=15):
            raise HTTPException(429, "The service is busy. Please try again in a moment.")
        try:
            result = predictor.predict(image_path)
        finally:
            prediction_lock.release()
        report_id = Path(result["gradcam"]).stem
        result["report_id"] = report_id
        result["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        REPORTS.mkdir(parents=True, exist_ok=True)
        try:
            create_report(result, image_path, ROOT / result["gradcam"], REPORTS / f"{report_id}.pdf")
            create_report(result, image_path, ROOT / result["gradcam"], REPORTS / f"{report_id}.hi.pdf", language="hi")
            result["report_url"] = f"reports/{report_id}"
        except Exception:
            logging.exception("Report generation failed")
            result["report_url"] = None
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(400, "Unable to decode this image. Try another JPEG or PNG.") from exc
    except Exception as exc:
        logging.exception("Prediction failed")
        raise HTTPException(500, "Prediction failed. Check the API terminal for details.") from exc
    finally:
        file.file.close()
        if image_path is not None:
            image_path.unlink(missing_ok=True)


@router.get("/reports/{report_id}")
def download_report(report_id: str, language: Literal["en", "hi"] = "en"):
    if not re.fullmatch(r"[a-f0-9]{32}", report_id):
        raise HTTPException(404, "Report not found.")
    suffix = ".hi" if language == "hi" else ""
    path = REPORTS / f"{report_id}{suffix}.pdf"
    if not path.is_file() or is_expired(path):
        raise HTTPException(404, "Report expired or not found. Analyze the image again.")
    return FileResponse(path, media_type="application/pdf", filename=f"AgriVisionAI-{report_id[:8]}-{language}.pdf")


@router.get("/evaluation")
def evaluation(predictor=Depends(get_predictor)):
    if os.environ.get("ENABLE_EVALUATION", "false").lower() != "true":
        raise HTTPException(404, "Not found")
    accuracy = predictor.recorded_accuracy
    return {
        "model": predictor.model_name,
        "image_size": predictor.image_size,
        "classes": predictor.classes,
        "validation_accuracy": round(float(accuracy) * 100, 2) if accuracy is not None else None,
        "recorded_epoch": predictor.recorded_epoch,
        "source": "checkpoints/best_model.pth (accuracy and epoch metadata)",
        "split": "Training-time validation; used to select the best checkpoint",
        "test_accuracy": None,
        "limitations": [
            "The saved validation score is not an independent test or field accuracy measurement.",
            "Validation sample count, per-class metrics, and data-leakage checks are not recorded in this checkpoint.",
            "Only the listed crop and disease classes are supported; other photos are not automatically rejected.",
            "Confidence scores are not calibrated probabilities of a correct diagnosis.",
            "Heatmaps indicate model attention, not disease severity or affected area."
        ]
    }

