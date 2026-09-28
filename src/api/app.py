from contextlib import asynccontextmanager
from pathlib import Path
import os
import asyncio
from contextlib import suppress
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api.dependencies import get_predictor
from src.api.routes import router
from src.api.storage import cleanup_outputs

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs" / "heatmaps"
OUTPUTS.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app):
    get_predictor()
    async def expire_outputs():
        while True:
            await asyncio.to_thread(cleanup_outputs)
            await asyncio.sleep(60)
    task = asyncio.create_task(expire_outputs())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

app = FastAPI(title="AgriVisionAI API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()],
    allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(router)
app.mount("/outputs/heatmaps", StaticFiles(directory=OUTPUTS), name="heatmaps")

@app.get("/")
def home():
    return {"message": "AgriVisionAI API is running"}

@app.get("/health")
def health():
    predictor = get_predictor()
    return {"status": "ok", "classes": len(predictor.classes), "device": str(predictor.device)}

