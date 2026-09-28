"""Run with: python -m uvicorn app:app --host 127.0.0.1 --port 8000."""
from src.api.app import app
__all__ = ["app"]

