"""Offline Hindi strings shared with the frontend."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HINDI = json.loads((ROOT / "data/locales/hi.json").read_text(encoding="utf-8"))


def translate(value, language="en"):
    text = str(value)
    if language != "hi":
        return text
    if text in HINDI:
        return HINDI[text]
    if text.endswith(" fungus."):
        return text[:-8] + " फफूंद।"
    if text.endswith(" bacteria."):
        return text[:-10] + " जीवाणु।"
    return text


def class_label(value, language="en"):
    crop, _, condition = value.partition("___")
    crop = crop.replace("_(including_sour)", "").replace("_(maize)", "").replace("Pepper,_bell", "Pepper").replace("_", " ").strip()
    condition = condition.replace("_", " ").strip()
    return translate(crop, language) + (" - " + translate(condition, language) if condition else "")
