"""Consistent EXIF orientation and color handling for training and inference."""
import numpy as np
from PIL import Image, ImageOps


def read_rgb(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            return np.asarray(background)
        return np.asarray(image.convert("RGB"))
