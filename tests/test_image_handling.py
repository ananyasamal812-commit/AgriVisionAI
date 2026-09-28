import tempfile
import unittest
from pathlib import Path
from PIL import Image
from src.inference.images import read_rgb


class ImageHandlingTests(unittest.TestCase):
    def test_exif_rotation_and_unicode_filename(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "leaf-é.jpg"
            image = Image.new("RGB", (180, 120), "green")
            exif = image.getexif()
            exif[274] = 6
            image.save(path, exif=exif)
            self.assertEqual(read_rgb(path).shape, (180, 120, 3))

    def test_transparency_uses_white_background(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "transparent.png"
            Image.new("RGBA", (20, 30), (0, 0, 0, 0)).save(path)
            self.assertTrue((read_rgb(path) == 255).all())
