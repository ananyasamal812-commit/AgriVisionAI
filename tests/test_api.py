"""Integration checks against a real model and a temporary local API server."""
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        cls.base = f"http://127.0.0.1:{port}"
        env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE="1", OMP_NUM_THREADS="2", ENABLE_EVALUATION="true")
        cls.log = open(ROOT / "outputs/api-test.log", "w", encoding="utf-8")
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=env, stdout=cls.log, stderr=subprocess.STDOUT)
        cls.addClassCleanup(cls.stop_server)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if cls.server.poll() is not None:
                raise RuntimeError("API startup failed. See outputs/api-test.log")
            try:
                with urlopen(cls.base + "/health", timeout=2):
                    return
            except OSError:
                time.sleep(1)
        raise RuntimeError("API startup timed out. See outputs/api-test.log")

    @classmethod
    def stop_server(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            cls.server.kill()
            cls.server.wait()
        cls.log.close()

    def upload(self, data, name="leaf.png"):
        boundary = "agrivision-test-boundary"
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
            'Content-Type: image/png\r\n\r\n'
        ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
        request = Request(self.base + "/predict", data=body,
                          headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            response = urlopen(request, timeout=120)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_health(self):
        with urlopen(self.base + "/health") as response:
            self.assertEqual(json.load(response)["classes"], 38)

    def test_evaluation_and_missing_report(self):
        with urlopen(self.base + "/evaluation") as response:
            result = json.load(response)
            self.assertEqual(len(result["classes"]), 38)
            self.assertIsNone(result["test_accuracy"])
            self.assertIn("validation", result["split"])
        with self.assertRaises(HTTPError) as error:
            urlopen(self.base + "/reports/not-a-report")
        self.assertEqual(error.exception.code, 404)

    def test_reject_invalid_empty_and_oversized_uploads(self):
        for data, status in [(b"not an image", 400), (b"", 400), (b"x" * (10 * 1024 * 1024 + 1), 413)]:
            with self.subTest(status=status, length=len(data)):
                actual, result = self.upload(data)
                self.assertEqual(actual, status, result)

    def test_real_prediction_heatmap_and_cleanup(self):
        sample = next(p for p in (ROOT / "sample_images").rglob("*")
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        buffer = io.BytesIO()
        with Image.open(sample) as leaf:
            leaf.convert("RGB").resize((320, 180)).save(buffer, format="PNG")
        before = set((ROOT / "uploads").glob("*"))
        heatmaps = []
        reports = []
        try:
            for _ in range(2):
                status, result = self.upload(buffer.getvalue(), "../../leaf.png")
                self.assertEqual(status, 200, result)
                self.assertTrue(0 <= result["confidence"] <= 100)
                self.assertIn("disease", result)
                self.assertIsInstance(result["uncertain"], bool)
                self.assertIsInstance(result["warnings"], list)
                top = result["top_predictions"]
                self.assertEqual(len(top), 3)
                self.assertEqual(len({item["class_index"] for item in top}), 3)
                self.assertEqual(top[0]["disease"], result["disease"])
                self.assertEqual([item["confidence"] for item in top], sorted([item["confidence"] for item in top], reverse=True))
                reports.append(ROOT / "outputs/reports" / (result["report_id"] + ".pdf"))
                reports.append(ROOT / "outputs/reports" / (result["report_id"] + ".hi.pdf"))
                with urlopen(self.base + "/" + result["report_url"] + "?language=hi") as response:
                    self.assertIn("-hi.pdf", response.headers["Content-Disposition"])
                    hindi_pdf = response.read()
                    self.assertTrue(hindi_pdf.startswith(b"%PDF"))
                    self.assertIn(b"NotoSansDevanagari", hindi_pdf)
                    (ROOT / "outputs/hindi-report-preview.pdf").write_bytes(hindi_pdf)
                with urlopen(self.base + "/" + result["report_url"]) as response:
                    self.assertIn("attachment", response.headers["Content-Disposition"])
                    pdf = response.read()
                    reader = PdfReader(io.BytesIO(pdf))
                    text = " ".join(page.extract_text() for page in reader.pages)
                    self.assertIn("Top 3 predictions", text)
                    self.assertIn(result["report_id"], text)
                    self.assertTrue(1 <= len(reader.pages) <= 3)
                    (ROOT / "outputs/report-preview.pdf").write_bytes(pdf)
                path = result["gradcam"]
                self.assertTrue(path.startswith("outputs/heatmaps/"))
                heatmaps.append(path)
                with urlopen(self.base + "/" + path) as response:
                    with Image.open(io.BytesIO(response.read())) as heatmap:
                        self.assertEqual(heatmap.size, (320, 180))
            self.assertNotEqual(*heatmaps)
            self.assertEqual(before, set((ROOT / "uploads").glob("*")))
        finally:
            for path in heatmaps:
                (ROOT / path).unlink(missing_ok=True)
            for path in reports:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
