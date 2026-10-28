"""Shared fixtures and fakes for the vision pipeline tests."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from analyzers.detection import DetectionAnalyzer


# ── Fake model classes ──────────────────────────────────────
class _FakeTensor:
    def __init__(self, values):
        self._values = values
    def tolist(self):
        return list(self._values)


class _FakeBox:
    def __init__(self, xyxy, cls, conf):
        self.xyxy = [_FakeTensor(xyxy)]
        self.cls = [cls]
        self.conf = [conf]


class _FakeResult:
    names = {0: "person", 1: "car"}
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeYOLO:
    def __init__(self, boxes):
        self._boxes = boxes
    def predict(self, *a, **kw):
        return [_FakeResult(self._boxes)]


class _FakeOCRReader:
    """Mimics easyocr.Reader.readtext() — returns (poly, text, conf) tuples."""
    def __init__(self, segments):
        self._segments = segments   # list of ((x1, y1, x2, y2), text, conf)
    def readtext(self, img):
        out = []
        for (x1, y1, x2, y2), text, conf in self._segments:
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            out.append((poly, text, conf))
        return out


# ── Image fixtures ──────────────────────────────────────────
@pytest.fixture
def solid_red():
    return np.full((200, 200, 3), [255, 0, 0], dtype=np.uint8)


@pytest.fixture
def solid_blue():
    return np.full((200, 200, 3), [0, 0, 255], dtype=np.uint8)


@pytest.fixture
def grayscale_noise():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    return np.stack([g, g, g], axis=-1)


@pytest.fixture
def sharp_text_image():
    img = Image.new("RGB", (400, 200), "white")
    d = ImageDraw.Draw(img)
    for i in range(5):
        d.rectangle([40 + i * 70, 60, 80 + i * 70, 140], fill="black")
    return np.array(img)


@pytest.fixture
def blurry_image(sharp_text_image):
    import cv2
    return cv2.GaussianBlur(sharp_text_image, (31, 31), 0)


@pytest.fixture
def pil_from(sharp_text_image):
    return Image.fromarray(sharp_text_image)


# ── Phase 2 fixtures ────────────────────────────────────────
@pytest.fixture
def qr_image():
    """A scannable QR code as an RGB numpy array."""
    import qrcode
    qr = qrcode.make("https://example.com/test")
    return np.array(qr.convert("RGB"))


@pytest.fixture
def barcode_image():
    """A scannable EAN-13 barcode as an RGB numpy array."""
    import barcode
    from barcode.writer import ImageWriter
    import io
    ean = barcode.get("ean13", "590123412345", writer=ImageWriter())
    buf = io.BytesIO()
    ean.write(buf)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


@pytest.fixture
def white_noise_image():
    """High-entropy image for embedding/comparison tests."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)


# ── Analyzer factories (bypass __init__ to skip model loading) ──
@pytest.fixture
def fake_detection():
    """Factory: fake_detection(boxes) → DetectionAnalyzer with a stub YOLO."""
    def _make(boxes):
        a = DetectionAnalyzer.__new__(DetectionAnalyzer)
        a.conf = 0.35
        a.model = _FakeYOLO(boxes)
        return a
    return _make


@pytest.fixture
def fake_ocr():
    """Factory: fake_ocr(segments) → OCRAnalyzer with a stub reader.

    segments: list of ((x1, y1, x2, y2), "text", confidence)
    """
    from analyzers.ocr import OCRAnalyzer
    def _make(segments, min_confidence=0.4):
        a = OCRAnalyzer.__new__(OCRAnalyzer)
        a.reader = _FakeOCRReader(segments)
        a.min_confidence = min_confidence
        return a
    return _make