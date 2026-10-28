# tests/unit/test_barcode_qr.py
import numpy as np
import pytest
from analyzers.barcode_qr import BarcodeQRAnalyzer


@pytest.fixture
def qr_image():
    import qrcode
    qr = qrcode.make("https://example.com/test")
    return np.array(qr.convert("RGB"))


def test_detects_qr_code(qr_image):
    r = BarcodeQRAnalyzer().analyze(qr_image, {})
    assert r.error is None
    assert r.data["count"] >= 1
    codes = r.data["codes"]
    assert any(c["type"] == "QRCODE" for c in codes)
    assert any("example.com" in c["data"] for c in codes)


def test_no_codes_found(solid_red):
    r = BarcodeQRAnalyzer().analyze(solid_red, {})
    assert r.data["count"] == 0