# tests/unit/test_face_detection.py
import numpy as np
import pytest
from analyzers.face_detection import FaceDetectionAnalyzer


@pytest.mark.slow
def test_face_detection_on_bus_image():
    """The bus.jpg fixture contains 3 people — expect at least 1 face."""
    import cv2
    img = cv2.imread("tests/fixtures/real/bus.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    a = FaceDetectionAnalyzer()
    r = a.analyze(img, {})

    assert r.error is None
    assert r.data["count"] >= 1
    for b in r.boxes:
        assert b.label == "face"
        assert 0.0 <= b.confidence <= 1.0


def test_face_detection_rejects_missing_models():
    with pytest.raises(FileNotFoundError):
        FaceDetectionAnalyzer(model_dir="nonexistent")