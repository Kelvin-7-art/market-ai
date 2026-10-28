# tests/unit/test_detection.py
import numpy as np
import pytest
from tests.fakes import FakeBox


def test_detection_parses_boxes(fake_detection):
    boxes = [
        FakeBox([10, 20, 50, 80], 0, 0.91),
        FakeBox([100, 120, 200, 220], 1, 0.72),
    ]
    r = fake_detection(boxes).analyze(
        np.zeros((300, 300, 3), dtype=np.uint8), {}
    )
    assert r.data["count"] == 2
    assert {b.label for b in r.boxes} == {"person", "car"}
    assert r.boxes[0].confidence == pytest.approx(0.91)


def test_detection_empty(fake_detection):
    r = fake_detection([]).analyze(
        np.zeros((300, 300, 3), dtype=np.uint8), {}
    )
    assert r.data["count"] == 0
    assert r.boxes == []