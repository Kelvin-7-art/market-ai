# tests/unit/test_counting.py
from core.result import BoundingBox
from analyzers.counting import CountingAnalyzer

def test_counts_by_class():
    boxes = [
        BoundingBox(0, 0, 10, 10, "person", 0.9),
        BoundingBox(0, 0, 10, 10, "person", 0.8),
        BoundingBox(0, 0, 10, 10, "car", 0.7),
    ]
    r = CountingAnalyzer().analyze(None, {"boxes": boxes})
    assert r.data["total"] == 3
    assert r.data["by_class"] == {"person": 2, "car": 1}

def test_empty():
    r = CountingAnalyzer().analyze(None, {})
    assert r.data["total"] == 0