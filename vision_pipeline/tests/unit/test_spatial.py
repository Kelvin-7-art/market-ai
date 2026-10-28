# tests/unit/test_spatial.py
import numpy as np
from core.result import BoundingBox
from analyzers.spatial import SpatialAnalyzer

def test_left_right_above_below():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    a = BoundingBox(10, 10, 40, 40, "a", 0.9)     # top-left
    b = BoundingBox(200, 200, 240, 240, "b", 0.9) # bottom-right
    r = SpatialAnalyzer().analyze(img, {"boxes": [a, b]})
    rel = r.data["pairwise"][0]
    assert rel["horizontal"] == "left_of"
    assert rel["vertical"] == "above"
    assert rel["iou"] == 0.0

def test_position_labels():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    a = BoundingBox(10, 10, 40, 40, "a", 0.9)  # top-left
    r = SpatialAnalyzer().analyze(img, {"boxes": [a]})
    assert r.data["positions"][0]["position"] == "top-left"

def test_overlap_iou():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    a = BoundingBox(0, 0, 10, 10, "a", 1.0)
    b = BoundingBox(0, 0, 10, 10, "b", 1.0)
    r = SpatialAnalyzer().analyze(img, {"boxes": [a, b]})
    assert r.data["pairwise"][0]["iou"] == 1.0