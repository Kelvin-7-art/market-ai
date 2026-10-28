# tests/unit/test_region_analysis.py
import numpy as np
from analyzers.region_analysis import RegionAnalysisAnalyzer
from analyzers.color import ColorAnalyzer


def test_region_crops_correctly(solid_red):
    a = RegionAnalysisAnalyzer(analyzers=[ColorAnalyzer()])
    r = a.analyze(solid_red, {"region": {"x1": 50, "y1": 50, "x2": 150, "y2": 150}})
    assert r.error is None
    assert r.data["crop_size"] == {"width": 100, "height": 100}
    assert "color" in r.data["results"]


def test_region_missing(solid_red):
    a = RegionAnalysisAnalyzer(analyzers=[ColorAnalyzer()])
    r = a.analyze(solid_red, {})
    assert r.error is not None
    assert "No region" in r.error


def test_region_clamps_out_of_bounds(solid_red):
    a = RegionAnalysisAnalyzer(analyzers=[ColorAnalyzer()])
    r = a.analyze(solid_red, {"region": {"x1": -100, "y1": -100, "x2": 999, "y2": 999}})
    # 200x200 image, region should clamp to full image
    assert r.data["crop_size"] == {"width": 200, "height": 200}


def test_region_invalid_after_clamp(solid_red):
    a = RegionAnalysisAnalyzer(analyzers=[ColorAnalyzer()])
    # x1 > x2 → empty region
    r = a.analyze(solid_red, {"region": {"x1": 150, "y1": 50, "x2": 50, "y2": 150}})
    assert r.error is not None
    assert "Invalid region" in r.error