# tests/unit/test_color.py
import numpy as np
from analyzers.color import ColorAnalyzer

def test_dominant_color_is_red(solid_red):
    r = ColorAnalyzer(k=3).analyze(solid_red, {})
    top = r.data["dominant_colors"][0]
    assert top["name"] == "red"
    assert top["share"] > 0.9

def test_grayscale_detection(grayscale_noise):
    r = ColorAnalyzer().analyze(grayscale_noise, {})
    assert r.data["is_grayscale"] is True

def test_color_names(solid_red):
    r = ColorAnalyzer(k=1).analyze(solid_red, {})
    assert r.data["dominant_colors"][0]["hex"] == "#ff0000"