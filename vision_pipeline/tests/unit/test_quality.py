# tests/unit/test_quality.py
from analyzers.quality import QualityAnalyzer

def test_sharp_beats_blurry(sharp_text_image, blurry_image):
    q = QualityAnalyzer()
    sharp = q.analyze(sharp_text_image, {}).data["sharpness"]
    blur  = q.analyze(blurry_image, {}).data["sharpness"]
    assert sharp > blur * 3

def test_blurry_flag(blurry_image):
    r = QualityAnalyzer().analyze(blurry_image, {})
    assert r.data["is_blurry"] is True

def test_brightness_range(sharp_text_image):
    r = QualityAnalyzer().analyze(sharp_text_image, {})
    assert 0 <= r.data["brightness"] <= 255
    assert 0 <= r.data["overexposed_ratio"] <= 1