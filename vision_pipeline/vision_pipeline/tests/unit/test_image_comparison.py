# tests/unit/test_image_comparison.py
import numpy as np
import pytest
from analyzers.image_comparison import ImageComparisonAnalyzer


@pytest.mark.slow
def test_identical_images_high_similarity(solid_red):
    r = ImageComparisonAnalyzer().analyze(
        solid_red, {"reference_image": solid_red.copy()}
    )
    assert r.data["clip_similarity"] > 0.95
    assert r.data["ssim"] > 0.99


@pytest.mark.slow
def test_different_images_lower_similarity(solid_red, sharp_text_image):
    r = ImageComparisonAnalyzer().analyze(
        solid_red, {"reference_image": sharp_text_image}
    )
    # Different content should not score as "very similar"
    assert r.data["clip_similarity"] < 0.95
    # SSIM will be low because shapes differ; resize makes it valid
    assert r.data["ssim"] is not None


def test_ssim_handles_different_dimensions(solid_red, sharp_text_image):
    r = ImageComparisonAnalyzer().analyze(
        solid_red, {"reference_image": sharp_text_image}
    )
    assert "resized" in r.data["ssim_note"] or "same" in r.data["ssim_note"]