# tests/unit/test_metadata.py
from PIL import Image
from analyzers.metadata import MetadataAnalyzer

def test_no_pil_in_context(sharp_text_image):
    r = MetadataAnalyzer().analyze(sharp_text_image, {})
    assert "note" in r.data

def test_reads_dimensions(sharp_text_image):
    pil = Image.fromarray(sharp_text_image)
    r = MetadataAnalyzer().analyze(sharp_text_image, {"pil_image": pil})
    assert r.data["width"] == 400
    assert r.data["height"] == 200
    assert r.data["mode"] == "RGB"