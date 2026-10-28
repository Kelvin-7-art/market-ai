# tests/integration/test_pipeline.py
import numpy as np
from core.pipeline import VisionPipeline
from analyzers.counting import CountingAnalyzer
from analyzers.spatial import SpatialAnalyzer
from tests.fakes import FakeBox


def test_pipeline_with_fake_detector(fake_detection, solid_red):
    p = VisionPipeline()
    p.register(fake_detection([
        FakeBox([10, 10, 60, 60], 0, 0.9),
        FakeBox([150, 150, 200, 200], 1, 0.8),
    ]))
    p.register(CountingAnalyzer())
    p.register(SpatialAnalyzer())

    results = p.run(solid_red)
    assert results["counting"].data["total"] == 2
    assert results["spatial"].data["pairwise"][0]["vertical"] == "above"