# tests/e2e/test_real_models.py
import numpy as np
import pytest
from PIL import Image
from pathlib import Path

FIXTURE = Path(__file__).parent.parent / "fixtures" / "real" / "bus.jpg"

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"missing {FIXTURE}")
def test_real_yolo_detects_objects():
    from analyzers.detection import DetectionAnalyzer
    img = np.array(Image.open(FIXTURE).convert("RGB"))
    r = DetectionAnalyzer(conf=0.4).analyze(img, {})
    assert r.error is None
    assert r.data["count"] > 0
    # bus.jpg famously contains a bus + people
    labels = {b.label for b in r.boxes}
    assert labels & {"bus", "person"}, f"expected bus/person, got {labels}"


@pytest.mark.skipif(not FIXTURE.exists(), reason=f"missing {FIXTURE}")
def test_real_pipeline_end_to_end():
    from core.pipeline import VisionPipeline
    from analyzers.detection import DetectionAnalyzer
    from analyzers.counting import CountingAnalyzer
    from analyzers.spatial import SpatialAnalyzer
    from analyzers.quality import QualityAnalyzer
    from analyzers.color import ColorAnalyzer

    img = np.array(Image.open(FIXTURE).convert("RGB"))

    p = VisionPipeline()
    p.register(DetectionAnalyzer(conf=0.4))
    p.register(CountingAnalyzer())
    p.register(SpatialAnalyzer())
    p.register(QualityAnalyzer())
    p.register(ColorAnalyzer())

    results = p.run(img)

    # Every analyzer should succeed
    for name, r in results.items():
        assert r.error is None, f"{name}: {r.error}"

    # Cross-analyzer consistency
    assert results["counting"].data["total"] == results["detection"].data["count"]