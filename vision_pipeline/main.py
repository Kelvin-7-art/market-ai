# main.py
import cv2, numpy as np
from PIL import Image
from core.pipeline import VisionPipeline
from analyzers.detection import DetectionAnalyzer
from analyzers.counting import CountingAnalyzer
from analyzers.ocr import OCRAnalyzer
from analyzers.classification import ClassificationAnalyzer
from analyzers.quality import QualityAnalyzer
from analyzers.color import ColorAnalyzer
from analyzers.metadata import MetadataAnalyzer
from analyzers.spatial import SpatialAnalyzer
import json

def build_pipeline(with_ml=True):
    p = VisionPipeline()
    p.register(MetadataAnalyzer())
    p.register(QualityAnalyzer())
    p.register(ColorAnalyzer())
    if with_ml:
        p.register(DetectionAnalyzer())
        p.register(CountingAnalyzer())
        p.register(OCRAnalyzer())
        p.register(ClassificationAnalyzer())
        p.register(SpatialAnalyzer())
    return p

def analyze_image(path, with_ml=True):
    pil = Image.open(path).convert("RGB")
    img = np.array(pil)
    pipe = build_pipeline(with_ml)
    # stash PIL on a context-carrying dict:
    results = {}
    context = {"pil_image": pil}
    for a in pipe.analyzers:
        try:
            r = a.analyze(img, context)
            results[a.name] = r
            if r.boxes:
                context.setdefault("boxes", []).extend(r.boxes)
        except Exception as e:
            results[a.name] = f"ERROR: {e}"
    return results

if __name__ == "__main__":
    import sys
    out = analyze_image(sys.argv[1])
    for name, r in out.items():
        print(f"\n=== {name} ===")
        if hasattr(r, "data"):
            print(json.dumps(r.data, indent=2, default=str))
        else:
            print(r)