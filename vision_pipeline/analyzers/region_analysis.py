# analyzers/region_analysis.py
from core.base import BaseAnalyzer
from core.result import AnalysisResult


class RegionAnalysisAnalyzer(BaseAnalyzer):
    name = "region_analysis"

    def __init__(self, analyzers=None):
        """analyzers: list of analyzer instances to run on the crop.

        If None, defaults to quality + color + detection + OCR
        (cheap + useful for "what's in this region?").
        """
        self.analyzers = analyzers or []

    def analyze(self, image, context):
        # Region must be supplied via context
        region = context.get("region")
        if not region:
            return AnalysisResult(
                self.name,
                error="No region specified in context (expected {x1, y1, x2, y2})",
            )

        x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
        h, w = image.shape[:2]

        # Clamp to image bounds — Pillow/OpenCV both require valid rectangles
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        if x1 >= x2 or y1 >= y2:
            return AnalysisResult(
                self.name,
                error=f"Invalid region after clamping: ({x1},{y1})-({x2},{y2})",
            )

        # Crop — OpenCV uses row-first slicing: image[y1:y2, x1:x2]
        crop = image[y1:y2, x1:x2]

        # Run analyzers on the crop
        results = {}
        crop_context = {"region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}

        for a in self.analyzers:
            try:
                r = a.analyze(crop, crop_context)
                results[a.name] = {
                    "data": r.data,
                    "boxes": [
                        {"x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2,
                         "label": b.label, "confidence": b.confidence}
                        for b in r.boxes
                    ],
                    "error": r.error,
                }
                if r.boxes:
                    crop_context.setdefault("boxes", []).extend(r.boxes)
            except Exception as e:
                results[a.name] = {"error": str(e)}

        return AnalysisResult(self.name, {
            "region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "crop_size": {"width": x2 - x1, "height": y2 - y1},
            "results": results,
        })