# analyzers/counting.py
from collections import Counter
from core.base import BaseAnalyzer
from core.result import AnalysisResult


class CountingAnalyzer(BaseAnalyzer):
    name = "counting"

    def __init__(self, iou_threshold=0.7):
        self.iou_threshold = iou_threshold

    def analyze(self, image, context):
        boxes = context.get("boxes", [])
        deduped = self._dedupe(boxes)
        counts = Counter(b.label for b in deduped)
        return AnalysisResult(self.name, {
            "total": len(deduped),
            "by_class": dict(counts),
        })

    def _dedupe(self, boxes):
        """Remove boxes that heavily overlap higher-priority boxes with the same label."""
        priority = {
            "person": 0,
            "car": 1, "bus": 1, "truck": 1, "bicycle": 1, "motorcycle": 1,
            "face": 5, "text": 6, "QRCODE": 7,
        }
        sorted_boxes = sorted(
            boxes,
            key=lambda b: (priority.get(b.label, 3), -b.confidence),
        )
        kept = []
        for b in sorted_boxes:
            overlaps = any(
                b.label == k.label and self._iou(b, k) > self.iou_threshold
                for k in kept
            )
            if not overlaps:
                kept.append(b)
        return kept

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
        ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return inter / union if union > 0 else 0.0