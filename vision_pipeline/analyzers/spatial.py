from core.base import BaseAnalyzer
from core.result import AnalysisResult


class SpatialAnalyzer(BaseAnalyzer):
    name = "spatial"

    def analyze(self, image, context):
        boxes = context.get("boxes", [])
        h, w = image.shape[:2]
        relations = []
        positions = []

        for i, a in enumerate(boxes):
            cx, cy = a.center
            pos_h = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
            pos_v = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"
            positions.append({
                "label": a.label,
                "position": f"{pos_v}-{pos_h}",
                "center": (int(cx), int(cy)),
            })

            for b in boxes[i + 1:]:
                bx, by = b.center
                dx = bx - cx
                dy = by - cy
                dist = float((dx * dx + dy * dy) ** 0.5)

                ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
                ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = a.area + b.area - inter
                iou = float(inter / union) if union > 0 else 0.0

                if dx > 20:
                    horizontal = "left_of"
                elif dx < -20:
                    horizontal = "right_of"
                else:
                    horizontal = "aligned_h"

                if dy > 20:
                    vertical = "above"
                elif dy < -20:
                    vertical = "below"
                else:
                    vertical = "aligned_v"

                relations.append({
                    "a": a.label,
                    "b": b.label,
                    "distance_px": round(dist, 1),
                    "iou": round(iou, 3),
                    "horizontal": horizontal,
                    "vertical": vertical,
                })

        return AnalysisResult(self.name, {
            "frame_size": (w, h),
            "positions": positions,
            "pairwise": relations[:100],
        })
