# analyzers/segmentation.py
from core.base import BaseAnalyzer
from core.result import AnalysisResult, BoundingBox


class SegmentationAnalyzer(BaseAnalyzer):
    name = "segmentation"

    def __init__(self, model="yolo11n-seg.pt", conf=0.4):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf = conf

    def analyze(self, image, context):
        res = self.model.predict(image, conf=self.conf, verbose=False)[0]

        segments = []
        boxes = []

        for i, b in enumerate(res.boxes):
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cls_idx = int(b.cls[0])
            label = res.names[cls_idx]
            confidence = float(b.conf[0])

            # Mask pixel count — how many pixels belong to this object
            mask_pixels = 0
            if res.masks is not None and i < len(res.masks.data):
                mask = res.masks.data[i].cpu().numpy()
                mask_pixels = int((mask > 0.5).sum())

            segments.append({
                "label": label,
                "confidence": confidence,
                "mask_pixels": mask_pixels,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })
            boxes.append(BoundingBox(x1, y1, x2, y2, label, confidence))

        return AnalysisResult(self.name, {
            "count": len(segments),
            "total_mask_pixels": sum(s["mask_pixels"] for s in segments),
            "segments": segments,
        }, boxes=boxes)