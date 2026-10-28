# analyzers/detection.py
import numpy as np
from core.base import BaseAnalyzer
from core.result import AnalysisResult, BoundingBox


class DetectionAnalyzer(BaseAnalyzer):
    name = "detection"

    def __init__(self, model="yolov8n.pt", conf=0.35, iou=0.5, classes=None):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf = conf
        self.iou = iou
        self.classes = classes

    def analyze(self, image, context):
        res = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            verbose=False,
        )[0]

        boxes = []
        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cls_idx = int(b.cls[0])
            confidence = float(b.conf[0])
            label = res.names[cls_idx]
            boxes.append(BoundingBox(x1, y1, x2, y2, label, confidence))

        return AnalysisResult(self.name, {"count": len(boxes)}, boxes=boxes)