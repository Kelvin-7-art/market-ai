# analyzers/face_detection.py
from pathlib import Path
import numpy as np
from core.base import BaseAnalyzer
from core.result import AnalysisResult, BoundingBox


class FaceDetectionAnalyzer(BaseAnalyzer):
    name = "face_detection"

    def __init__(self, model_dir="models", confidence=0.5):
        import cv2

        model_dir = Path(model_dir)
        prototxt = model_dir / "deploy.prototxt"
        caffemodel = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        if not prototxt.exists() or not caffemodel.exists():
            raise FileNotFoundError(
                f"Face detection models not found in {model_dir}. "
                "Download deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel"
            )

        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        self.confidence = confidence

    def analyze(self, image, context):
        import cv2

        h, w = image.shape[:2]

        # Prepare input blob — note the mean values match the training set
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self.confidence:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = map(int, box)
            boxes.append(BoundingBox(x1, y1, x2, y2, "face", conf))

        return AnalysisResult(self.name, {
            "count": len(boxes),
            "min_confidence": self.confidence,
        }, boxes=boxes)