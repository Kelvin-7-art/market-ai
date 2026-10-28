# analyzers/pose_detection.py
import math
from core.base import BaseAnalyzer
from core.result import AnalysisResult, BoundingBox


class PoseDetectionAnalyzer(BaseAnalyzer):
    name = "pose_detection"

    KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    def __init__(self, model="yolo11n-pose.pt", conf=0.5):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf = conf

    def analyze(self, image, context):
        res = self.model.predict(image, conf=self.conf, verbose=False)[0]

        people = []
        boxes = []

        if res.keypoints is None or len(res.keypoints) == 0:
            return AnalysisResult(self.name, {"count": 0, "people": []})

        for i, kpts in enumerate(res.keypoints.data):
            pts = kpts.cpu().numpy()  # shape (17, 3) — x, y, confidence
            person = {"keypoints": {}, "action": "unknown"}

            for j, name in enumerate(self.KEYPOINTS):
                x, y, c = pts[j]
                person["keypoints"][name] = {
                    "x": float(x), "y": float(y), "confidence": float(c),
                }

            person["action"] = self._classify_action(person["keypoints"])
            people.append(person)

            if res.boxes is not None and i < len(res.boxes):
                x1, y1, x2, y2 = map(int, res.boxes[i].xyxy[0].tolist())
                c = float(res.boxes[i].conf[0])
                boxes.append(BoundingBox(x1, y1, x2, y2, "person", c))

        return AnalysisResult(self.name, {
            "count": len(people),
            "people": people,
        }, boxes=boxes)

    def _classify_action(self, kps):
        """Coarse action label from keypoint geometry. Not a trained classifier —
        simple heuristics on shoulder/hip positions."""
        def pt(name):
            k = kps.get(name, {})
            return (k.get("x", 0), k.get("y", 0), k.get("confidence", 0))

        ls, rs = pt("left_shoulder"), pt("right_shoulder")
        lh, rh = pt("left_hip"), pt("right_hip")

        if ls[2] < 0.3 or rs[2] < 0.3 or lh[2] < 0.3 or rh[2] < 0.3:
            return "unknown"

        shoulder_y = (ls[1] + rs[1]) / 2
        hip_y = (lh[1] + rh[1]) / 2
        shoulder_x = (ls[0] + rs[0]) / 2
        hip_x = (lh[0] + rh[0]) / 2

        dy = abs(hip_y - shoulder_y)
        dx = abs(hip_x - shoulder_x)

        if dx > dy * 2:
            return "lying"
        if dy < 50:
            return "sitting"
        return "standing"