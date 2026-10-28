# analyzers/color.py
import cv2, numpy as np
from collections import Counter
from core.base import BaseAnalyzer
from core.result import AnalysisResult

class ColorAnalyzer(BaseAnalyzer):
    name = "color"

    def __init__(self, k=5):
        self.k = k

    def analyze(self, image, context):
        # Dominant colors via k-means on downsampled pixels
        small = cv2.resize(image, (100, 100)).reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(small, self.k, None, criteria, 5,
                                        cv2.KMEANS_PP_CENTERS)
        counts = Counter(labels.flatten().tolist())
        total = sum(counts.values())
        dominant = []
        for idx, count in counts.most_common():
            r, g, b = [int(c) for c in centers[idx]]
            dominant.append({
                "hex": f"#{r:02x}{g:02x}{b:02x}",
                "rgb": (r, g, b),
                "share": round(count / total, 3),
                "name": self._name_color(r, g, b),
            })

        # Overall stats
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return AnalysisResult(self.name, {
            "dominant_colors": dominant,
            "mean_rgb": [int(x) for x in image.reshape(-1, 3).mean(axis=0)],
            "mean_saturation": round(float(hsv[..., 1].mean()), 2),
            "is_grayscale": float(np.abs(image[..., 0].astype(int) -
                                          image[..., 2].astype(int)).mean()) < 5,
        })

    @staticmethod
    def _name_color(r, g, b):
        import colorsys
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        if v < 0.15: return "black"
        if v > 0.9 and s < 0.1: return "white"
        if s < 0.12: return "gray"
        h *= 360
        for name, (lo, hi) in {
            "red": (0, 15), "orange": (15, 45), "yellow": (45, 70),
            "green": (70, 165), "cyan": (165, 195), "blue": (195, 260),
            "purple": (260, 300), "magenta": (300, 345), "red2": (345, 360),
        }.items():
            if lo <= h < hi:
                return "red" if name == "red2" else name
        return "unknown"