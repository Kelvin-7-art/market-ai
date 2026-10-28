# analyzers/quality.py
import cv2
import numpy as np
from core.base import BaseAnalyzer
from core.result import AnalysisResult


class QualityAnalyzer(BaseAnalyzer):
    name = "quality"

    def analyze(self, image, context):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Sharpness: variance of Laplacian
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Brightness: mean luminance
        brightness = float(gray.mean())

        # Contrast: std of luminance
        contrast = float(gray.std())

        # Noise: estimated via median filter residual
        noise = float(
            np.abs(gray.astype(np.int16) - cv2.medianBlur(gray, 3).astype(np.int16)).mean()
        )

        # Exposure clipping
        over = float((gray > 245).mean())
        under = float((gray < 10).mean())

        return AnalysisResult(self.name, {
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "noise_estimate": round(noise, 2),
            "overexposed_ratio": round(over, 4),
            "underexposed_ratio": round(under, 4),
            "is_blurry": bool(sharpness < 100),
        })