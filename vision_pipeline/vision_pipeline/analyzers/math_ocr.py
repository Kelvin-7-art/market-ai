# analyzers/math_ocr.py — enhanced preprocessing
from core.base import BaseAnalyzer
from core.result import AnalysisResult, BoundingBox


class MathOCRAnalyzer(BaseAnalyzer):
    name = "math_ocr"

    def __init__(self):
        from pix2tex.cli import LatexOCR
        self.model = LatexOCR()

    def _preprocess(self, pil_img):
        """Normalize to black-on-white, then threshold to remove anti-aliasing."""
        import numpy as np
        from PIL import Image, ImageOps

        # Step 1: Invert if dark background
        arr = np.array(pil_img.convert("L"))
        mean_luminance = float(arr.mean())

        if mean_luminance < 128:
            pil_img = ImageOps.invert(pil_img.convert("RGB"))

        # Step 2: Threshold to pure black/white (removes anti-aliasing)
        gray = np.array(pil_img.convert("L"))
        thresholded = ((gray > 128) * 255).astype(np.uint8)
        pil_img = Image.fromarray(thresholded).convert("RGB")

        return pil_img, mean_luminance < 128

    def analyze(self, image, context):
        import numpy as np
        from PIL import Image

        pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        pil_img = pil_img.convert("RGB")

        pil_img, was_inverted = self._preprocess(pil_img)

        try:
            latex = self.model(pil_img)
        except Exception as e:
            return AnalysisResult(self.name, error=f"pix2tex failed: {e}")

        h, w = pil_img.size[1], pil_img.size[0]

        if not latex or not latex.strip():
            return AnalysisResult(self.name, {
                "latex": "",
                "formula_count": 0,
                "formulas": [],
                "inverted": was_inverted,
            })

        box = BoundingBox(0, 0, w, h, "formula", 1.0)

        return AnalysisResult(
            self.name,
            {
                "latex": latex,
                "formula_count": 1,
                "formulas": [{
                    "latex": latex,
                    "bbox": {"x1": 0, "y1": 0, "x2": w, "y2": h},
                }],
                "inverted": was_inverted,
            },
            boxes=[box],
        )