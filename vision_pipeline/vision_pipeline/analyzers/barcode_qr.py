# analyzers/barcode_qr.py
from core.base import BaseAnalyzer
from core.result import AnalysisResult, BoundingBox


class BarcodeQRAnalyzer(BaseAnalyzer):
    name = "barcode_qr"

    def __init__(self, min_confidence=0.0):
        # pyzbar doesn't expose confidence, kept for API consistency
        self.min_confidence = min_confidence

    def analyze(self, image, context):
        from pyzbar.pyzbar import decode
        import cv2

        # pyzbar expects grayscale or BGR; convert from RGB
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        decoded = decode(gray)
        results, boxes = [], []

        for obj in decoded:
            data = obj.data.decode("utf-8", errors="replace")
            code_type = obj.type  # e.g. "QRCODE", "CODE128", "EAN13"

            # pyzbar gives rect as (x, y, w, h)
            x, y, w, h = obj.rect
            boxes.append(BoundingBox(
                int(x), int(y), int(x + w), int(y + h),
                code_type, 1.0,  # pyzbar has no confidence
            ))
            results.append({
                "type": code_type,
                "data": data,
                "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            })

        return AnalysisResult(self.name, {
            "count": len(results),
            "codes": results,
        }, boxes=boxes)