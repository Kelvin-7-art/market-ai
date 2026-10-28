# analyzers/ocr.py
from core.base import BaseAnalyzer
from core.result import AnalysisResult, BoundingBox


class OCRAnalyzer(BaseAnalyzer):
    name = "ocr"

    def __init__(self, langs=("en",), min_confidence=0.4):
        import easyocr
        self.reader = easyocr.Reader(list(langs), gpu=False)
        self.min_confidence = min_confidence

    def analyze(self, image, context):
        raw = self.reader.readtext(image)

        texts, boxes = [], []
        for poly, text, conf in raw:
            conf = float(conf)
            if conf < self.min_confidence:
                continue

            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))

            boxes.append(BoundingBox(x1, y1, x2, y2, "text", conf))
            texts.append({
                "text": text,
                "confidence": conf,
                "box": [x1, y1, x2, y2],   # ← NEW: box coordinates per segment
            })

        # Sort top-to-bottom, left-to-right — matches reading order.
        # EasyOCR usually already returns this, but not guaranteed.
        paired = sorted(
            zip(boxes, texts),
            key=lambda pair: (pair[0].y1, pair[0].x1),
        )
        boxes = [b for b, _ in paired]
        texts = [t for _, t in paired]

        full_text = " ".join(t["text"].strip() for t in texts if t["text"].strip())

        # ── NEW: share segments with downstream analyzers ────────
        # ReceiptParserAnalyzer reads this to get words + boxes.
        # Must match the order of `boxes` (same sort applied above).
        context["ocr_segments"] = texts

        return AnalysisResult(
            self.name,
            {
                "full_text": full_text,
                "segments": texts,
                "segment_count": len(texts),
                "min_confidence": self.min_confidence,
            },
            boxes=boxes,
        )