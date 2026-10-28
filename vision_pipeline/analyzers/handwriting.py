# analyzers/handwriting.py
from core.base import BaseAnalyzer
from core.result import AnalysisResult, BoundingBox


class HandwritingAnalyzer(BaseAnalyzer):
    name = "handwriting"

    def __init__(self, yolo_conf=0.25, yolo_iou=0.45):
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        import torch

        # Load YOLOv8 handwritten text detector
        yolo_path = hf_hub_download(
            repo_id="armvectores/yolov8n_handwritten_text_detection",
            filename="best.pt",
        )
        self.yolo = YOLO(yolo_path)
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou

        # Load TrOCR for recognition
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _merge_into_lines(self, boxes, y_tolerance=15, x_gap=30):
        """Merge character-level boxes into line-level regions.

        The YOLO model detects individual characters/tokens. TrOCR expects
        full lines. This groups boxes that share a vertical center and
        merges each group horizontally into a single line-level box.
        """
        if not boxes:
            return []

        # Sort by vertical center, then horizontal position
        sorted_boxes = sorted(boxes, key=lambda b: ((b[1] + b[3]) / 2, b[0]))

        lines = []
        current_line = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            x1, y1, x2, y2 = box
            _, cy1, _, cy2 = current_line[-1]
            # Same line if vertical centers are close
            if abs((y1 + y2) / 2 - (cy1 + cy2) / 2) < y_tolerance:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
        lines.append(current_line)

        # Merge each line's boxes horizontally
        merged = []
        for line in lines:
            x1 = min(b[0] for b in line)
            y1 = min(b[1] for b in line)
            x2 = max(b[2] for b in line)
            y2 = max(b[3] for b in line)
            merged.append((x1, y1, x2, y2))

        return merged

    def analyze(self, image, context):
        import cv2
        import numpy as np
        import torch
        from PIL import Image

        # Stage 1: Detect handwritten text regions
        results = self.yolo.predict(
            source=image,
            imgsz=1280,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            return AnalysisResult(self.name, {
                "count": 0,
                "full_text": "",
                "segments": [],
            })

        # Collect raw character-level boxes
        raw_boxes = results[0].boxes.xyxy.cpu().numpy()
        raw_boxes_tuples = [tuple(b[:4]) for b in raw_boxes]

        # Merge into line-level regions (matches TrOCR's expectation)
        merged_boxes = self._merge_into_lines(raw_boxes_tuples)

        segments = []
        all_text = []

        # Stage 2: Crop each line region and transcribe with TrOCR
        for x1, y1, x2, y2 in merged_boxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Guard against empty or inverted regions
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop from original image
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Skip tiny crops — TrOCR can't do anything with them
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            # Convert to PIL for TrOCR
            crop_pil = Image.fromarray(crop).convert("RGB")

            # Transcribe
            pixel_values = self.processor(
                images=crop_pil, return_tensors="pt"
            ).pixel_values.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            if text:
                all_text.append(text)
                segments.append({
                    "text": text,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

        return AnalysisResult(
            self.name,
            {
                "count": len(segments),
                "raw_detections": len(raw_boxes_tuples),
                "merged_regions": len(merged_boxes),
                "full_text": " ".join(all_text),
                "segments": segments,
            },
            boxes=[
                BoundingBox(
                    s["bbox"]["x1"], s["bbox"]["y1"],
                    s["bbox"]["x2"], s["bbox"]["y2"],
                    "handwriting", 1.0,
                )
                for s in segments
            ],
        )