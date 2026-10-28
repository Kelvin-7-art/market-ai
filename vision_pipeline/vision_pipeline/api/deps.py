# api/deps.py
import os
from functools import lru_cache

from core.pipeline import VisionPipeline

from analyzers.metadata import MetadataAnalyzer
from analyzers.quality import QualityAnalyzer
from analyzers.color import ColorAnalyzer
from analyzers.detection import DetectionAnalyzer
from analyzers.segmentation import SegmentationAnalyzer
from analyzers.counting import CountingAnalyzer
from analyzers.ocr import OCRAnalyzer
from analyzers.receipt_parser import ReceiptParserAnalyzer
from analyzers.table_extractor import TableExtractorAnalyzer
from analyzers.handwriting import HandwritingAnalyzer
from analyzers.math_ocr import MathOCRAnalyzer
from analyzers.doc_layout import DocLayoutAnalyzer
from analyzers.charts import ChartsAnalyzer                  # ← NEW
from analyzers.barcode_qr import BarcodeQRAnalyzer
from analyzers.face_detection import FaceDetectionAnalyzer
from analyzers.pose_detection import PoseDetectionAnalyzer
from analyzers.classification import ClassificationAnalyzer
from analyzers.embeddings import EmbeddingsAnalyzer
from analyzers.spatial import SpatialAnalyzer


# Config
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.3"))
DETECTION_CONFIDENCE = float(os.getenv("DETECTION_CONFIDENCE", "0.35"))
DETECTION_IOU = float(os.getenv("DETECTION_IOU", "0.5"))
SEGMENTATION_CONFIDENCE = float(os.getenv("SEGMENTATION_CONFIDENCE", "0.4"))
FACE_CONFIDENCE = float(os.getenv("FACE_CONFIDENCE", "0.5"))
POSE_CONFIDENCE = float(os.getenv("POSE_CONFIDENCE", "0.5"))
HANDWRITING_CONFIDENCE = float(os.getenv("HANDWRITING_CONFIDENCE", "0.25"))
HANDWRITING_IOU = float(os.getenv("HANDWRITING_IOU", "0.45"))
DOC_LAYOUT_MODEL = os.getenv("DOC_LAYOUT_MODEL", "doclayout_docstructbench")
CHARTS_ENABLED = os.getenv("CHARTS_ENABLED", "1") == "1"     # ← NEW

DETECTION_CLASSES = [0, 1, 2, 3, 5, 7]


@lru_cache(maxsize=1)
def get_clip_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("clip-ViT-B-32")


@lru_cache(maxsize=1)
def get_pipeline() -> VisionPipeline:
    p = VisionPipeline()

    # File & pixel-level
    p.register(MetadataAnalyzer())
    p.register(QualityAnalyzer())
    p.register(ColorAnalyzer())

    # Object detection
    p.register(DetectionAnalyzer(
        conf=DETECTION_CONFIDENCE,
        iou=DETECTION_IOU,
        classes=DETECTION_CLASSES,
    ))
    p.register(SegmentationAnalyzer(conf=SEGMENTATION_CONFIDENCE))
    p.register(CountingAnalyzer())

    # Text & document intelligence
    p.register(OCRAnalyzer(min_confidence=OCR_MIN_CONFIDENCE))
    p.register(ReceiptParserAnalyzer())          # Donut — image → structured JSON
    p.register(TableExtractorAnalyzer())         # TableFormer — line-item grids
    p.register(HandwritingAnalyzer(              # YOLO + TrOCR
        yolo_conf=HANDWRITING_CONFIDENCE,
        yolo_iou=HANDWRITING_IOU,
    ))
    p.register(MathOCRAnalyzer())                # Pix2Text — formula → LaTeX
    p.register(DocLayoutAnalyzer(                # RapidLayout — zone detection
        model_type=DOC_LAYOUT_MODEL,
    ))
    if CHARTS_ENABLED:                           # ← NEW
        p.register(ChartsAnalyzer())             # LLM-backed chart understanding
    p.register(BarcodeQRAnalyzer())

    # People
    p.register(FaceDetectionAnalyzer(confidence=FACE_CONFIDENCE))
    p.register(PoseDetectionAnalyzer(conf=POSE_CONFIDENCE))

    # Scene
    p.register(ClassificationAnalyzer())
    p.register(EmbeddingsAnalyzer())

    # Spatial last
    p.register(SpatialAnalyzer())

    return p