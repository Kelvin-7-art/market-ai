# analyzers/doc_layout.py
from core.base import BaseAnalyzer
from core.result import AnalysisResult, BoundingBox


class DocLayoutAnalyzer(BaseAnalyzer):
    name = "doc_layout"

    def __init__(self, model_type="doclayout_docstructbench"):
        # RapidLayout bundles multiple models. 'doclayout_docstructbench'
        # is the general-purpose model trained on the most diverse
        # document set. Other valid values (from ModelType enum):
        #   pp_layout_cdla, pp_layout_publaynet, pp_layout_table,
        #   yolov8n_layout_paper, yolov8n_layout_report,
        #   yolov8n_layout_publaynet, yolov8n_layout_general6,
        #   doclayout_docstructbench, doclayout_d4la, doclayout_docsynth,
        #   pp_doc_layoutv2, pp_doc_layoutv3
        from rapid_layout import RapidLayout
        self.engine = RapidLayout(model_type=model_type)

    def analyze(self, image, context):
        import numpy as np
        from PIL import Image

        pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        pil_img = pil_img.convert("RGB")

        try:
            results = self.engine(np.array(pil_img))
        except Exception as e:
            return AnalysisResult(self.name, error=f"RapidLayout failed: {e}")

        regions = []
        boxes = []

        if hasattr(results, "boxes") and results.boxes is not None:
            for i in range(len(results.boxes)):
                bbox = results.boxes[i]
                label = results.labels[i] if hasattr(results, "labels") else "unknown"
                score = results.scores[i] if hasattr(results, "scores") else 1.0

                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                regions.append({
                    "type": str(label),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": float(score),
                })
                boxes.append(BoundingBox(x1, y1, x2, y2, str(label), float(score)))

        type_counts = {}
        for r in regions:
            t = r["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return AnalysisResult(
            self.name,
            {
                "region_count": len(regions),
                "type_counts": type_counts,
                "regions": regions,
            },
            boxes=boxes,
        )