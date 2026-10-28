# api/app.py
import io
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np

from api.deps import get_pipeline
from api.schemas import (
    AnalysisResponse,
    AnalyzerResultSchema,
    BoundingBoxSchema,
    HealthResponse,
    ComparisonResponse,
)

log = logging.getLogger(__name__)


# ── App ─────────────────────────────────────────────────────
app = FastAPI(
    title="Vision Pipeline API",
    description="Object detection, OCR, classification, quality, embeddings, and more.",
    version="0.2.0",
)


# ── Helpers ─────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


async def _load_upload(file: UploadFile) -> tuple[Image.Image, np.ndarray]:
    """Read a multipart upload → (PIL image, RGB numpy array).

    Raises HTTPException with a useful message on any failure.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected an image, got {file.content_type}")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"Image too large (max {MAX_UPLOAD_BYTES // 1024 // 1024} MB)"
        )

    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    return pil, np.array(pil)


# ── Routes ──────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health():
    p = get_pipeline()
    return HealthResponse(status="ok", analyzers=[a.name for a in p.analyzers])


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)):
    pil, img = await _load_upload(file)

    pipeline = get_pipeline()
    context = {"pil_image": pil, "image_id": file.filename}
    results: dict[str, AnalyzerResultSchema] = {}

    for a in pipeline.analyzers:
        try:
            r = a.analyze(img, context)
            if r.boxes:
                context.setdefault("boxes", []).extend(r.boxes)
            results[a.name] = AnalyzerResultSchema(
                analyzer=r.analyzer,
                data=r.data,
                boxes=[BoundingBoxSchema(**vars(b)) for b in r.boxes],
                error=r.error,
            )
        except Exception as e:
            log.exception("Analyzer %s failed", a.name)
            results[a.name] = AnalyzerResultSchema(
                analyzer=a.name, error=str(e)
            )

    return AnalysisResponse(
        filename=file.filename or "upload",
        width=pil.width,
        height=pil.height,
        results=results,
    )


@app.post("/analyze/region")
async def analyze_region(
    file: UploadFile = File(...),
    x1: int = 0,
    y1: int = 0,
    x2: int = 100,
    y2: int = 100,
):
    """Run a subset of analyzers on a specific region of the image.

    Query params x1/y1/x2/y2 define the bounding box in pixel coordinates.
    Values outside the image are clamped; an empty region returns 400.
    """
    from analyzers.region_analysis import RegionAnalysisAnalyzer
    from analyzers.quality import QualityAnalyzer
    from analyzers.color import ColorAnalyzer
    from analyzers.detection import DetectionAnalyzer
    from analyzers.ocr import OCRAnalyzer

    _, img = await _load_upload(file)

    # Region-scoped analyzer set — lighter than the full pipeline.
    # Detection is included so "what objects are in this crop?" works.
    region_pipeline = RegionAnalysisAnalyzer(analyzers=[
        QualityAnalyzer(),
        ColorAnalyzer(),
        DetectionAnalyzer(),
        OCRAnalyzer(min_confidence=0.3),
    ])

    result = region_pipeline.analyze(
        img, {"region": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
    )
    if result.error:
        raise HTTPException(400, result.error)

    return {"region_analysis": result.data}


@app.post("/compare", response_model=ComparisonResponse)
async def compare(
    file: UploadFile = File(...),
    reference: UploadFile = File(...),
):
    """Compare two images via CLIP (semantic) + SSIM (pixel)."""
    _, img_a = await _load_upload(file)
    _, img_b = await _load_upload(reference)

    from analyzers.image_comparison import ImageComparisonAnalyzer

    result = ImageComparisonAnalyzer().analyze(img_a, {"reference_image": img_b})
    if result.error:
        raise HTTPException(500, result.error)

    return ComparisonResponse(comparison=result.data)