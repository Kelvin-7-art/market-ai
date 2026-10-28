# api/schemas.py
from pydantic import BaseModel
from typing import Any


class BoundingBoxSchema(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    confidence: float


class AnalyzerResultSchema(BaseModel):
    analyzer: str
    data: dict[str, Any] = {}
    boxes: list[BoundingBoxSchema] = []
    error: str | None = None


class AnalysisResponse(BaseModel):
    filename: str
    width: int
    height: int
    results: dict[str, AnalyzerResultSchema]


class HealthResponse(BaseModel):
    status: str
    analyzers: list[str]


# ── Phase 2 ─────────────────────────────────────────────────
class ComparisonResponse(BaseModel):
    comparison: dict[str, Any]