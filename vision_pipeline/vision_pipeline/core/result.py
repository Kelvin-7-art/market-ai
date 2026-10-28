# core/result.py
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class BoundingBox:
    x1: int; y1: int; x2: int; y2: int
    label: str
    confidence: float

    @property
    def center(self): return ((self.x1+self.x2)//2, (self.y1+self.y2)//2)
    @property
    def area(self): return (self.x2-self.x1) * (self.y2-self.y1)

@dataclass
class AnalysisResult:
    analyzer: str
    data: dict[str, Any] = field(default_factory=dict)
    boxes: list[BoundingBox] = field(default_factory=list)
    error: str | None = None