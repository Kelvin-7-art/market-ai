# core/base.py
from abc import ABC, abstractmethod
import numpy as np
from .result import AnalysisResult

class BaseAnalyzer(ABC):
    name: str = "base"

    @abstractmethod
    def analyze(self, image: np.ndarray, context: dict) -> AnalysisResult:
        ...