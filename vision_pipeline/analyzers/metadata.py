# analyzers/metadata.py
from PIL import Image, ExifTags
from core.base import BaseAnalyzer
from core.result import AnalysisResult
import numpy as np

class MetadataAnalyzer(BaseAnalyzer):
    name = "metadata"

    def analyze(self, image, context):
        pil = context.get("pil_image")
        if pil is None:
            return AnalysisResult(self.name, {"note": "no PIL image in context"})
        exif = pil.getexif()
        tags = {ExifTags.TAGS.get(k, k): str(v) for k, v in exif.items()}
        return AnalysisResult(self.name, {
            "width": pil.width, "height": pil.height,
            "mode": pil.mode, "format": pil.format,
            "exif": tags,
        })