# analyzers/table_extractor.py
import os

# Bypass Windows symlink permission issue (must be set before docling imports)
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from core.base import BaseAnalyzer
from core.result import AnalysisResult


class TableExtractorAnalyzer(BaseAnalyzer):
    name = "table_extractor"

    def __init__(self):
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
            TableFormerMode,
            EasyOcrOptions,
        )
        from docling.document_converter import DocumentConverter, ImageFormatOption

        # Build pipeline options — note: NO backend= here
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            mode=TableFormerMode.ACCURATE,
            do_cell_matching=True,
        )
        pipeline_options.ocr_options = EasyOcrOptions(lang=["en"])

        # Wrap pipeline_options in ImageFormatOption (per Docling v2 API)
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.IMAGE],
            format_options={
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
            },
        )

    def analyze(self, image, context):
        import tempfile
        from pathlib import Path
        import numpy as np
        from PIL import Image

        pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            pil_img.save(tmp_path)

        try:
            result = self.converter.convert(str(tmp_path))
            doc = result.document

            tables = []
            for i, table in enumerate(doc.tables):
                df = table.export_to_dataframe(doc=doc)
                df = df.fillna("")

                cells = []
                for row_idx, row in df.iterrows():
                    for col_idx, value in enumerate(row):
                        cells.append({
                            "row": int(row_idx),
                            "col": int(col_idx),
                            "text": str(value).strip(),
                        })

                tables.append({
                    "index": i,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                    "headers": [str(c).strip() for c in df.columns.tolist()],
                    "data": df.values.tolist(),
                    "cells": cells,
                })

            return AnalysisResult(self.name, {
                "count": len(tables),
                "tables": tables,
            })

        except Exception as e:
            return AnalysisResult(self.name, error=str(e))
        finally:
            tmp_path.unlink(missing_ok=True)