# scripts/test_docling.py
from docling.document_converter import DocumentConverter, ImageFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, TableStructureOptions, TableFormerMode, EasyOcrOptions,
)

pipeline_options = PdfPipelineOptions(
    do_ocr=True,
    do_table_structure=True,
    images_scale=2.0,
    table_structure_options=TableStructureOptions(
        mode=TableFormerMode.ACCURATE,
        do_cell_matching=True,
    ),
    ocr_options=EasyOcrOptions(lang=["en"], confidence_threshold=0.5, use_gpu=False),
)

converter = DocumentConverter(
    format_options={ImageFormatOption: pipeline_options}
)

path = r"C:\Users\Khots\Downloads\image.png"
result = converter.convert(path)
doc = result.document

print(f"Tables found: {len(doc.tables)}")
for i, table in enumerate(doc.tables):
    df = table.export_to_dataframe(doc=doc)
    df = df.fillna("")
    print(f"\nTable {i+1}: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Headers: {df.columns.tolist()}")
    print(df.to_string())