"""Bound API context while retaining source identities and omission notices."""
from __future__ import annotations
import json


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


def compact_context(analysis: dict, budget: int = 7000) -> dict:
    """Return a bounded copy; original cached analyses are never modified."""
    results = analysis.get("results", {})
    images = results.get("images", {}) if isinstance(results, dict) else {}
    records = images.get("data") if isinstance(images, dict) else images
    if not isinstance(records, list):
        records = [{"filename": analysis.get("source_filename", "Image 1"), "analysis": analysis}]
    if len(records) > 12:
        raise ValueError("This request contains more than 12 images. Send a smaller batch.")
    skipped = {"embeddings", "embedding", "vector", "vectors", "base64", "image_base64",
               "raw_bytes", "pixel_values", "heatmap", "mask", "masks"}

    def prune(value, width, text_length, depth=0):
        if depth > 12:
            return {"omitted": "Nested detail exceeds API summary depth"}
        if isinstance(value, dict):
            out = {}
            # Prefer chart interpretation and extracted text over technical metadata.
            ordered = sorted(value.items(), key=lambda pair: (
                pair[0] not in {"charts", "ocr", "tables", "text", "insights", "results", "data", "error"},
            ))
            for key, item in ordered:
                if str(key).lower() in skipped:
                    out[key] = {"omitted": "Large machine-readable analyzer output"}
                else:
                    out[key] = prune(item, width, text_length, depth + 1)
            return out
        if isinstance(value, list):
            if len(value) <= width:
                return [prune(x, width, text_length, depth + 1) for x in value]
            return {"total_items": len(value), "first_items": [prune(x, width, text_length, depth + 1) for x in value[:width]],
                    "omitted_items": len(value) - width, "note": "Sample only; not the full series or its extrema"}
        if isinstance(value, str) and len(value) > text_length:
            return value[:text_length] + " [text shortened]"
        return value

    for width, length in [(12, 1200), (6, 600), (3, 300), (1, 120)]:
        output = {"notice": "API summary only. Machine vectors and long detail may be omitted. Do not infer missing values or extrema. Full analyses remain cached in the app.",
                  "images": [{"image_id": i + 1, "filename": row.get("filename", f"Image {i+1}"),
                              "analysis": prune(row.get("analysis", {}), width, length)}
                             for i, row in enumerate(records)]}
        if isinstance(results, dict) and "document_text" in results and len(records) > 1:
            output["documents"] = prune(results["document_text"], width, length)
        if len(_dump(output).encode("utf-8")) <= budget:
            return output
    raise ValueError("Extracted context remains too large after summarization. Send fewer images or a focused crop. Cached results have not been deleted.")
