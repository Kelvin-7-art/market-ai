# analyzers/charts.py — Groq version
import os
import json
import base64
import io
from core.base import BaseAnalyzer
from core.result import AnalysisResult


class ChartsAnalyzer(BaseAnalyzer):
    name = "charts"

    # Keywords that suggest a chart is present in the OCR text
    CHART_KEYWORDS = [
        "axis", "chart", "graph", "series", "forecast",
        "trend", "legend", "sma", "price", "volume",
        "bar", "line", "pie", "scatter", "candlestick",
    ]

    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            self.client = None
            return

        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = "qwen/qwen3.8-27b"

    def analyze(self, image, context):
        import numpy as np
        from PIL import Image

        if self.client is None:
            return AnalysisResult(self.name, {
                "is_chart": None,
                "note": "GROQ_API_KEY not set — chart analysis skipped",
            })

        # ── Pre-filter: skip API call if OCR gives no chart signal ──
        # Saves Groq API quota and avoids rate limits on
        # non-chart images (receipts, portraits, documents).
        ocr_segments = context.get("ocr_segments", [])
        has_chart_indicators = any(
            any(kw in seg.get("text", "").lower() for kw in self.CHART_KEYWORDS)
            for seg in ocr_segments
        )

        # If OCR found many segments but none suggest a chart,
        # skip the API call entirely.
        if not has_chart_indicators and len(ocr_segments) > 5:
            return AnalysisResult(self.name, {
                "is_chart": False,
                "note": "No chart indicators in OCR — skipped API call",
            })

        # ── Prepare image ──
        pil_img = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        pil_img = pil_img.convert("RGB")

        max_dim = 1568
        if max(pil_img.size) > max_dim:
            ratio = max_dim / max(pil_img.size)
            new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
            pil_img = pil_img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

        prompt = """Analyze this image. Determine if it contains a chart, graph, or data visualization.

If YES, return ONLY valid JSON matching this schema:
{
  "is_chart": true,
  "chart_type": "candlestick" | "bar" | "line" | "pie" | "scatter" | "area" | "other",
  "title": "string or null",
  "x_axis": {"label": "string or null", "range": ["min", "max"]},
  "y_axis": {"label": "string or null", "range": ["min", "max"]},
  "series": [
    {"name": "string", "type": "string", "is_projection": true|false,
     "trend": "upward"|"downward"|"flat"|"mixed"}
  ],
  "insights": ["short bullet points"]
}

If NOT a chart, return ONLY: {"is_chart": false}

Do not include any text outside the JSON."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=2000,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}",
                            },
                        },
                    ],
                }],
            )

            text = response.choices[0].message.content.strip()

            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

            parsed = json.loads(text)

        except json.JSONDecodeError as e:
            return AnalysisResult(self.name, error=f"LLM returned invalid JSON: {e}")
        except Exception as e:
            err_str = str(e)
            # Handle rate limit gracefully — don't crash the pipeline
            if "429" in err_str or "rate limit" in err_str.lower():
                return AnalysisResult(self.name, {
                    "is_chart": None,
                    "note": "Groq rate limit reached — retry later",
                })
            return AnalysisResult(self.name, error=f"Chart analyzer failed: {e}")

        if not parsed.get("is_chart"):
            return AnalysisResult(self.name, {
                "is_chart": False,
                "chart_type": None,
                "series": [],
            })

        return AnalysisResult(self.name, {
            "is_chart": True,
            "chart_type": parsed.get("chart_type"),
            "title": parsed.get("title"),
            "x_axis": parsed.get("x_axis"),
            "y_axis": parsed.get("y_axis"),
            "series": parsed.get("series", []),
            "insights": parsed.get("insights", []),
        })