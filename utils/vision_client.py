# market_ai/utils/vision_client.py
import io
import requests
from PIL import Image

VISION_PIPELINE_URL = "http://localhost:8000"


def analyze_image(pil_image: Image.Image, timeout: int = 180) -> dict:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    files = {"file": ("image.png", buf, "image/png")}
    try:
        r = requests.post(f"{VISION_PIPELINE_URL}/analyze", files=files, timeout=timeout)
    except requests.exceptions.ConnectionError:
        return {"error": "Vision pipeline not running. Start it with: python -m api.run"}
    except requests.exceptions.Timeout:
        return {"error": f"Vision pipeline timed out after {timeout}s"}
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    return r.json()


def health_check() -> bool:
    try:
        r = requests.get(f"{VISION_PIPELINE_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False