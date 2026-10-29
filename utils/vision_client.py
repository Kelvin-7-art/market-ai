"""Vision client supporting both the desktop pipeline and Render backend."""
from __future__ import annotations
import io
from urllib.parse import urlsplit
import requests
from PIL import Image
from market_ai.utils.cloud_config import setting

# ============================================================
# CONNECTION AND RESPONSE VALIDATION
# ============================================================
def hosted_backend() -> bool:
    """Whether the lightweight hosted backend is configured."""
    return bool(setting("BACKEND_URL"))

def _connection() -> tuple[str, dict]:
    url = (setting("BACKEND_URL") or setting("VISION_PIPELINE_URL", "http://localhost:8000")).rstrip("/")
    headers = {}
    if hosted_backend():
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("BACKEND_URL must be an HTTPS base URL without /health or credentials.")
        token = setting("SERVICE_TOKEN")
        if not token:
            raise ValueError("Set SERVICE_TOKEN in Streamlit secrets to match the Render backend.")
        headers["X-Service-Token"] = token
    return url, headers

def backend_request(path: str, *, method: str = "POST", timeout: int = 180, **kwargs) -> dict:
    """Make one request; return a safe, actionable error on failure."""
    try:
        url, headers = _connection()
        response = requests.request(method, url + path, headers=headers, timeout=(15, timeout), allow_redirects=False, **kwargs)
        if response.status_code == 401:
            return {"error": "SERVICE_TOKEN does not match the Render backend."}
        if response.status_code != 200:
            messages = {413: "Request exceeds the backend size limit. Use fewer files or shorter text.",
                        429: "Groq rate limit reached. Wait before trying again.",
                        502: "Backend or Groq rejected the request. Check the Render backend logs.",
                        504: "Backend or Groq timed out. Try again; successful analyses remain cached."}
            return {"error": messages.get(response.status_code, f"Backend returned HTTP {response.status_code}. Check its deployment and logs.")}
        data = response.json()
        return data if isinstance(data, dict) else {"error": "Backend returned an unexpected response."}
    except requests.exceptions.Timeout:
        return {"error": "Backend timed out. It may be waking up; retry after checking its status."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot reach the vision backend. Check its URL and service status."}
    except requests.exceptions.JSONDecodeError:
        return {"error": "Backend returned invalid JSON."}
    except requests.exceptions.RequestException:
        return {"error": "Backend request failed. Check the service status."}
    except ValueError as error:
        return {"error": str(error) if not isinstance(error, requests.exceptions.JSONDecodeError) else "Backend returned invalid JSON."}

# ============================================================
# PUBLIC IMAGE CLIENT
# ============================================================
def analyze_image(pil_image: Image.Image, timeout: int = 180) -> dict:
    """Send an in-memory PNG and normalize hosted output for the existing UI."""
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    raw = buf.getvalue()
    if hosted_backend() and len(raw) > 5 * 1024 * 1024:
        return {"error": "The converted PNG exceeds 5 MB. Resize the image before uploading."}
    data = backend_request("/analyze", timeout=timeout, files={"file": ("image.png", raw, "image/png")})
    if "error" in data:
        return data
    if hosted_backend():
        extraction = data.get("extraction")
        if not isinstance(extraction, dict):
            return {"error": "Expected the hosted backend's extraction JSON."}
        return {"mode": "groq_only", "results": {"charts": {"data": extraction}}}
    return data

def health_check() -> bool:
    """Check hosted authentication/readiness or the local health endpoint."""
    data = backend_request("/ready" if hosted_backend() else "/health", method="GET", timeout=15)
    return data.get("status") == "ok" and (not hosted_backend() or data.get("configured") is True)
