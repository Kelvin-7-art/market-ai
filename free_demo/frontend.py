"""Independent free-hosted demo; the full local market_ai app stays unchanged."""
import hashlib
import hmac
import json
import os
from urllib.parse import urlsplit, urlunsplit
import requests
import streamlit as st

st.set_page_config(page_title="AI Market Assistant", page_icon="◈", layout="centered")
st.title("AI Market Assistant")
st.caption("Charts, images and model reports · Free hosted demo")
password = os.getenv("APP_PASSWORD", "")
if not password:
    st.error("Set APP_PASSWORD in the frontend's Render environment settings.")
    st.stop()
if not st.session_state.get("authenticated"):
    with st.form("login"):
        entered = st.text_input("Access password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if hmac.compare_digest(entered, password):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password.")
    st.stop()

def normalize_backend_url(value):
    """Validate configuration before transmitting an upload or service token."""
    value = value.strip().rstrip("/")
    parts = urlsplit(value)
    if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or parts.port not in (None, 443)):
        raise ValueError("BACKEND_URL must be a public HTTPS base URL, without quotes, credentials or query parameters.")
    if parts.hostname in {"example.invalid", "localhost", "127.0.0.1", "0.0.0.0"}:
        raise ValueError("BACKEND_URL still points to a placeholder/local address. Set it to https://market-vision-free.onrender.com in Render > market-ai-free > Environment, then save and redeploy.")
    if parts.path not in ("", "/health", "/ready"):
        raise ValueError("BACKEND_URL must contain only the backend base address, without /analyze or another path.")
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


try:
    url = normalize_backend_url(os.getenv("BACKEND_URL", ""))
except ValueError as error:
    st.error(str(error))
    st.stop()
token = os.getenv("SERVICE_TOKEN", "").strip()
if not token:
    st.error("Set SERVICE_TOKEN to the same nonempty secret on both Render services, then redeploy.")
    st.stop()
st.caption(f"Backend in use: {url}")
for key, default in [("messages", []), ("analyses", {}), ("generation", 0)]:
    st.session_state.setdefault(key, default)


def api(path, method="POST", **kwargs):
    """Distinguish transport, HTTP and payload errors; never expose credentials."""
    try:
        response = requests.request(method, url + path,
                                    headers={"X-Service-Token": token},
                                    timeout=(15, 150), allow_redirects=False, **kwargs)
    except requests.exceptions.SSLError:
        raise ValueError(f"{path}: HTTPS certificate verification failed. Check BACKEND_URL; do not disable TLS verification.") from None
    except requests.exceptions.ConnectTimeout:
        raise ValueError(f"{path}: connection to the backend timed out after 15 seconds. Check its Render status and public URL.") from None
    except requests.exceptions.ReadTimeout:
        raise ValueError(f"{path}: no response arrived within 150 seconds. Open backend Logs to see whether this request reached /analyze or /chat. Do not repeatedly submit it.") from None
    except requests.exceptions.ConnectionError:
        raise ValueError(f"{path}: could not establish a connection to {url}. Check the displayed URL and the backend's Render status; browser /health alone does not verify this frontend's connection.") from None
    except requests.exceptions.RequestException:
        raise ValueError(f"{path}: HTTP transport failed. Check the backend Logs and URL.") from None
    if 300 <= response.status_code < 400:
        raise ValueError(f"{path}: backend returned a redirect (HTTP {response.status_code}). Use its final public HTTPS base URL.")
    if response.status_code == 401:
        raise ValueError(f"{path}: backend is reachable but rejected SERVICE_TOKEN. Copy the same token into both services and redeploy both.")
    if response.status_code == 404:
        raise ValueError(f"{path}: endpoint not found (HTTP 404). Check that BACKEND_URL points to market-vision-free, not market-ai-free, and that the backend runs free_demo.backend:app.")
    try:
        data = response.json()
    except ValueError:
        raise ValueError(f"{path}: HTTP {response.status_code}, but the response was not JSON. It may be a Render startup/error page; inspect backend status and Logs.") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path}: unexpected backend response format.")
    if response.status_code != 200:
        raise ValueError(f"{path}: HTTP {response.status_code}: {data.get('detail', 'Backend request failed.')}")
    return data


if st.button("Test backend connection", key="connection_test"):
    with st.spinner("Checking backend; a sleeping instance may need about a minute…"):
        try:
            health = api("/health", method="GET")
            if health.get("mode") != "groq_only" or health.get("status") != "ok":
                raise ValueError("Connected endpoint is not the expected Groq-only backend.")
            ready = api("/ready", method="GET")
            if not ready.get("configured"):
                raise ValueError("Backend reachable and SERVICE_TOKEN accepted, but GROQ_API_KEY is missing on the backend.")
            st.success("Frontend reached backend; SERVICE_TOKEN accepted; Groq key is present. This test did not call Groq or verify the key's validity. Now try one image.")
        except ValueError as error:
            st.error(str(error))


def compact(value):
    if isinstance(value, dict):
        return {k: ({"omitted": "Use the full export for row-level details", "count": len(v) if isinstance(v, list) else None}
                    if k in {"test_predictions", "embeddings", "cv_results"} else compact(v)) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 20:
            return {"count": len(value), "first": compact(value[0]), "last": compact(value[-1]),
                    "notice": "Endpoints only, not extrema; middle items omitted"}
        return [compact(x) for x in value]
    return value


if st.button("Reset chat and attachments"):
    st.session_state.messages = []
    st.session_state.analyses = {}
    st.session_state.generation += 1
    st.rerun()
st.info("This demo explains images and JSON. Model training and the 20 local vision analyzers remain in your desktop app.")
files = st.file_uploader("Attach up to 3 images (5 MB each)", type=["png", "jpg", "jpeg", "webp"],
                         accept_multiple_files=True, key=f"files_{st.session_state.generation}")
raw = st.text_area("Analysis JSON (optional)", height=160, key=f"json_{st.session_state.generation}")
valid = len(files) <= 3
if not valid:
    st.error("Choose at most 3 images per conversation.")
payload = None
if raw.strip():
    try:
        payload = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
    except ValueError as error:
        st.error(f"Invalid JSON: {error}")
        valid = False
signature = hashlib.sha256((raw + ''.join(hashlib.sha256(f.getvalue()).hexdigest()+f.name for f in files)).encode()).hexdigest()
if st.session_state.get("signature") != signature:
    st.session_state.messages = []
    st.session_state.signature = signature
    # Keep only currently attached images; do not grow session memory indefinitely.
    active = {hashlib.sha256(f.getvalue()).hexdigest() for f in files}
    st.session_state.analyses = {k:v for k,v in st.session_state.analyses.items() if k in active}
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
st.caption("Selected files and JSON are sent to Groq when you ask. Successful image analyses are reused for follow-ups. Changing attachments starts a new conversation.")
question = st.chat_input("Ask about these images or JSON…", disabled=not valid)
if question:
    try:
        if len(question) > 1000:
            raise ValueError("Please keep your question under 1000 characters.")
        images = []
        for index, file in enumerate(files, 1):
            blob = file.getvalue()
            if len(blob) > 5 * 1024 * 1024:
                raise ValueError(f"Image {index} exceeds 5 MB.")
            digest = hashlib.sha256(blob).hexdigest()
            if digest not in st.session_state.analyses:
                with st.spinner(f"Analyzing image {index}; a sleeping backend can take a minute to wake up…"):
                    st.session_state.analyses[digest] = api("/analyze", files={"file": (file.name, blob, file.type)})
            images.append({"image_id": index, "filename": file.name,
                           "extraction": st.session_state.analyses[digest]["extraction"]})
        context = json.dumps({"images": images, "report": compact(payload)}, ensure_ascii=False, separators=(",", ":"))
        if len(context.encode()) > 12000:
            raise ValueError("These results exceed the demo context budget. Use fewer images or smaller JSON. No chat request was sent.")
        history = []
        size = 0
        for message in reversed(st.session_state.messages[-4:]):
            cost = len(message['content'].encode())
            if size + cost > 4000 or len(message['content']) > 5000:
                break
            history.insert(0, message)
            size += cost
        with st.spinner("Preparing your explanation…"):
            reply = api("/chat", json={"question": question, "context": context, "history": history})["reply"]
        st.session_state.messages.extend([{"role": "user", "content": question}, {"role": "assistant", "content": reply}])
        st.rerun()
    except ValueError as error:
        st.error(str(error))
if st.session_state.analyses:
    with st.expander("Extracted image data"):
        st.json(list(st.session_state.analyses.values()))
