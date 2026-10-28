"""Independent free-hosted demo; the full local market_ai app stays unchanged."""
import hashlib
import hmac
import json
import os
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

url = os.getenv("BACKEND_URL", "").rstrip("/")
token = os.getenv("SERVICE_TOKEN", "")
if not url or not token:
    st.info("Set BACKEND_URL to the backend's public HTTPS URL and SERVICE_TOKEN to the same secret on both services.")
    st.stop()
for key, default in [("messages", []), ("analyses", {}), ("generation", 0)]:
    st.session_state.setdefault(key, default)


def api(path, **kwargs):
    try:
        response = requests.post(url + path, headers={"X-Service-Token": token}, timeout=(10, 150), **kwargs)
        try:
            data = response.json()
        except ValueError:
            raise ValueError("Backend is starting or returned an unexpected response. Wait a minute and retry.")
        if response.status_code != 200:
            raise ValueError(str(data.get("detail", "Backend request failed.")))
        return data
    except requests.RequestException:
        raise ValueError("Backend is waking up, unreachable or timed out. Wait a minute and retry.") from None


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
