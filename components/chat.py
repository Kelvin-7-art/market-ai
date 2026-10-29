"""Chat interface with vision pipeline integration.

Supports text chat plus image and document uploads.
- Images are sent to the vision pipeline (20 analyzers) for structured extraction.
- Documents are text-extracted (PDF, DOCX, TXT, MD, CSV).
- The LLM reasons over the extracted JSON to answer follow-up questions.
"""
from __future__ import annotations

import io
import copy
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import streamlit as st

from market_ai.utils.vision_client import analyze_image
from market_ai.utils.llm_chat import chat_completion


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================
def _try_import_pypdf():
    try:
        from pypdf import PdfReader
        return PdfReader
    except Exception:
        return None


def _try_import_docx():
    try:
        import docx  # python-docx
        return docx
    except Exception:
        return None


# ============================================================
# ATTACHMENT + MESSAGE MODELS
# ============================================================
@dataclass
class Attachment:
    """A single uploaded file plus any text we managed to extract."""
    name: str
    mime_type: str
    size_bytes: int
    kind: str  # "image" | "document" | "unknown"
    raw_bytes: bytes = b""
    text: str = ""
    preview_data_url: Optional[str] = None
    analysis: Optional[dict] = None  # vision pipeline JSON (images only)

    @property
    def is_image(self) -> bool:
        return self.kind == "image"

    @property
    def is_document(self) -> bool:
        return self.kind == "document"


@dataclass
class Message:
    """A chat message with optional attachments."""
    role: str  # "user" | "assistant"
    content: str
    attachments: list[Attachment] = field(default_factory=list)


# ============================================================
# FILE CLASSIFICATION + EXTRACTION
# ============================================================
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}
DOCUMENT_EXTENSIONS = {"pdf", "docx", "txt", "md", "csv", "json", "log"}


def _classify(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    return "unknown"


def _extract_image_preview(uploaded_file) -> Optional[str]:
    """Return a base64 data URL for inline preview."""
    try:
        import base64
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        mime = uploaded_file.type or "image/png"
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _extract_document_text(uploaded_file) -> str:
    """Best-effort text extraction for common document types."""
    name = (uploaded_file.name or "").lower()
    ext = name.rsplit(".", 1)[-1] if "." in name else ""

    try:
        uploaded_file.seek(0)
        raw = uploaded_file.read()
        uploaded_file.seek(0)
    except Exception:
        return ""

    # Plain text formats
    if ext in {"txt", "md", "csv", "json", "log"}:
        try:
            return raw.decode("utf-8", errors="replace")
        except Exception:
            return ""

    # PDF
    if ext == "pdf":
        PdfReader = _try_import_pypdf()
        if PdfReader is None:
            return "[PDF text extraction unavailable — install `pypdf`]"
        try:
            reader = PdfReader(io.BytesIO(raw))
            chunks = []
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
            return "\n".join(chunks).strip()
        except Exception as e:
            return f"[PDF extraction failed: {e}]"

    # DOCX
    if ext == "docx":
        docx = _try_import_docx()
        if docx is None:
            return "[DOCX text extraction unavailable — install `python-docx`]"
        try:
            document = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in document.paragraphs).strip()
        except Exception as e:
            return f"[DOCX extraction failed: {e}]"

    return ""


def _build_attachment(uploaded_file) -> Attachment:
    """Build an Attachment, extracting preview or text as appropriate."""
    kind = _classify(uploaded_file)
    try:
        uploaded_file.seek(0, io.SEEK_END)
        size = uploaded_file.tell()
        uploaded_file.seek(0)
    except Exception:
        size = 0

    att = Attachment(
        name=uploaded_file.name,
        mime_type=uploaded_file.type or "application/octet-stream",
        size_bytes=size,
        raw_bytes=uploaded_file.getvalue(),
        kind=kind,
    )

    if kind == "image":
        att.preview_data_url = _extract_image_preview(uploaded_file)
    elif kind == "document":
        att.text = _extract_document_text(uploaded_file)

    return att


# ============================================================
# VISION PIPELINE INTEGRATION
# ============================================================
def _run_vision_pipeline(att: Attachment) -> dict:
    """Send an image attachment through the vision pipeline.

    Returns the pipeline JSON (or {"error": ...} on failure).
    """
    cache = st.session_state.setdefault("vision_analysis_cache", {})
    fingerprint = hashlib.sha256(att.raw_bytes).hexdigest()
    if fingerprint in cache:
        att.analysis = copy.deepcopy(cache[fingerprint])
        return att.analysis
    try:
        from PIL import Image
        with Image.open(io.BytesIO(att.raw_bytes)) as image:
            analysis = analyze_image(image)
    except Exception:
        analysis = {"error": "Could not decode this image. Try another image file."}
    if not analysis.get("error"):
        cache[fingerprint] = copy.deepcopy(analysis)
    att.analysis = copy.deepcopy(analysis)
    st.session_state.analysis = copy.deepcopy(analysis)
    st.session_state.last_upload_name = att.name
    return att.analysis


def _summarize_analysis(analysis: dict) -> str:
    """Summarize successful and partial analyzer results without assuming fields."""
    if not analysis:
        return "No analysis available."
    if analysis.get("error"):
        return str(analysis["error"])
    results = analysis.get("results") or {}
    if not isinstance(results, dict):
        return "No structured analyzer results available."
    lines = []
    def data_for(name: str) -> dict:
        item = results.get(name)
        data = item.get("data") if isinstance(item, dict) else None
        return data if isinstance(data, dict) else {}

    charts = data_for("charts")
    if charts.get("is_chart") or charts.get("chart_type"):
        lines.append(f"**Chart:** {charts.get('chart_type') or 'unknown'}")
        series = charts.get("series") or []
        if isinstance(series, list) and series:
            names = [str(item.get("name") or "Unnamed series") if isinstance(item, dict)
                     else str(item) for item in series[:4]]
            lines.append(f"**Series ({len(series)}):** " + ", ".join(names))
        insights = charts.get("insights") or []
        if isinstance(insights, str):
            insights = [insights]
        if isinstance(insights, list):
            lines.extend(f"- {insight}" for insight in insights[:3])
    if charts.get("note"):
        lines.append(f"**Chart note:** {charts['note']}")
    for name in ("table_extractor", "tables"):
        tables = data_for(name).get("tables") or []
        if isinstance(tables, list) and tables:
            lines.append(f"**Tables:** {len(tables)}")
            for table in tables[:3]:
                if isinstance(table, dict):
                    lines.append(f"- {table.get('rows', '?')} rows × {table.get('cols', '?')} columns")
    ocr = data_for("ocr")
    if ocr:
        count = ocr.get("segment_count")
        if count is None:
            count = len(ocr.get("segments") or [])
        lines.append(f"**OCR text segments:** {count}")
    for name, item in results.items():
        if isinstance(item, dict) and item.get("error"):
            lines.append(f"**{name} unavailable:** {item['error']}")
    return "\n\n".join(lines) or "Analysis complete; no chart, table or OCR summary available."


def _all_images_in_conversation() -> list[Attachment]:
    """Collect every image attachment across the chat history."""
    images = []
    for msg in st.session_state.get("messages", []):
        for att in msg.attachments:
            if att.is_image and att.analysis:
                images.append(att)
    return images


# ============================================================
# RESPONSE BUILDER
# ============================================================
def _build_response(prompt: str, attachments: list[Attachment]) -> str:
    """Send the prompt + vision JSON + doc text to the LLM."""
    new_images = [att for att in attachments if att.is_image]
    images = new_images
    if not images:
        # Follow-ups retain the most recent image batch, not only its last image.
        for previous in reversed(st.session_state.get("messages", [])):
            batch = [att for att in previous.attachments if att.is_image and att.analysis]
            if batch:
                images = batch
                break
    payloads = [{"filename": att.name, "analysis": copy.deepcopy(att.analysis)}
                for att in images if att.analysis is not None]
    # Preserve a single pipeline's full response. Multiple images retain every
    # result separately, with filenames, instead of silently taking the first.
    if len(payloads) == 1:
        merged = copy.deepcopy(payloads[0]["analysis"])
        merged["source_filename"] = payloads[0]["filename"]
        merged.setdefault("results", {})
    else:
        merged = {"results": {"images": {"data": payloads}}} if payloads else {"results": {}}

    # Keep documents available on subsequent turns as well as the upload turn.
    documents = []
    seen = set()
    historical = [att for msg in st.session_state.get("messages", [])
                  for att in msg.attachments]
    for att in [*historical, *attachments]:
        if not att.is_document or not att.text:
            continue
        identity = (att.name, hashlib.sha256(att.text.encode("utf-8")).hexdigest())
        if identity in seen:
            continue
        seen.add(identity)
        documents.append({"filename": att.name, "text": att.text})
    if documents:
        merged["results"]["document_text"] = {"data": {"documents": documents}}
    if images and all(not a.analysis or a.analysis.get("error") for a in images) and not documents:
        errors = [str(a.analysis.get("error", "No analysis available."))
                  for a in images if a.analysis]
        return "Image analysis is unavailable. " + " ".join(errors)
    if not payloads and not documents:
        return "Attach an image or document first, then ask a question about its extracted data."

    # render_chat has already appended the current user turn. Do not append it
    # twice. The fallback also permits calling this helper before that append.
    history = [{"role": msg.role, "content": msg.content}
               for msg in st.session_state.get("messages", []) if msg.content]
    content = prompt or "Please summarize the attached files using the extracted data."
    if history and history[-1]["role"] == "user":
        if not prompt:
            history[-1] = {"role": "user", "content": content}
    else:
        history.append({"role": "user", "content": content})
    st.session_state.analysis = copy.deepcopy(merged)
    return chat_completion(history, merged)


# ============================================================
# UI HELPERS
# ============================================================
def _render_attachment_preview(att: Attachment, *, compact: bool = True) -> None:
    """Render a single attachment preview beneath a chat bubble."""
    if att.is_image:
        try:
            st.image(att.raw_bytes, caption=att.name, width=360 if compact else "stretch")
        except Exception:
            st.warning(f"Could not preview image: {att.name}")

        # If we ran the vision pipeline, show a compact summary
        if att.analysis and "error" not in att.analysis:
            with st.expander("🔍 Vision analysis", expanded=False):
                st.markdown(_summarize_analysis(att.analysis))
        elif att.analysis and "error" in att.analysis:
            st.caption(f"⚠ Vision pipeline: {att.analysis['error']}")

    elif att.is_document:
        with st.expander(f"📄 {att.name}", expanded=False):
            if att.text:
                shown = att.text[:4000]
                if len(att.text) > 4000:
                    shown += "\n\n… (truncated)"
                st.text(shown)
            else:
                st.caption(
                    "No text extracted. Install `pypdf` or `python-docx` "
                    "to enable extraction."
                )
    else:
        st.caption(f"📎 {att.name} ({att.mime_type})")


def _assistant_avatar():
    """Use the bundled brand image in every assistant message."""
    logo = Path(__file__).resolve().parents[1] / "assets" / "ai_framework_logo.png"
    return str(logo) if logo.is_file() else "✦"


def _message_marker(role: str) -> None:
    marker = "research-user-marker" if role == "user" else "research-assistant-marker"
    st.markdown(f'<span class="{marker}" aria-hidden="true"></span>', unsafe_allow_html=True)


def _render_message(msg: Message) -> None:
    avatar = ":material/person:" if msg.role == "user" else _assistant_avatar()
    with st.chat_message(msg.role, avatar=avatar):
        _message_marker(msg.role)
        st.caption("You" if msg.role == "user" else "AI Assistant")
        if msg.content:
            st.markdown(msg.content)
        for att in msg.attachments:
            _render_attachment_preview(att)


def _render_chat_style() -> None:
    """Style only this conversation; use native containers for all content."""
    st.markdown("""
<style>
.st-key-research_conversation {max-width:1000px;margin:0 auto;}
.st-key-research_conversation [data-testid="stChatMessage"] {
    border:1px solid rgba(148,163,184,.18);border-radius:20px;
    padding:18px;margin-bottom:16px;max-width:90%;
    background:rgba(148,163,184,.06);min-width:0;
}
.st-key-research_conversation [data-testid="stChatMessage"]:has(.research-user-marker) {
    margin-left:auto;flex-direction:row-reverse;
    background:rgba(59,130,246,.14);border-color:rgba(96,165,250,.25);
    border-bottom-right-radius:5px;
}
.st-key-research_conversation [data-testid="stChatMessage"]:has(.research-assistant-marker) {
    margin-right:auto;border-bottom-left-radius:5px;
}
.st-key-research_conversation [data-testid="stChatMessageContent"] {min-width:0;overflow-wrap:anywhere;}
.st-key-research_conversation [data-testid="stChatMessageAvatarUser"] {
    background:#2563eb;color:white;border-radius:50%;
}
.st-key-research_conversation [data-testid="stChatMessage"] > img {
    object-fit:contain;border-radius:50%;
}
.st-key-research_conversation [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] .research-user-marker),
.st-key-research_conversation [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] .research-assistant-marker) {display:none;}
@media(max-width:640px) {
 .st-key-research_conversation [data-testid="stChatMessage"] {max-width:100%;padding:12px;}
}
</style>
""", unsafe_allow_html=True)


def _render_context() -> None:
    images = _all_images_in_conversation()
    documents = list(dict.fromkeys(att.name for msg in st.session_state.messages
                     for att in msg.attachments if att.is_document))
    names = ([images[-1].name] if images else []) + documents
    if names:
        with st.expander("Attached files · " + str(len(names)), expanded=False):
            st.caption("Follow-up questions reuse extracted information from your conversation.")
            for name in names:
                st.text(name)




# ============================================================
# SESSION STATE
# ============================================================
def _ensure_state() -> None:
    if not st.session_state.get("messages"):
        st.session_state.messages = [
            Message(
                role="assistant",
                content=(
                    "Hi! Upload a chart, image or document using the + button below, "
                    "then tell me what you would like to understand. You can ask "
                    "follow-up questions about the same file."
                ),
            )
        ]
    st.session_state.setdefault("uploader_nonce", 0)
    st.session_state.setdefault("vision_analysis_cache", {})
    st.session_state.setdefault("analysis", None)
    st.session_state.setdefault("last_upload_name", None)
    st.session_state.setdefault("pipeline_healthy", None)


def _reset_chat() -> None:
    st.session_state.pop("messages", None)
    st.session_state.pop("pending_prompt", None)
    st.session_state.pop("vision_analysis_cache", None)
    st.session_state.analysis = None
    st.session_state.last_upload_name = None
    st.session_state.pipeline_healthy = None
    st.session_state.uploader_nonce = st.session_state.get("uploader_nonce", 0) + 1
    _ensure_state()


# ============================================================
# MAIN RENDER
# ============================================================
def render_chat() -> None:
    """Render a conversation-first page with assistant and user avatars."""
    _ensure_state()
    _render_chat_style()
    header, action = st.columns([4, 1])
    with header:
        st.subheader("Your research assistant")
        st.caption("Ask questions about your charts and documents.")
    with action:
        if st.button("New chat", key="research_new_chat", use_container_width=True):
            _reset_chat()
            st.rerun()
    from market_ai.utils.vision_client import hosted_backend, health_check
    if hosted_backend():
        st.caption("Images and extracted document text are sent to the Render backend and Groq when you send. Up to 3 images, 5 MB per file.")
        with st.expander("Backend connection"):
            if st.button("Test backend connection", key="research_backend_test"):
                if health_check():
                    st.success("Backend reachable; service token accepted; Groq key present.")
                else:
                    st.warning("Backend unavailable or not configured. Check Render and your deployment secrets.")
    _render_context()

    # Native container grows with the messages: no empty HTML chat-shell panel.
    conversation = st.container(key="research_conversation")
    with conversation:
        for msg in st.session_state.messages:
            _render_message(msg)

    input_key = f"research_chat_input_{st.session_state.uploader_nonce}"
    if not any(msg.role == "user" for msg in st.session_state.messages):
        starters = [
            ("📈 Explain a chart", "Explain the chart I attach, including its axes, series and main trends."),
            ("📄 Summarize a document", "Summarize the document I attach and highlight the key findings."),
            ("🌲 Explain model results", "Explain the model results I attach, including the metrics and limitations."),
        ]
        cols = st.columns(3)
        for index, (label, prompt) in enumerate(starters):
            if cols[index].button(label, key=f"research_starter_{index}", use_container_width=True):
                # Populate only: user can attach a file and edit before sending.
                st.session_state[input_key] = prompt
        st.caption("Choose a starter or type below. Add files with the + button, then press Send.")

    result = st.chat_input(
        "Message your assistant…", accept_file="multiple",
        file_type=sorted(IMAGE_EXTENSIONS | DOCUMENT_EXTENSIONS), key=input_key,
    )
    if result is None:
        return
    prompt = (result.text or "").strip()
    uploads = result.files or []
    if not prompt and not uploads:
        return

    if hosted_backend() and (sum(Path(f.name).suffix.lower().lstrip(".") in {"png", "jpg", "jpeg", "gif", "webp", "bmp"} for f in uploads) > 3 or any(f.size > 5 * 1024 * 1024 for f in uploads)):
        st.error("Upload up to 3 images, with each file no larger than 5 MB.")
        return
    with conversation:
        with st.spinner("Reading your files…"):
            attachments = [_build_attachment(file) for file in uploads]
        user_msg = Message("user", prompt or "Please explain the attached files.", attachments)
        st.session_state.messages.append(user_msg)
        _render_message(user_msg)

        with st.chat_message("assistant", avatar=_assistant_avatar()):
            _message_marker("assistant")
            st.caption("AI Assistant")
            for att in attachments:
                if att.is_image:
                    with st.spinner(f"Analyzing {att.name}…"):
                        att.analysis = _run_vision_pipeline(att)
                        st.session_state.analysis = copy.deepcopy(att.analysis)
                        st.session_state.last_upload_name = att.name
            with st.spinner("Preparing your answer…"):
                response = _build_response(prompt, attachments)
            st.markdown(response)
        st.session_state.messages.append(Message("assistant", response))

    st.session_state.uploader_nonce += 1
    st.rerun()


if __name__ == "__main__":
    st.set_page_config(page_title="AI Research Desk", page_icon="💬", layout="wide")
    render_chat()
