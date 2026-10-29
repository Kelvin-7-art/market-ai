"""TradeGPT — market intelligence workspace with vision analysis."""

from __future__ import annotations

import streamlit as st

# Bootstrap before importing models/providers; preserve the original root layout.
import os
import sys
import hmac
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
from market_ai.utils.cloud_config import setting

st.set_page_config(page_title="AI Framework — AI Market Assistant", page_icon="📈", layout="wide", initial_sidebar_state="expanded")
# Keep existing providers that read os.environ compatible with Cloud secrets.
for name in ("GROQ_API_KEY", "GROQ_PRIMARY_API_KEY", "GROQ_SECONDARY_API_KEY", "LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    value = setting(name)
    if value:
        os.environ[name] = value
if setting("CLOUD_MODE", "false").lower() == "true":
    password = setting("APP_PASSWORD")
    if not password:
        st.error("Set APP_PASSWORD in Streamlit Cloud secrets before using this deployment.")
        st.stop()
    if not st.session_state.get("cloud_authenticated"):
        st.title("AI Framework")
        with st.form("cloud_login"):
            entered = st.text_input("Workspace password", type="password")
            submitted = st.form_submit_button("Open workspace")
        if submitted:
            if hmac.compare_digest(entered.encode(), password.encode()):
                st.session_state.cloud_authenticated = True
                st.rerun()
            st.error("Incorrect password.")
        st.stop()

from market_ai.analytics.router import render_analytics
from market_ai.components.cards import render_hero
from market_ai.components.chat import render_chat
from market_ai.components.footer import render_footer
from market_ai.components.header import render_header
from market_ai.components.models_registry import render_models_registry
from market_ai.components.projects import render_projects
from market_ai.components.settings import render_settings
from market_ai.components.sidebar import render_sidebar
from market_ai.utils.helpers import html_card
from market_ai.utils.theme import load_css
from market_ai.utils.vision_client import health_check


load_css()


# ── Session state defaults ─────────────────────────────────
if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "last_upload_name" not in st.session_state:
    st.session_state.last_upload_name = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline_healthy" not in st.session_state:
    st.session_state.pipeline_healthy = None


def render_module_page(title: str, icon: str, description: str) -> None:
    """Render a considered empty state for modules beyond the first build."""
    render_header(title)
    html_card(
        f"""
        <div class="empty-state">
          <div class="empty-icon">{icon}</div>
          <h3>{title} workspace</h3>
          <p>{description}</p>
        </div>
        """
    )


def render_pipeline_status() -> None:
    """Small banner showing whether the vision pipeline is reachable."""
    healthy = st.session_state.get("pipeline_healthy")
    if healthy is None:
        return

    if healthy:
        st.caption("🟢 Vision pipeline: online")
    else:
        st.caption("🔴 Vision backend: unavailable — check the service and deployment secrets")


# ── Route ─────────────────────────────────────────────────
page = render_sidebar()


if page == "Dashboard":
    render_header("Dashboard")
    render_hero()
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    from market_ai.components.analytics import render_dashboard

    render_dashboard()


elif page == "Analytics":
    render_header("Analytics")
    analytics_view = st.session_state.get("analytics_view", "Dashboard")
    if analytics_view == "Dashboard":
        render_hero()
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    render_analytics(
        analytics_view,
        st.session_state.get("analytics_model_category", "Random Forest"),
        st.session_state.get("analytics_model_variant", "Base Model"),
    )


elif page == "Chat":
    render_header("Chat")

    # Health check on first visit to this page
    if st.session_state.pipeline_healthy is None:
        st.session_state.pipeline_healthy = health_check()

    render_pipeline_status()

    if not st.session_state.pipeline_healthy:
        st.warning("The vision backend is unavailable or still waking up. Check BACKEND_URL and SERVICE_TOKEN, then re-check.")
        if st.button("Re-check pipeline"):
            st.session_state.pipeline_healthy = health_check()
            st.rerun()

    render_chat()


elif page == "Projects":
    render_header("Projects")
    render_projects()


elif page == "Models":
    render_header("Model Registry")
    render_models_registry()


else:
    render_header("Settings")
    render_settings()


render_footer()