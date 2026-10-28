"""Compact sidebar navigation with monochrome icons and nested routes."""

from __future__ import annotations

from base64 import b64encode
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from market_ai.analytics.models.categories import MODEL_CATEGORIES, variants_for


# ===== Navigation =====
NAV_ITEMS = ["Dashboard", "Chat", "Analytics", "Projects", "Models", "Settings"]
ANALYTICS_ITEMS = [
    "Dashboard", "Chat", "Chat AI", "Models",
    "Sentiment", "Forecasting",
]
_ICONS = {
    "Dashboard": '<rect x="3" y="3" width="18" height="18" rx="3"/><path d="M3 9h18M9 9v12"/>',
    "Chat": '<path d="M21 11a8 8 0 0 1-8 8H7l-4 3V7a4 4 0 0 1 4-4h6a8 8 0 0 1 8 8Z"/><path d="M7 8h9M7 12h6"/>',
    "Analytics": '<path d="M4 3v18h17M8 16v-5M13 16V6M18 16V9"/>',
    "Projects": '<path d="m3 7 9-4 9 4-9 4-9-4Zm0 5 9 4 9-4M3 17l9 4 9-4"/>',
    "Models": '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M9 1v3m6-3v3M9 20v3m6-3v3M1 9h3m-3 6h3m16-6h3m-3 6h3"/><rect x="9" y="9" width="6" height="6" rx="1"/>',
    "Settings": '<path d="M3 7h18M3 17h18"/><circle cx="8" cy="7" r="3"/><circle cx="16" cy="17" r="3"/>',
}


# ===== Scoped appearance =====
def _sidebar_css() -> str:
    """Keep all navigation styling inside the sidebar."""
    css = """
    [data-testid="stSidebar"] {background:#101923!important;border-right:1px solid #29323d;}
    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding:1.2rem .8rem;}
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.35rem;}
    [data-testid="stSidebar"] .nav-brand {display:flex;align-items:center;gap:10px;margin:4px 4px 24px;}
    [data-testid="stSidebar"] .nav-brand img {width:36px;height:36px;object-fit:contain;}
    [data-testid="stSidebar"] .nav-brand strong {font-size:15px;color:#e8edf3;font-weight:600;}
    [data-testid="stSidebar"] .nav-brand small {display:block;color:#8995a5;font-size:10px;margin-top:3px;}
    [data-testid="stSidebar"] [class*="st-key-workspace_nav_"] button {
      width:100%;min-height:38px;padding:8px 10px!important;background:transparent!important;
      border:0!important;box-shadow:none!important;border-radius:7px!important;color:#b8c2ce!important;
      justify-content:flex-start!important;text-align:left!important;transition:background .15s;
    }
    [data-testid="stSidebar"] [class*="st-key-workspace_nav_"] button:hover {background:#1c2733!important;color:#f0f3f7!important;}
    [data-testid="stSidebar"] [class*="st-key-workspace_nav_"] button[kind="primary"] {background:#27313d!important;color:#fff!important;}
    [data-testid="stSidebar"] [class*="st-key-workspace_nav_"] button:focus-visible {outline:2px solid #8394a9!important;outline-offset:2px;}
    [data-testid="stSidebar"] [class*="st-key-workspace_nav_"] button p {font-size:13px!important;font-weight:500!important;}
    [data-testid="stSidebar"] [class*="st-key-workspace_nav_"] button::before {
      content:"";display:block;width:17px;height:17px;flex:0 0 17px;margin-right:10px;
      background:currentColor;opacity:.75;mask-size:contain;mask-repeat:no-repeat;
      -webkit-mask-size:contain;-webkit-mask-repeat:no-repeat;
    }
    [data-testid="stSidebar"] .st-key-workspace_nav_Analytics button::after {
      content:"";width:6px;height:6px;border-right:1.5px solid currentColor;border-bottom:1.5px solid currentColor;
      transform:rotate(45deg);margin-left:auto;margin-right:4px;opacity:.65;
    }
    [data-testid="stSidebar"] .st-key-workspace_nav_Analytics button[kind="primary"]::after {transform:rotate(225deg);}
    [data-testid="stSidebar"] [data-testid="stRadio"] {padding:0 4px 4px 27px;}
    [data-testid="stSidebar"] [role="radiogroup"] {gap:2px!important;}
    [data-testid="stSidebar"] [role="radiogroup"] label {
      padding:7px 9px!important;margin:0!important;border-radius:6px;background:transparent!important;
      border:0!important;min-height:32px;color:#aeb9c7!important;width:100%;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#1c2733!important;}
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:#202c39!important;color:#fff!important;}
    [data-testid="stSidebar"] [role="radiogroup"] label:focus-within {outline:1px solid #8394a9;}
    [data-testid="stSidebar"] [role="radiogroup"] label>div:first-child {display:none!important;}
    [data-testid="stSidebar"] [role="radiogroup"] label p {font-size:12px!important;}
    [data-testid="stSidebar"] .nav-section {color:#778597;font-size:10px;letter-spacing:.09em;margin:10px 0 5px 36px;text-transform:uppercase;}
    [data-testid="stSidebar"] .nav-divider {border-top:1px solid #29323d;margin:20px 0 9px;}
    [data-testid="stSidebar"] .nav-footer {color:#738194;font-size:10px;margin:20px 10px 6px;line-height:1.8;}
    """
    for name, path in _ICONS.items():
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{path}</svg>'
        uri = "data:image/svg+xml," + quote(svg, safe="")
        css += f'[data-testid="stSidebar"] .st-key-workspace_nav_{name} button::before {{mask-image:url("{uri}");-webkit-mask-image:url("{uri}");}}'
    return f"<style>{css}</style>"


def _render_brand() -> None:
    logo = Path(__file__).resolve().parents[1] / "assets" / "ai_framework_logo.png"
    image = ""
    if logo.is_file():
        encoded = b64encode(logo.read_bytes()).decode("ascii")
        image = f'<img src="data:image/png;base64,{encoded}" alt="TradeGPT logo">'
    st.markdown(
        f'<div class="nav-brand">{image}<div><strong>TradeGPT</strong>'
        '<small>Time Series Intelligence</small></div></div>',
        unsafe_allow_html=True,
    )


# ===== Existing Analytics routes =====
def _render_analytics_navigation() -> None:
    current = st.session_state.get("analytics_view", "Dashboard")
    # Separate visual widget keys avoid reusing emoji-valued state from the old sidebar.
    choice = st.radio(
        "Analytics navigation", ANALYTICS_ITEMS,
        index=ANALYTICS_ITEMS.index(current) if current in ANALYTICS_ITEMS else 0,
        label_visibility="collapsed", key="sidebar_clean_analytics",
    )
    st.session_state.analytics_view = choice
    if choice != "Models":
        return
    st.markdown('<div class="nav-section">Model families</div>', unsafe_allow_html=True)
    categories = list(MODEL_CATEGORIES)
    current_category = st.session_state.get("analytics_model_category", categories[0])
    category = st.radio(
        "Model families", categories,
        index=categories.index(current_category) if current_category in categories else 0,
        label_visibility="collapsed", key="sidebar_clean_category",
    )
    st.session_state.analytics_model_category = category
    variants = variants_for(category)
    current_variant = st.session_state.get("analytics_model_variant", variants[0])
    st.markdown('<div class="nav-section">Model variants</div>', unsafe_allow_html=True)
    variant = st.radio(
        "Model variants", variants,
        index=variants.index(current_variant) if current_variant in variants else 0,
        label_visibility="collapsed", key=f"sidebar_clean_variant_{category}",
    )
    st.session_state.analytics_model_variant = variant


def _select_page(page: str) -> None:
    st.session_state.sidebar_workspace_page = page


def render_sidebar() -> str:
    """Render the sidebar and return the existing router's plain page name."""
    if st.session_state.get("sidebar_workspace_page") not in NAV_ITEMS:
        st.session_state.sidebar_workspace_page = "Analytics"
    selected = st.session_state.sidebar_workspace_page
    with st.sidebar:
        st.markdown(_sidebar_css(), unsafe_allow_html=True)
        _render_brand()
        for page in NAV_ITEMS:
            if page == "Settings":
                st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
            st.button(
                page, key=f"workspace_nav_{page}",
                type="primary" if selected == page else "secondary",
                use_container_width=True, on_click=_select_page, args=(page,),
            )
            if page == "Analytics" and selected == "Analytics":
                _render_analytics_navigation()
        st.markdown(
            '<div class="nav-footer">v2.0.0 · Research workspace<br>Made with Streamlit</div>',
            unsafe_allow_html=True,
        )
    return selected
