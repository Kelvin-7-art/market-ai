"""Application header."""

from __future__ import annotations

from html import escape

import streamlit as st

from market_ai.utils.branding import logo_html

from market_ai.utils.helpers import format_timestamp


def render_header(page: str) -> None:
    """Render the page title and live status."""
    title = escape(page)
    if page == "Dashboard":
        title_html = (
            '<div style="display:flex;align-items:center;gap:14px;'
            'margin:0 0 1rem;">'
            f'{logo_html(58)}<h1 style="margin:0;">{title}</h1></div>'
        )
    else:
        title_html = f"<h1>{title}</h1>"
    st.markdown(
        f"""
        <div class="topbar">
          <div class="topbar-copy">
            <div class="eyebrow">TradeGPT / {title}</div>
            {title_html}
            <p>Time series intelligence for faster, clearer market decisions.</p>
          </div>
          <div class="status-pill"><span class="status-dot"></span> Systems nominal · {format_timestamp()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )