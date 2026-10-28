"""Application footer."""

from __future__ import annotations

import streamlit as st


def render_footer() -> None:
    """Render a lightweight product footer."""
    st.markdown(
        """
        <div class="footer">
          <div><strong>TradeGPT</strong> · Time Series Intelligence · v2.0.0</div>
          <div>Documentation ↗ &nbsp;&nbsp; GitHub ↗ &nbsp;&nbsp; All systems operational</div>
        </div>
        """,
        unsafe_allow_html=True,
    )