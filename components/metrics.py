"""Metric tile components."""

from __future__ import annotations

import streamlit as st

from market_ai.utils.helpers import sparkline


def render_metric(label: str, value: str, delta: str, points: list[float], tone: str = "positive") -> None:
    """Render a premium metric card with an inline sparkline."""
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-delta {tone}">{delta}</div>
          {sparkline(points)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_row() -> None:
    """Render dashboard KPIs."""
    cols = st.columns(4)
    data = [
        ("Total models", "18", "+3 this quarter", [4, 5, 4, 7, 8, 9, 12, 16, 18], "positive"),
        ("Portfolio value", "$12.48M", "+8.24% YTD", [8, 9, 8, 11, 12, 14, 13, 17, 20], "positive"),
        ("Risk score", "32.4", "−4.8 vs. last week", [17, 16, 18, 15, 14, 14, 12, 11, 10], "positive"),
        ("Prediction accuracy", "94.8%", "+2.1% vs. baseline", [11, 12, 12, 13, 14, 15, 15, 17, 19], "positive"),
    ]
    for col, item in zip(cols, data):
        with col:
            render_metric(*item)