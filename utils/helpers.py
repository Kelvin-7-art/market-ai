"""Small rendering helpers used across the application."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable

import streamlit as st


def html_card(content: str, classes: str = "panel") -> None:
    """Render a raw HTML card with a shared class."""
    st.markdown(f'<div class="{classes}">{content}</div>', unsafe_allow_html=True)


def sparkline(points: Iterable[float], color: str = "#31D7EE") -> str:
    """Return a compact inline SVG sparkline."""
    values = list(points)
    if not values:
        return ""
    low, high = min(values), max(values)
    spread = high - low or 1
    coords = []
    for index, value in enumerate(values):
        x = (index / max(len(values) - 1, 1)) * 100
        y = 28 - ((value - low) / spread) * 24
        coords.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg class="sparkline" viewBox="0 0 100 32" aria-hidden="true">'
        f'<polyline points="{" ".join(coords)}" style="stroke:{color}"/></svg>'
    )


def format_timestamp() -> str:
    """Return the current local timestamp for the interface."""
    return datetime.now().strftime("%d %b %Y · %H:%M")


def status_badge(status: str) -> str:
    """Render a model status badge."""
    class_name = status.lower().replace(" ", "-")
    return f'<span class="badge badge-{class_name}">{status}</span>'