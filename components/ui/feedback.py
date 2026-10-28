"""Feedback, empty-state, and toast helpers."""

from __future__ import annotations

import streamlit as st

from market_ai.utils.helpers import html_card


def empty_state(icon: str, title: str, description: str) -> None:
    """Render a consistent empty state."""
    html_card(
        f'<div class="empty-state"><div class="empty-icon">{icon}</div><h3>{title}</h3><p>{description}</p></div>'
    )


def skeleton(lines: int = 3) -> None:
    """Render static skeleton lines for future async sections."""
    html_card(
        '<div class="skeleton-card">'
        + "".join('<div class="skeleton-line"></div>' for _ in range(lines))
        + "</div>"
    )


def toast_success(message: str) -> None:
    """Show an action confirmation using Streamlit's native toast."""
    st.toast(message, icon="✅")