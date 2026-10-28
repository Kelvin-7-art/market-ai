"""Reusable presentation cards."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st

from market_ai.utils.helpers import html_card, sparkline


def badge(label: str, tone: str = "info") -> str:
    """Return a styled badge fragment."""
    return f'<span class="badge badge-{tone}">{label}</span>'


def avatar_stack(names: Iterable[str]) -> str:
    """Return a compact initials avatar stack."""
    initials = "".join(
        f'<span class="avatar" title="{name}">{name[:1].upper()}</span>'
        for name in list(names)[:4]
    )
    return f'<span class="avatar-stack">{initials}</span>'


def stat_card(
    label: str,
    value: str,
    delta: str,
    icon: str,
    points: list[float] | None = None,
    tone: str = "positive",
    footer: str = "",
) -> None:
    """Render a reusable stat card."""
    html_card(
        f"""
        <div class="stat-card">
          <div class="stat-card-top"><span class="stat-icon">{icon}</span><span class="caption">{label}</span></div>
          <div class="stat-value">{value}</div>
          <div class="stat-delta {tone}">{delta}</div>
          {sparkline(points or [1, 2, 1.5, 3, 3.5, 4], "#31D7EE")}
          <div class="stat-footer">{footer}</div>
        </div>
        """,
        classes="stat-card-wrap",
    )


def setting_row(label: str, description: str, control: str) -> None:
    """Render a setting row with an already-rendered control."""
    st.markdown(
        f'<div class="setting-row"><div><div class="setting-label">{label}</div><div class="setting-description">{description}</div></div><div class="setting-control">{control}</div></div>',
        unsafe_allow_html=True,
    )