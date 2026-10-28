"""Shared theme helpers for Plotly and Streamlit."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


COLORS = {
    "bg": "#090B12",
    "panel": "#111827",
    "panel_alt": "#141C2B",
    "text": "#F8FAFC",
    "muted": "#8C97AB",
    "cyan": "#31D7EE",
    "blue": "#4C8DFF",
    "violet": "#9C6BFF",
    "green": "#3FE0A0",
    "yellow": "#F6C868",
    "red": "#FF6F91",
    "grid": "rgba(148,163,184,0.10)",
}


def load_css() -> None:
    """Load the app stylesheet from the local assets directory."""
    css_path = Path(__file__).resolve().parents[1] / "assets" / "styles.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)