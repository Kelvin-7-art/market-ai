"""Reusable input and navigation primitives."""

from __future__ import annotations

import streamlit as st


def search_bar(label: str, key: str, placeholder: str) -> str:
    """Render a compact search field."""
    return st.text_input(label, key=key, placeholder=placeholder, label_visibility="collapsed")


def filter_chips(label: str, options: list[str], key: str) -> str:
    """Render horizontally arranged filter chips using native radio controls."""
    return st.radio(label, options, key=key, horizontal=True, label_visibility="collapsed")


def tab_nav(label: str, options: list[str], key: str) -> str:
    """Render a vertical tab navigation list."""
    return st.radio(label, options, key=key, label_visibility="collapsed")