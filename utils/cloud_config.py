"""Shared environment and Streamlit secrets settings."""
import os

def setting(name: str, default: str = "") -> str:
    """Read an environment variable or a top-level Streamlit secret."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default)).strip()
    except (FileNotFoundError, KeyError):
        return default
