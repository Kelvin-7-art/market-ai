"""Show the exact last prompt sent or previewed."""

from __future__ import annotations

import streamlit as st


def render_prompt_preview(system_prompt: str, user_prompt: str) -> None:
    if not system_prompt and not user_prompt:
        return
    with st.expander("📋 Full prompt sent (copy-paste ready)", expanded=False):
        st.markdown("**System prompt**")
        st.code(system_prompt, language="text")
        st.markdown("**User prompt**")
        st.code(user_prompt, language="json")
