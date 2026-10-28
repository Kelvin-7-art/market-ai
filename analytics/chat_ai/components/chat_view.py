"""Render stored user and assistant messages."""

from __future__ import annotations

import streamlit as st


def render_chat_view(history: list[dict[str, object]]) -> None:
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        '<span>💬</span><div><h3>Conversation</h3>'
        "<p>Messages are kept in this Streamlit session.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    if not history:
        st.caption("Your conversation will appear here after the first response.")
    else:
        for message in history:
            role = message.get("role", "assistant")
            if role not in {"user", "assistant"}:
                continue
            with st.chat_message(str(role)):
                st.markdown(str(message.get("content", "")))
    st.markdown("</div>", unsafe_allow_html=True)
