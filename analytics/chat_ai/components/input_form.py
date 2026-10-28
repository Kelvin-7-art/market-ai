"""Controls for the analyst task, tone, detail, and focus."""

from __future__ import annotations

from typing import Any

import streamlit as st


TASK_TYPES = [
    "Explain this JSON",
    "Forecast review",
    "Model diagnostic",
    "Feature analysis",
    "Risk assessment",
    "Custom question",
]
TONES = ["Conversational", "Analyst", "Executive", "Technical", "Casual"]
DETAIL_LEVELS = ["Standard", "Concise", "Deep dive"]
FOCUS_AREAS = [
    "Model quality",
    "Feature drivers",
    "Forecast",
    "Risks",
    "Recommendations",
]


def render_input_form() -> dict[str, Any]:
    """Render structured prompt controls and return their current values."""
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        '<span>🎛️</span><div><h3>Ask configuration</h3>'
        "<p>Choose the analyst’s task, tone, depth, and focus.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    row = st.columns([1.25, 1.1, 1, 1])
    task_type = row[0].selectbox("Task type", TASK_TYPES, key="chat_task_type")
    tone = row[1].selectbox("Tone", TONES, key="chat_tone")
    detail_level = row[2].selectbox(
        "Detail level",
        DETAIL_LEVELS,
        key="chat_detail_level",
    )
    analysis_horizon_days = row[3].number_input(
        "Review horizon (sessions)",
        min_value=5,
        max_value=730,
        value=160,
        step=5,
        key="chat_horizon",
    )
    focus_areas = st.multiselect(
        "Focus areas",
        FOCUS_AREAS,
        default=["Model quality", "Feature drivers", "Forecast", "Risks"],
        key="chat_focus_areas",
    )
    custom_question = st.text_area(
        "Additional question (optional)",
        height=80,
        placeholder="e.g. Which feature most influenced this run?",
        key="chat_custom_question",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    return {
        "task_type": task_type,
        "tone": tone,
        "detail_level": detail_level,
        "analysis_horizon_days": int(analysis_horizon_days),
        "focus_areas": focus_areas,
        "custom_question": custom_question,
    }
