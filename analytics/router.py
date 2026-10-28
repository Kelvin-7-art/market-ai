"""Main Analytics router."""

from __future__ import annotations

import streamlit as st

from market_ai.components.analytics import render_dashboard
from market_ai.components.chat import render_chat
from market_ai.analytics.chat_ai.page import render_chat_ai_page
from market_ai.analytics.models.classification.page import render_classification_model
from market_ai.analytics.forecasts.page import render_forecasts
from market_ai.analytics.sentiment.page import render_sentiment


def render_analytics(view: str, model_category: str | None = None, model_variant: str | None = None) -> None:
    """Render the selected Analytics destination."""
    if view == "Dashboard":
        render_dashboard()
    elif view == "Chat":
        render_chat()
    elif view == "Chat AI":
        render_chat_ai_page()
    elif view == "Models":
        render_classification_model(model_category or "Random Forest", model_variant or "Base Model")
    elif view == "Sentiment":
        render_sentiment()
    elif view == "Forecasting":
        render_forecasts()
    else:
        st.markdown(
            f'<div class="empty-state"><div class="empty-icon">◇</div><h3>{view} workspace</h3><p>Build, compare and explain {view.lower()} workflows in a focused research workspace.</p></div>',
            unsafe_allow_html=True,
        )


__all__ = ["render_analytics"]