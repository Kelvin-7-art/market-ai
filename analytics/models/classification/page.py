"""Classification model-family router."""

from __future__ import annotations

from market_ai.analytics.models.classification.lightgbm.page import render_lightgbm
from market_ai.analytics.models.classification.lstm.page import render_lstm
from market_ai.analytics.models.classification.neural_network.page import render_neural_network
from market_ai.analytics.models.classification.random_forest.page import render_random_forest
from market_ai.analytics.models.classification.svm.page import render_svm
from market_ai.analytics.models.classification.transformer.page import render_transformer
from market_ai.analytics.models.classification.xgboost.page import render_xgboost


ROUTERS = {
    "Random Forest": render_random_forest,
    "XGBoost": render_xgboost,
    "LightGBM": render_lightgbm,
    "SVM": render_svm,
    "Neural Network": render_neural_network,
    "LSTM": render_lstm,
    "Transformer": render_transformer,
}


def render_classification_model(category: str, variant: str) -> None:
    """Dispatch a classification model family to its page router."""
    from market_ai.utils.cloud_config import setting
    if setting("CLOUD_MODE", "false").lower() == "true" and category in {"LSTM", "Transformer"}:
        import streamlit as st
        st.subheader(f"{category} / {variant}")
        st.info("This training model remains available in the desktop app. The free cloud profile omits PyTorch to reduce memory use. Upload its exported JSON to Chat AI for interpretation.")
        return
    ROUTERS.get(category, render_random_forest)(variant)


__all__ = ["render_classification_model"]