"""LSTM variant router."""
from market_ai.analytics.models.classification.lstm.base_model_page import render_base_model
from market_ai.analytics.models.classification.lstm.tuned_model_page import render_tuned_model
from market_ai.analytics.models.classification.lstm.optimized_model_page import render_optimized_model


def render_lstm(variant: str) -> None:
    """Render an LSTM page through the existing Analytics router."""
    pages = {"Base Model": render_base_model, "Tuned Model": render_tuned_model,
             "Optimized Model": render_optimized_model}
    pages.get(variant, render_base_model)()
