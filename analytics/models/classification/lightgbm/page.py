"""LightGBM variant router."""
from market_ai.analytics.models.classification.lightgbm.base_model_page import render_base_model
from market_ai.analytics.models.classification.lightgbm.advanced_model_page import render_advanced_model
from market_ai.analytics.models.classification.lightgbm.optimized_model_page import render_optimized_model


def render_lightgbm(variant: str) -> None:
    """Render the requested LightGBM page using the existing router contract."""
    pages = {
        "Base Model": render_base_model,
        "Advanced Model": render_advanced_model,
        "Optimized Model": render_optimized_model,
    }
    pages.get(variant, render_base_model)()
