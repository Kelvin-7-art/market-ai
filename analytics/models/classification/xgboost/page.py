"""XGBoost variant router."""
from market_ai.analytics.models.classification.xgboost.base_model_page import render_base_model
from market_ai.analytics.models.classification.xgboost.advanced_model_page import render_advanced_model
from market_ai.analytics.models.classification.xgboost.optimized_model_page import render_optimized_model


def render_xgboost(variant: str) -> None:
    """Render the requested interactive XGBoost workbench."""
    routes = {"Base Model": render_base_model, "Advanced Model": render_advanced_model,
              "Optimized Model": render_optimized_model}
    routes.get(variant, render_base_model)()
