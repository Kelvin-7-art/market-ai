"""Random Forest variant router."""

from market_ai.analytics.models.classification.random_forest.advanced_model_page import render_advanced_model
from market_ai.analytics.models.classification.random_forest.base_model_page import render_base_model
from market_ai.analytics.models.classification.random_forest.fast_model_page import render_fast_model
from market_ai.analytics.models.classification.random_forest.optimized_model_page import render_optimized_model
from market_ai.analytics.models.classification.random_forest.smote_model_page import render_smote_model


def render_random_forest(variant: str) -> None:
    {
        "Base Model": render_base_model,
        "Advanced Model": render_advanced_model,
        "Optimized Model": render_optimized_model,
        "Fast Model": render_fast_model,
        "With SMOTE": render_smote_model,
    }.get(variant, render_base_model)()