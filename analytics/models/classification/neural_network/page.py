"""Neural Network variant router."""
from market_ai.analytics.models.classification.neural_network.base_model_page import render_base_model
from market_ai.analytics.models.classification.neural_network.deep_model_page import render_deep_model
from market_ai.analytics.models.classification.neural_network.optimized_model_page import render_optimized_model


def render_neural_network(variant: str) -> None:
    """Render a Neural Network page using the existing Analytics router."""
    pages = {"Base Model": render_base_model, "Deep Model": render_deep_model,
             "Optimized Model": render_optimized_model}
    pages.get(variant, render_base_model)()
