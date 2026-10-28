"""Transformer variant router."""
from market_ai.analytics.models.classification.transformer.base_model_page import render_base_model
from market_ai.analytics.models.classification.transformer.fine_tuned_model_page import render_fine_tuned_model
from market_ai.analytics.models.classification.transformer.optimized_model_page import render_optimized_model


def render_transformer(variant: str) -> None:
    """Render a Transformer page through the existing Analytics router."""
    pages = {"Base Model": render_base_model, "Fine-tuned Model": render_fine_tuned_model,
             "Optimized Model": render_optimized_model}
    pages.get(variant, render_base_model)()
