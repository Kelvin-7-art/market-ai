"""Shared model-variant view and lightweight frontend model abstraction."""

from __future__ import annotations

import streamlit as st

from market_ai.analytics.export import build_envelope, render_download_button
from market_ai.components.charts import render_model_performance
from market_ai.components.ui.cards import stat_card
from market_ai.components.ui.feedback import toast_success
from market_ai.analytics.models.categories import MODEL_CATEGORIES, variants_for
from market_ai.utils.sample_data import model_data

SENTIMENT_VARIANTS = ["Transformer (FinBERT)", "Traditional ML", "Ensemble"]
TIME_SERIES_VARIANTS = ["AR", "MA", "ARMA", "ARIMA", "Prophet", "SARIMA / SARIMAX"]


def render_model_navigation(category: str, variant: str) -> None:
    """Render direct in-page navigation for model families and variants."""
    if category == "Chat AI":
        st.markdown(
            '<div class="model-page-navigation"><div class="caption">CHAT AI · ANALYST</div></div>',
            unsafe_allow_html=True,
        )
        return
    if category == "Sentiment" and not variant:
        current = st.session_state.get("sent_variant", SENTIMENT_VARIANTS[0])
        st.markdown('<div class="model-page-navigation"><div class="caption">SENTIMENT MODEL</div>', unsafe_allow_html=True)
        variant_cols = st.columns(len(SENTIMENT_VARIANTS))
        for column, option in zip(variant_cols, SENTIMENT_VARIANTS):
            with column:
                if st.button(
                    option,
                    key=f"sent_variant_nav_{option}",
                    type="primary" if option == current else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.sent_variant = option
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return
    if category == "Forecasting" and not variant:
        current = st.session_state.get("forecast_model", "ARIMA")
        st.markdown('<div class="model-page-navigation"><div class="caption">TIME-SERIES MODEL</div>', unsafe_allow_html=True)
        model_cols = st.columns(len(TIME_SERIES_VARIANTS))
        for column, option in zip(model_cols, TIME_SERIES_VARIANTS):
            with column:
                key = option.lower().replace(" ", "_").replace("/", "_")
                if st.button(
                    option,
                    key=f"forecast_model_nav_{key}",
                    type="primary" if option == current else "secondary",
                    width="stretch",
                ):
                    st.session_state.forecast_model = option
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return
    st.markdown('<div class="model-page-navigation"><div class="caption">MODEL WORKBENCH</div>', unsafe_allow_html=True)
    family_cols = st.columns(len(MODEL_CATEGORIES))
    for column, family, config in zip(family_cols, MODEL_CATEGORIES, MODEL_CATEGORIES.values()):
        with column:
            active = family == category
            if st.button(f'{config["icon"]} {family}', key=f"model_family_{family}", type="primary" if active else "secondary", use_container_width=True):
                st.session_state.analytics_model_category = family
                st.session_state.analytics_model_variant = variants_for(family)[0]
                st.rerun()
    variants = variants_for(category)
    variant_cols = st.columns(len(variants))
    for column, option in zip(variant_cols, variants):
        with column:
            if st.button(option, key=f"model_variant_{category}_{option}", type="primary" if option == variant else "secondary", use_container_width=True):
                st.session_state.analytics_model_category = category
                st.session_state.analytics_model_variant = option
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


class ConfiguredVariant:
    """Frontend-only model variant descriptor."""

    key = "base"
    label = "Base Model"

    def __init__(self, family: str) -> None:
        self.family = family

    @property
    def display_name(self) -> str:
        return f"{self.family} · {self.label}"


def render_variant_page(category: str, variant: str) -> None:
    """Render the shared research page for a selected model variant."""
    render_model_navigation(category, variant)
    data = model_data()
    row = data.iloc[0]
    if category == "XGBoost":
        row = data.iloc[4]
    elif category == "LSTM":
        row = data.iloc[2]
    elif category == "Transformer":
        row = data.iloc[3]
    elif category == "LightGBM":
        row = data.iloc[1]

    st.markdown(
        f'<div class="model-route-breadcrumb">Analytics <span>/</span> Models <span>/</span> {category} <span>/</span> <strong>{variant}</strong></div>',
        unsafe_allow_html=True,
    )
    title_col, action_col = st.columns([1, 0.38])
    with title_col:
        st.markdown(f'<div class="page-kicker">MODEL EXPERIMENT · {category.upper()}</div><h2>{category} / {variant}</h2><p class="page-subtitle">Evaluate the {variant.lower()} configuration and track its validation behavior.</p>', unsafe_allow_html=True)
    with action_col:
        if st.button("Run prediction", type="primary", use_container_width=True):
            toast_success(f"Prediction queued for {category} {variant}")
        if st.button("Compare variant", use_container_width=True):
            toast_success("Variant comparison ready")

    stats = st.columns(4)
    with stats[0]:
        stat_card("Accuracy", f'{float(row["accuracy"]):.1f}%', "+2.4% vs baseline", "◎", [82, 83, 85, 84, 87, float(row["accuracy"])], footer="Validation set")
    with stats[1]:
        stat_card("Precision", f'{float(row["precision"]):.1f}%', "Stable", "⌁", [79, 81, 82, 83, 84, float(row["precision"])], footer="Positive class")
    with stats[2]:
        stat_card("Recall", f'{float(row["recall"]):.1f}%', "+1.1% this run", "↗", [78, 80, 81, 84, 85, float(row["recall"])], footer="Positive class")
    with stats[3]:
        stat_card("Training time", str(row.get("training_time", "4.2s")), "−0.8s vs baseline", "ϟ", [7, 6, 5.5, 5, 4.6, 4.2], tone="positive", footer="CPU · 1,247 samples")

    chart_col, details_col = st.columns([1.5, 0.85], gap="large")
    with chart_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Validation performance</h3><span>Accuracy by experiment</span></div>', unsafe_allow_html=True)
        render_model_performance(data)
        st.markdown("</div>", unsafe_allow_html=True)
    with details_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Variant details</h3><span>Configuration</span></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="model-detail-list"><div><span>Family</span><strong>{category}</strong></div><div><span>Variant</span><strong>{variant}</strong></div><div><span>Status</span><strong class="active-text">Active</strong></div><div><span>Features</span><strong>46</strong></div><div><span>Inference</span><strong>12 ms</strong></div><div><span>Dataset</span><strong>Market factors</strong></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    metric_values = {
        "accuracy": float(row["accuracy"]) / 100,
        "precision": float(row["precision"]) / 100,
        "recall": float(row["recall"]) / 100,
        "f1": float(row["f1"]) / 100,
    }
    model_names = {
        "Random Forest": "RandomForestClassifier",
        "XGBoost": "XGBClassifier",
        "LightGBM": "LGBMClassifier",
        "SVM": "SVC",
        "Neural Network": "NeuralNetworkClassifier",
        "LSTM": "LSTMClassifier",
        "Transformer": "TransformerClassifier",
    }
    envelope = build_envelope(
        workbench="classification",
        model_name=model_names.get(category, category),
        model_variant=variant,
        request={
            "ticker": None,
            "market": "Market factors",
            "model_category": category,
            "model_variant": variant,
        },
        inputs={
            "source": "Built-in model registry demonstration data",
            "registry_record": str(row["name"]),
            "feature_count": int(row["features_count"]),
        },
        results={
            "displayed_registry_metrics": {
                "name": str(row["name"]),
                "category": str(row["category"]),
                "status": str(row["status"]),
                "last_trained": str(row["last_trained"]),
                "owner": str(row["owner"]),
            },
            "feature_count": int(row["features_count"]),
        },
        metrics=metric_values,
        diagnostics={
            "data_kind": "static demonstration values; no model fit was run on this page",
        },
        caveats=[
            "This page displays built-in demonstration metrics; it is not a model trained on the currently selected market data.",
            "Reported accuracy is on a single chronological holdout; out-of-sample performance may differ.",
            "Class balance in the test window affects accuracy, precision, and recall; always compare against a majority-class baseline.",
            "Skill score near zero means the model has no reliable edge over a naive baseline.",
            "Feature importances are model-specific and not causal.",
            "This is informational only and not investment advice.",
        ],
    )
    classification_key = category.lower().replace(" ", "_")
    variant_key = variant.lower().replace(" ", "_")
    render_download_button(
        envelope,
        key=f"classification_{classification_key}_{variant_key}_json",
    )


__all__ = ["ConfiguredVariant", "render_variant_page"]