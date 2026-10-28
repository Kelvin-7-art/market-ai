"""JSON paste validation and adapters for legacy and analytics-export payloads."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import streamlit as st


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def normalize_context_payload(payload: Any) -> dict[str, Any]:
    """Normalize ChatRequest JSON, legacy model JSON, or a shared export envelope."""
    if not isinstance(payload, Mapping):
        raise ValueError("The JSON root must be an object.")

    root = _mapping(payload)
    context = _mapping(root.get("context"))
    if context:
        root = context

    # Accept an already-canonical ChatRequest context.
    if isinstance(root.get("market"), Mapping) and isinstance(root.get("metrics"), Mapping):
        market = _mapping(root.get("market"))
        training = _mapping(root.get("training"))
        metrics = _mapping(root.get("metrics"))
        forecast = _mapping(root.get("forecast"))
        if "available" not in forecast:
            forecast["available"] = False
        return {
            "market": market,
            "training": training,
            "metrics": metrics,
            "forecast": forecast,
            "feature_importance_top": root.get("feature_importance_top", []),
            "warnings": root.get("warnings", []),
        }

    app = _mapping(root.get("app"))
    request = _mapping(root.get("request"))
    inputs = _mapping(root.get("inputs"))
    results = _mapping(root.get("results"))
    diagnostics = _mapping(root.get("diagnostics"))
    metadata = _mapping(root.get("metadata"))
    source_market = _mapping(root.get("market"))
    source_training = _mapping(root.get("training"))
    source_metrics = _mapping(root.get("metrics"))

    market = {
        "ticker": _first(
            source_market.get("ticker"),
            request.get("ticker"),
            inputs.get("ticker"),
            metadata.get("ticker"),
            "Unknown",
        ),
        "market_label": _first(
            source_market.get("market_label"),
            request.get("market") if isinstance(request.get("market"), str) else None,
            metadata.get("market"),
            inputs.get("market"),
        ),
        "interval": _first(
            source_market.get("interval"),
            request.get("interval"),
            request.get("sampling_interval"),
            inputs.get("interval"),
            metadata.get("interval"),
            "1d",
        ),
        "start_date": _first(
            source_market.get("start_date"),
            request.get("start_date"),
            request.get("start"),
            _mapping(inputs.get("date_range")).get("start"),
            metadata.get("start"),
        ),
        "end_date": _first(
            source_market.get("end_date"),
            request.get("end_date"),
            request.get("end"),
            _mapping(inputs.get("date_range")).get("end"),
            metadata.get("end"),
        ),
        "last_close": _number(
            _first(
                source_market.get("last_close"),
                metadata.get("last_close"),
                results.get("starting_price"),
            )
        ),
        "period_return": _number(
            _first(
                source_market.get("period_return"),
                metadata.get("period_return"),
                inputs.get("period_return"),
            )
        ),
    }

    model_name = _first(
        _mapping(source_training).get("model_name"),
        app.get("model_name"),
        root.get("model"),
        results.get("model"),
    )
    training = {
        "model_name": model_name,
        "preset": _first(
            source_training.get("preset"),
            request.get("preset"),
            app.get("model_variant"),
            root.get("variant"),
        ),
        "cv_splits": _first(source_training.get("cv_splits"), request.get("cv_splits")),
        "scoring": _first(source_training.get("scoring"), request.get("scoring")),
        "best_params": _first(
            source_training.get("best_params"),
            results.get("best_params"),
            request.get("best_params"),
        ),
        "parameters": _first(
            source_training.get("parameters"),
            request.get("parameters"),
            results.get("model_parameters"),
        ),
        "training_time_seconds": _number(
            _first(
                source_training.get("training_time_seconds"),
                source_metrics.get("training_time_seconds"),
                source_metrics.get("training_time"),
            )
        ),
    }

    metrics = {
        name: source_metrics.get(name)
        for name in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "best_cv_score",
            "skill_score",
            "low_skill",
        )
        if name in source_metrics
    }

    if "forecast" in root and isinstance(root.get("forecast"), Mapping):
        source_forecast = _mapping(root.get("forecast"))
    else:
        source_forecast = _mapping(
            _first(
                results.get("forecast"),
                results.get("recursive_price_forecast"),
                results if results.get("path") else None,
            )
        )
    path = source_forecast.get("path")
    if not isinstance(path, list):
        path = []
    available = source_forecast.get("available")
    if available is None:
        available = bool(
            path
            or _first(
                source_forecast.get("point_final"),
                source_forecast.get("final_price"),
            )
            is not None
        )
    if source_forecast.get("available") is False:
        available = False

    probabilities = [
        _number(_mapping(row).get("bullish_probability"))
        for row in path
        if _number(_mapping(row).get("bullish_probability")) is not None
    ]
    bullish_days = sum(
        str(_mapping(row).get("predicted_trend", "")).lower() == "bullish"
        for row in path
    )
    bearish_days = sum(
        str(_mapping(row).get("predicted_trend", "")).lower() == "bearish"
        for row in path
    )
    forecast = {
        "available": bool(available),
        "horizon_days": _first(
            source_forecast.get("horizon_days"),
            source_forecast.get("horizon_sessions"),
            len(path) or None,
        ),
        "starting_price": _number(source_forecast.get("starting_price")),
        "final_price": _number(
            _first(source_forecast.get("final_price"), source_forecast.get("point_final"))
        ),
        "total_return": _number(
            _first(source_forecast.get("total_return"), source_forecast.get("return_pct"))
        ),
        "bullish_days": _first(
            source_forecast.get("bullish_days"),
            bullish_days if path else None,
        ),
        "bearish_days": _first(
            source_forecast.get("bearish_days"),
            bearish_days if path else None,
        ),
        "avg_bullish_probability": _number(
            _first(
                source_forecast.get("avg_bullish_probability"),
                sum(probabilities) / len(probabilities) if probabilities else None,
            )
        ),
    }
    feature_rows = _first(
        root.get("feature_importance_top"),
        results.get("feature_importance_top"),
        diagnostics.get("feature_importances"),
        [],
    )
    if not isinstance(feature_rows, list):
        feature_rows = []

    warning_items = _first(
        root.get("warnings"),
        diagnostics.get("warnings"),
        root.get("caveats"),
        [],
    )
    if not isinstance(warning_items, list):
        warning_items = [str(warning_items)]
    warnings = [str(item) for item in warning_items]

    return {
        "market": market,
        "training": training,
        "metrics": metrics,
        "forecast": forecast,
        "feature_importance_top": feature_rows,
        "warnings": warnings,
    }


def render_json_paste() -> dict[str, Any] | None:
    """Render the manual JSON input and return a normalized context if valid."""
    raw = st.text_area(
        "Paste analysis JSON",
        height=240,
        placeholder='{"schema_version":"1.0","app":{...},"results":{...}}',
        key="chat_json_paste",
    )
    if not raw.strip():
        st.caption(
            "Paste a structured ChatRequest, a legacy model result, or an "
            "AI Market Assistant analysis JSON export."
        )
        return None
    try:
        normalized = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON constant: {value}")))
        if normalized is None:
            normalized = {"value": None}
    except (json.JSONDecodeError, ValueError, TypeError) as error:
        st.error(f"Could not validate the JSON: {error}")
        return None
    st.success("JSON ready. All fields are preserved for the assistant.")
    return normalized


__all__ = ["normalize_context_payload", "render_json_paste"]
