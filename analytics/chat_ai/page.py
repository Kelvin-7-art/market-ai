"""Interactive Chat AI workbench for explaining structured market-model outputs."""

from __future__ import annotations

import time
import base64
import html
from pathlib import Path
from typing import Any

import streamlit as st

from market_ai.analytics.chat_ai.components.chat_view import render_chat_view
from market_ai.analytics.chat_ai.components.input_form import render_input_form
from market_ai.analytics.chat_ai.components.json_paste import (
    normalize_context_payload,
    render_json_paste,
)
from market_ai.analytics.chat_ai.components.prompt_preview import (
    render_prompt_preview,
)
from market_ai.analytics.chat_ai.input_builder import build_prompts
from market_ai.analytics.chat_ai.llm_client import (
    PROVIDER,
    is_configured,
    provider_status,
    stream_response,
)
from market_ai.analytics.chat_ai.schema import ChatRequest
from market_ai.analytics.models.variant import render_model_navigation
from market_ai.components.ui.cards import stat_card
from market_ai.components.ui.feedback import toast_success
from market_ai.utils.theme import COLORS


PROVIDERS = ["groq_primary", "groq_secondary", "openai", "anthropic", "claude"]
PROVIDER_LABELS = {
    "groq_primary": "Groq Primary",
    "groq_secondary": "Groq Secondary",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "claude": "Claude",
}


def _ensure_state() -> None:
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("chat_last_prompt_system", "")
    st.session_state.setdefault("chat_last_prompt_user", "")
    st.session_state.setdefault("chat_last_context_source", "last_run")
    st.session_state.setdefault("chat_provider", PROVIDER if PROVIDER in PROVIDERS else "openai")


def _reset_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.chat_last_prompt_system = ""
    st.session_state.chat_last_prompt_user = ""
    toast_success("Conversation cleared")


def _latest_rf_context() -> dict[str, Any] | None:
    """Adapt the project's most recent trained Random Forest state."""
    if not st.session_state.get("rf_trained"):
        return None
    metrics = st.session_state.get("rf_metrics")
    meta = st.session_state.get("rf_market_meta")
    if not isinstance(metrics, dict) or not isinstance(meta, dict):
        return None

    model_params = {
        key: metrics[key]
        for key in ("estimators", "max_depth", "class_weight", "random_state")
        if key in metrics
    }
    forecast = st.session_state.get("rf_forecast")
    has_forecast = (
        forecast is not None
        and hasattr(forecast, "empty")
        and not forecast.empty
        and "Forecast_Price" in forecast.columns
    )
    forecast_context: dict[str, Any] = {"available": bool(has_forecast)}
    last_close = meta.get("last_close")
    if has_forecast:
        forecast_values = forecast["Forecast_Price"].astype(float)
        starting_price = float(last_close) if last_close is not None else float(forecast_values.iloc[0])
        final_price = float(forecast_values.iloc[-1])
        trends = (
            forecast["Predicted_Trend"].astype(str).str.lower()
            if "Predicted_Trend" in forecast.columns
            else None
        )
        probabilities = (
            forecast["Bullish_Probability"].astype(float)
            if "Bullish_Probability" in forecast.columns
            else None
        )
        forecast_context.update(
            {
                "horizon_days": int(len(forecast)),
                "starting_price": starting_price,
                "final_price": final_price,
                "total_return": (
                    final_price / starting_price - 1 if starting_price else None
                ),
                "bullish_days": (
                    int((trends == "bullish").sum()) if trends is not None else None
                ),
                "bearish_days": (
                    int((trends == "bearish").sum()) if trends is not None else None
                ),
                "avg_bullish_probability": (
                    float(probabilities.mean()) if probabilities is not None else None
                ),
            }
        )

    importance = st.session_state.get("rf_feature_importance")
    importance_rows: list[dict[str, Any]] = []
    if importance is not None and hasattr(importance, "empty") and not importance.empty:
        feature_column = next(
            (column for column in ("feature", "Feature") if column in importance.columns),
            None,
        )
        importance_column = next(
            (column for column in ("importance", "Builtin_Importance") if column in importance.columns),
            None,
        )
        if feature_column and importance_column:
            top = importance.sort_values(importance_column, ascending=False).head(10)
            importance_rows = [
                {
                    "feature": str(row[feature_column]),
                    "importance": float(row[importance_column]),
                }
                for _, row in top.iterrows()
            ]

    warnings = []
    if not forecast_context["available"]:
        warnings.append("No price forecast is available for the latest Random Forest run.")
    if metrics.get("low_skill") is None and metrics.get("skill_score") is None:
        warnings.append("A low-skill baseline assessment was not recorded for this run.")
    for key in ("rf_market_error", "rf_training_error"):
        if st.session_state.get(key):
            warnings.append(str(st.session_state[key]))

    return {
        "market": {
            "ticker": str(meta.get("ticker", "Unknown")),
            "market_label": meta.get("market", "Selected market"),
            "interval": meta.get("interval", "1d"),
            "start_date": meta.get("start"),
            "end_date": meta.get("end"),
            "last_close": last_close,
            "period_return": meta.get("period_return"),
        },
        "training": {
            "model_name": "RandomForestClassifier · Base Model",
            "preset": "Base Model",
            "best_params": metrics.get("best_params"),
            "parameters": model_params,
            "training_time_seconds": metrics.get("training_time"),
        },
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "best_cv_score": metrics.get("best_cv_score"),
            "skill_score": metrics.get("skill_score"),
            "low_skill": metrics.get("low_skill"),
        },
        "forecast": forecast_context,
        "feature_importance_top": importance_rows,
        "warnings": warnings,
    }


def _context_section() -> dict[str, Any] | None:
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        '<span>📥</span><div><h3>Context source</h3>'
        "<p>Use the latest trained Random Forest run or paste structured JSON.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    source_options = ["Use last Random Forest run", "Paste JSON manually"]
    current_index = 0 if st.session_state.chat_last_context_source == "last_run" else 1
    selected = st.radio(
        "Analysis context",
        source_options,
        index=current_index,
        horizontal=True,
        key="chat_context_source",
    )
    st.session_state.chat_last_context_source = (
        "last_run" if selected == source_options[0] else "manual"
    )

    context: dict[str, Any] | None = None
    if st.session_state.chat_last_context_source == "last_run":
        context = _latest_rf_context()
        if context is None:
            st.warning(
                "No trained Random Forest run is available in this session. "
                "Run the Base Model first, or paste JSON manually."
            )
        else:
            model_name = context["training"]["model_name"]
            cols = st.columns(4)
            cols[0].metric("Ticker", context["market"]["ticker"])
            cols[1].metric("Model", model_name)
            accuracy = context["metrics"]["accuracy"]
            cols[2].metric(
                "Accuracy",
                f"{float(accuracy):.1%}" if accuracy is not None else "—",
            )
            cols[3].metric(
                "Forecast",
                "Available" if context["forecast"]["available"] else "Suppressed",
            )
    else:
        context = render_json_paste()
    st.markdown("</div>", unsafe_allow_html=True)
    return context


def _build_chat_request(context: dict[str, Any], form: dict[str, Any]) -> ChatRequest:
    normalized = normalize_context_payload(context)
    request_data = {**normalized, **form}
    return ChatRequest.model_validate(request_data)


def _provider_logo(provider: str) -> str:
    """Embed the selected supplied artwork without changing the source image."""
    if provider in {"groq_primary", "groq_secondary"}:
        return '<span style="font-size:18px;font-weight:700;color:#f55036">Groq</span>'
    name = provider if provider in PROVIDERS else "openai"
    path = Path(__file__).resolve().parents[2] / "assets" / "providers" / f"{name}.webp"
    if not path.is_file():
        return html.escape(PROVIDER_LABELS[name])
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return (f'<img src="data:image/webp;base64,{encoded}" alt="{PROVIDER_LABELS[name]}" '
            'style="display:block;width:112px;height:64px;object-fit:contain;background:white;border-radius:8px;">')


def _provider_controls() -> str:
    if st.session_state.chat_provider not in PROVIDERS:
        st.session_state.chat_provider = PROVIDER if PROVIDER in PROVIDERS else "openai"
    selected = st.selectbox(
        "Provider",
        PROVIDERS,
        format_func=lambda value: PROVIDER_LABELS[value],
        key="chat_provider",
        label_visibility="collapsed",
    )
    st.caption(provider_status(selected) if is_configured(selected) else "API key missing. Configure this provider to enable replies.")
    if selected in {"groq_primary", "groq_secondary"}:
        st.caption("Uses the selected Groq credential. Switching keys is manual; no automatic key rotation.")
    if selected in {"anthropic", "claude"}:
        st.caption("Anthropic and Claude use the same Anthropic API key; their model settings can differ.")
    st.caption(
        "The prompt includes the selected analysis data. The selected remote provider receives it when you send."
    )
    return selected


def _send_reply(context: Any, form: dict[str, Any], provider: str,
                question: str | None = None) -> None:
    """Stream a reply and commit a complete turn only after success."""
    if not is_configured(provider):
        st.warning("Configure the selected provider API key before sending.")
        return
    try:
        system, user = build_prompts(context, form, question)
    except (ValueError, TypeError) as error:
        st.error(str(error))
        return
    st.session_state.chat_last_prompt_system = system
    st.session_state.chat_last_prompt_user = user
    placeholder = st.empty()
    response = ""
    try:
        for chunk in stream_response(system, user, provider=provider,
                                     history=list(st.session_state.chat_history)):
            response += chunk
            placeholder.markdown(response + "▌")
        if not response.strip():
            raise ValueError("The provider returned an empty response.")
    except Exception as error:
        placeholder.empty()
        st.error(f"Response failed: {error}. Your conversation has been preserved; you can retry.")
        return
    placeholder.empty()
    text = question or form.get("custom_question") or "Explain this JSON."
    st.session_state.chat_history.extend([
        {"role": "user", "content": text, "ts": time.time()},
        {"role": "assistant", "content": response, "ts": time.time()},
    ])
    st.rerun()


def _send_button_row(context: Any, form: dict[str, Any], provider: str) -> None:
    """Preview the complete prompt or request a natural explanation."""
    preview, send = st.columns(2)
    if preview.button("Preview prompt", disabled=context is None):
        try:
            system, user = build_prompts(context, form)
            st.session_state.chat_last_prompt_system = system
            st.session_state.chat_last_prompt_user = user
        except (ValueError, TypeError) as error:
            st.error(str(error))
    if send.button("Explain JSON", type="primary",
                   disabled=context is None or not is_configured(provider)):
        _send_reply(context, form, provider)


def render_chat_ai_page() -> None:
    """Render the Chat AI page and its session-local conversation."""
    _ensure_state()
    render_model_navigation("Chat AI", "Analyst")

    meta = st.session_state.get("rf_market_meta", {})
    ticker = meta.get("ticker", "—") if isinstance(meta, dict) else "—"
    title_col, action_col = st.columns([1.45, 1])
    with title_col:
        st.markdown(
            '<div class="page-kicker">CHAT AI · STRUCTURED ANALYSIS</div>'
            "<h2>💬 Chat AI Analyst</h2>"
            '<p class="page-subtitle">Interpret a Random Forest run or paste an '
            "analysis export. No market model is rerun here.</p>",
            unsafe_allow_html=True,
        )
    with action_col:
        provider = _provider_controls()
        configured = is_configured(provider)
        status_color = COLORS["green"] if configured or provider == "local" else COLORS["red"]
        if provider == "local":
            status_text = "● Local echo"
        elif configured:
            status_text = "● Ready"
        else:
            status_text = "API key missing"
        st.markdown(
            '<div class="workbench-badges">'
            f'<span class="badge badge-active" style="color:{status_color}">{status_text}</span>'
            f'<span class="badge badge-info">{ticker}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("↻ Reset conversation", use_container_width=True, key="chat_reset"):
            _reset_chat()

    st.markdown("<style>.stat-icon:has(img[alt]) {width:auto!important;height:auto!important;background:transparent!important;}</style>", unsafe_allow_html=True)
    quick = st.columns(4)
    with quick[0]:
        stat_card(
            "Provider",
            PROVIDER_LABELS[provider],
            (
                "No model call"
                if provider == "local"
                else "Configured"
                if is_configured(provider)
                else "Not configured"
            ),
            _provider_logo(provider),
            [1, 1, 1, 1],
            footer=provider_status(provider) if is_configured(provider) else "Set the selected provider API key.",
        )
    with quick[1]:
        stat_card(
            "Context",
            "Last run" if st.session_state.chat_last_context_source == "last_run" else "Manual JSON",
            "Source",
            "📥",
            [1, 1, 1, 1],
            footer="Switch in the context section",
        )
    with quick[2]:
        stat_card(
            "Messages",
            str(len(st.session_state.chat_history)),
            "This session",
            "💬",
            [1, 1, 1, 1],
            footer="Cleared by reset",
        )
    with quick[3]:
        last_message = (
            st.session_state.chat_history[-1]
            if st.session_state.chat_history
            else None
        )
        timestamp = last_message.get("ts") if last_message else None
        freshness = f"{max(0, int(time.time() - timestamp))}s ago" if timestamp else "—"
        stat_card(
            "Last message",
            freshness,
            "Freshness",
            "⏱",
            [1, 1, 1, 1],
            footer="Session-local",
        )

    context = _context_section()
    # Changing JSON starts a new thread so old reports cannot contaminate it.
    import json
    fingerprint = json.dumps(context, sort_keys=True, ensure_ascii=False)
    if st.session_state.get("chat_context_fingerprint") != fingerprint:
        st.session_state.chat_history = []
        st.session_state.chat_last_prompt_system = ""
        st.session_state.chat_last_prompt_user = ""
        st.session_state.chat_context_fingerprint = fingerprint
    with st.expander("Response preferences", expanded=False):
        form = render_input_form()
    _send_button_row(context, form, provider)
    render_chat_view(st.session_state.chat_history)
    followup = st.chat_input("Ask a question about this JSON…",
                             disabled=context is None or not is_configured(provider),
                             key="chat_json_followup")
    if followup:
        _send_reply(context, form, provider, followup)
    render_prompt_preview(
        st.session_state.chat_last_prompt_system,
        st.session_state.chat_last_prompt_user,
    )

    with st.expander("📚 Documentation & warnings"):
        st.markdown(
            "**About this page**\n\n"
            "Chat AI interprets existing model output; it does not train a model "
            "or fetch market data. You can use the most recent trained Random "
            "Forest run or paste a structured ChatRequest or analysis export.\n\n"
            "The full JSON is sent with each question. Changing context starts a new "
            "conversation. Replies explain supplied results without rerunning models. "
            "Inputs over 100,000 characters are rejected without truncation; provider "
            "context limits may be lower."
        )
        st.warning(
            "AI responses are informational only, may be incomplete, and are not investment advice."
        )


__all__ = ["render_chat_ai_page"]
