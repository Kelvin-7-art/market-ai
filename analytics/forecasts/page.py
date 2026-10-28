"""Interactive time-series forecasting workbench."""

from __future__ import annotations

import html
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_ai.analytics.forecasts.engine import (
    FORECAST_MODELS,
    INTERVALS,
    load_price_series,
    normalize_ticker,
    run_time_series_forecast,
)
from market_ai.analytics.export import build_envelope, render_download_button
from market_ai.analytics.market_catalog import catalog_name, market_groups, tickers_for
from market_ai.analytics.models.variant import render_model_navigation
from market_ai.utils.theme import COLORS


def _ensure_state() -> None:
    """Initialize page state while keeping it isolated from other workbenches."""
    today = date.today()
    defaults: dict[str, Any] = {
        "forecast_model": "ARIMA",
        "forecast_market": "US equities",
        "forecast_ticker": "AAPL",
        "forecast_custom_override": False,
        "forecast_custom_ticker": "",
        "forecast_date_range": (today - timedelta(days=730), today),
        "forecast_interval": "Daily",
        "forecast_horizon": 30,
        "forecast_ar_lags": 2,
        "forecast_ma_order": 2,
        "forecast_arma_p": 2,
        "forecast_arma_q": 2,
        "forecast_arima_p": 2,
        "forecast_arima_d": 1,
        "forecast_arima_q": 2,
        "forecast_sarima_p": 1,
        "forecast_sarima_d": 1,
        "forecast_sarima_q": 1,
        "forecast_seasonal_p": 1,
        "forecast_seasonal_d": 1,
        "forecast_seasonal_q": 1,
        "forecast_seasonal_period": 5,
        "forecast_seasonal_context": None,
        "forecast_history": pd.Series(dtype=float),
        "forecast_result": None,
        "forecast_snapshot": None,
        "forecast_error": "",
        "forecast_export": None,
        "forecast_data_cache": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.forecast_model not in FORECAST_MODELS:
        st.session_state.forecast_model = "ARIMA"


def _market_controls() -> tuple[str, str, date | None, date | None, str, int]:
    """Render market, ticker, date range, frequency, and forecast length controls."""
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title"><span>🌐</span><div><h3>Market & Date Range</h3><p>Choose any catalog symbol or enter a custom stock, crypto, or forex ticker.</p></div></div>',
        unsafe_allow_html=True,
    )
    controls = st.columns([1.0, 1.2, 1.1, 2.0, 0.85, 0.85])
    group = controls[0].selectbox("Market", market_groups(), key="forecast_market")
    entries = tickers_for(group)
    symbols = [entry["ticker"] for entry in entries] or ["AAPL"]
    if st.session_state.forecast_ticker not in symbols:
        st.session_state.forecast_ticker = symbols[0]
    selected = controls[1].selectbox(
        "Ticker",
        symbols,
        format_func=lambda symbol: f"{symbol} · {catalog_name(symbol)}",
        key="forecast_ticker",
    )
    custom_override = controls[2].checkbox(
        "Custom ticker override",
        key="forecast_custom_override",
    )
    date_range = controls[3].date_input(
        "History date range (inclusive)",
        min_value=date(1990, 1, 1),
        max_value=date.today(),
        key="forecast_date_range",
    )
    interval_label = controls[4].selectbox(
        "Sampling",
        list(INTERVALS),
        key="forecast_interval",
    )
    horizon_options = [5, 10, 20, 30, 60, 90]
    if st.session_state.forecast_horizon not in horizon_options:
        st.session_state.forecast_horizon = 30
    horizon = controls[5].selectbox(
        "Periods ahead",
        horizon_options,
        key="forecast_horizon",
    )
    ticker = selected
    if custom_override:
        custom = st.text_input(
            "Custom ticker",
            placeholder="e.g. BRK-B, BTC-USD, EURUSD=X",
            key="forecast_custom_ticker",
        )
        ticker = custom.strip().upper()

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = None, None
    st.markdown("</div>", unsafe_allow_html=True)
    return ticker, group, start_date, end_date, INTERVALS[interval_label], int(horizon)


def _model_parameters(model_name: str, interval: str, group: str) -> dict[str, int]:
    """Show compact order controls relevant to the selected model."""
    settings: dict[str, int] = {}
    if model_name == "AR":
        settings["p"] = int(
            st.number_input("Autoregressive lags (p)", 1, 20, key="forecast_ar_lags")
        )
    elif model_name == "MA":
        settings["q"] = int(
            st.number_input("Moving-average order (q)", 1, 10, key="forecast_ma_order")
        )
    elif model_name == "ARMA":
        cols = st.columns(2)
        settings["p"] = int(
            cols[0].number_input("AR order (p)", 0, 5, key="forecast_arma_p")
        )
        settings["q"] = int(
            cols[1].number_input("MA order (q)", 0, 5, key="forecast_arma_q")
        )
        settings["d"] = 0
    elif model_name in {"ARIMA", "SARIMA / SARIMAX"}:
        cols = st.columns(3)
        defaults = ("forecast_arima_p", "forecast_arima_d", "forecast_arima_q")
        if model_name == "SARIMA / SARIMAX":
            defaults = ("forecast_sarima_p", "forecast_sarima_d", "forecast_sarima_q")
        settings["p"] = int(cols[0].number_input("AR order (p)", 0, 5, key=defaults[0]))
        settings["d"] = int(cols[1].number_input("Difference order (d)", 0, 2, key=defaults[1]))
        settings["q"] = int(cols[2].number_input("MA order (q)", 0, 5, key=defaults[2]))
        if model_name == "SARIMA / SARIMAX":
            seasonal_cols = st.columns(4)
            settings["seasonal_p"] = int(
                seasonal_cols[0].number_input("Seasonal AR (P)", 0, 3, key="forecast_seasonal_p")
            )
            settings["seasonal_d"] = int(
                seasonal_cols[1].number_input("Seasonal difference (D)", 0, 2, key="forecast_seasonal_d")
            )
            settings["seasonal_q"] = int(
                seasonal_cols[2].number_input("Seasonal MA (Q)", 0, 3, key="forecast_seasonal_q")
            )
            default_period = 7 if interval == "1d" and group.lower() == "crypto" else 5 if interval == "1d" else 13 if interval == "1wk" else 12
            period_options = [5, 7, 12, 13, 21, 26, 52]
            season_context = (interval, group)
            if st.session_state.forecast_seasonal_context != season_context:
                st.session_state.forecast_seasonal_period = default_period
                st.session_state.forecast_seasonal_context = season_context
            period = seasonal_cols[3].selectbox(
                "Season length",
                period_options,
                key="forecast_seasonal_period",
                help=f"Suggested for this series: {default_period} samples per season.",
            )
            settings["seasonal_period"] = int(period)
    else:
        st.caption(
            "Prophet fits trend and date-based seasonal components; no AR/MA order is required."
        )
    return settings


def _snapshot(
    ticker: str,
    group: str,
    start_date: date,
    end_date: date,
    interval: str,
    horizon: int,
    model_name: str,
    params: dict[str, int],
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "market": group,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
        "interval": interval,
        "horizon": horizon,
        "model": model_name,
        "params": params,
    }


# ============================================================
# STANDARDIZED CHART AND REPORT
# ============================================================
def build_forecast_figure(history: pd.Series, result: dict, ticker: str) -> go.Figure:
    """Distinguish estimation history, holdout observations, and future forecasts."""
    future, holdout = result["forecast"], result["backtest"]
    training = history.loc[history.index < pd.Timestamp(holdout["date"].iloc[0])]
    fig = go.Figure()
    hover = "Date: %{x|%Y-%m-%d}<br>Price: %{y:,.4f}<extra>%{fullData.name}</extra>"
    fig.add_trace(go.Scatter(x=training.index, y=training, name="Training history",
                            line=dict(color="#2459A6", width=2), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=holdout["date"], y=holdout["actual"], name="Holdout actual",
                            line=dict(color="#F59E0B", width=2, dash="dash"), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=holdout["date"], y=holdout["forecast"], name="Holdout prediction",
                            line=dict(color="#A78BFA", width=2, dash="dot"), hovertemplate=hover))
    fig.add_trace(go.Scatter(x=future["date"], y=future["lower"], mode="lines", line=dict(width=0),
                            showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future["date"], y=future["upper"], mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor="rgba(0,255,120,0.18)", name="95% model prediction interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future["date"], y=future["forecast"], name="Future forecast",
                            line=dict(color="#00DA82", width=3), customdata=future[["lower","upper"]],
                            hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:,.4f}<br>Lower 95%: %{customdata[0]:,.4f}<br>Upper 95%: %{customdata[1]:,.4f}<extra>Forecast</extra>"))
    # Avoid Plotly's timestamp annotation arithmetic by adding annotations separately.
    start = pd.Timestamp(future["date"].iloc[0]).isoformat()
    fig.add_vline(x=start, line_dash="dash", line_color="#00DA82", line_width=1)
    fig.add_annotation(x=start, y=1, yref="paper", text="Forecast begins", showarrow=False, yshift=14)
    test_start = pd.Timestamp(holdout["date"].iloc[0]).isoformat()
    fig.add_vline(x=test_start, line_dash="dot", line_color="#F59E0B", line_width=1)
    fig.update_layout(title=f"{html.escape(ticker)} · {html.escape(result['model'])}", height=480,
                      hovermode="x unified", margin=dict(l=20,r=20,t=80,b=20),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      yaxis_title="Adjusted price (instrument quote units)", xaxis_title="Date / period end",
                      legend=dict(orientation="h",y=1.18), xaxis=dict(rangeslider=dict(visible=False)))
    return fig


def _forecast_export_envelope(history: pd.Series, result: dict, snapshot: dict) -> dict:
    request = {**snapshot, "train_start": str(history.index[0]), "train_end": str(history.index[-1]),
               "hyperparameters": dict(result["params"]),
               "evaluation_train_end": str(result["holdout_train_end"])}
    return build_envelope(workbench="forecast", model_name=result["model"], model_variant=None,
        request=request,
        inputs={"observations":len(history), "source":"Yahoo Finance", "loader":history.attrs,
                "train_samples":result["train_observations"], "test_samples":result["test_observations"]},
        results={"forecast_path":result["forecast"], "holdout_path":result["backtest"]},
        metrics={**result["metrics"], "mape_unit":"percent", "error_unit":"instrument quote price units"},
        diagnostics={"model_summary":result["summary"], "fit_warnings":result["fit_warnings"],
                     "evaluation":"Single fixed-origin forecast over the last 20% (subject to minimum sample limits); refit on all history for the future path.",
                     "aic_bic_fit":"Full-history refit", "error_definition":"actual minus predicted",
                     "interval_level":0.95},
        caveats=["Point forecasts are conditional on the fitted model and historical data.",
                 "Intervals generally widen with horizon; their shape depends on the model and they do not guarantee 95% realized coverage.",
                 "Holdout errors come from a separate earlier fit; the future model is refitted on all selected history.",
                 "Daily non-crypto dates are weekdays, not an exchange holiday calendar. Weekly/monthly dates are calendar period-end labels.",
                 "Only completed periods are used; today's daily bar is excluded. Adjusted prices can change after corporate actions.",
                 "MAPE excludes zero actual prices. Expected return is a model projection, not an investment recommendation.",
                 "SARIMA/SARIMAX uses no external regressors. Prophet AIC and BIC are not reported.",
                 "No positivity constraint is imposed; inspect forecasts and warnings before interpreting price levels."])


def render_forecast_results(fig, metrics, forecast_df, holdout_df, export_dict) -> None:
    """Render the identical chart, metrics, tables, and summary for all six models."""
    if forecast_df is None or forecast_df.empty or holdout_df is None or holdout_df.empty:
        st.info("Run a forecast with sufficient price history to see results.")
        return
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False}, key="forecast_result_chart")
    render_download_button(export_dict, key="forecast_analysis_json", label="⬇️ Download analysis JSON")
    def display(value, suffix="", decimals=4):
        return "N/A" if value is None or not np.isfinite(value) else f"{value:,.{decimals}f}{suffix}"
    cols = st.columns(3)
    cols[0].metric("Last actual price", display(metrics.get("last_actual")))
    cols[1].metric("Forecast at horizon end", display(metrics.get("final_forecast")))
    cols[2].metric("Expected return", display(metrics.get("expected_return_pct"), "%", 2))
    cols = st.columns(3)
    for col, name in zip(cols, ["rmse","mae","mape"]):
        col.metric("Holdout " + name.upper(), display(metrics.get(name), "%" if name=="mape" else ""))
    if export_dict["app"]["model_name"] != "Prophet":
        cols = st.columns(2)
        cols[0].metric("AIC (full-history fit)", display(metrics.get("aic"), decimals=2))
        cols[1].metric("BIC (full-history fit)", display(metrics.get("bic"), decimals=2))
    st.caption("Holdout predictions use only the earlier training window. The future forecast uses a refit on all selected history. Error = actual − predicted; MAPE is a percentage.")
    for message in export_dict.get("diagnostics",{}).get("fit_warnings",[]):
        st.warning(message)
    st.subheader("Forecast table")
    st.dataframe(forecast_df.rename(columns={"date":"Date","forecast":"Predicted Price","lower":"Lower Bound 95%","upper":"Upper Bound 95%"}), hide_index=True, use_container_width=True)
    st.download_button("Download forecast CSV", forecast_df.to_csv(index=False), "forecast.csv", "text/csv", key="forecast_csv")
    st.subheader("Holdout evaluation")
    st.dataframe(holdout_df[["date","actual","forecast","error"]].rename(columns={"date":"Date","actual":"Actual Price","forecast":"Predicted Price","error":"Error (Actual − Predicted)"}), hide_index=True, use_container_width=True)
    st.download_button("Download holdout CSV", holdout_df.to_csv(index=False), "holdout.csv", "text/csv", key="forecast_holdout_csv")
    with st.expander("Model summary"):
        st.code(export_dict.get("diagnostics",{}).get("model_summary","Summary unavailable."), language=None)
    with st.expander("Methodology and caveats"):
        for caveat in export_dict.get("caveats",[]):
            st.write(caveat)


# ============================================================
# PAGE ENTRY POINT
# ============================================================
def render_forecasts() -> None:
    """Render the existing Forecasting route with persistent, explicit model runs."""
    _ensure_state()
    render_model_navigation("Forecasting", "")
    model_name = st.session_state.forecast_model
    st.title(f"{model_name} Forecast")
    st.caption("Stocks, global equities, ETFs, crypto and forex · Chronological evaluation and future projections")
    ticker, group, start_date, end_date, interval, horizon = _market_controls()
    st.subheader("Model configuration")
    params = _model_parameters(model_name, interval, group)
    force_refresh = st.checkbox("Refresh Yahoo data on next run", value=False, key="forecast_refresh")
    current = None
    validation_error = ""
    try:
        ticker = normalize_ticker(ticker, group)
        if start_date is None or end_date is None:
            raise ValueError("Select both a start date and an end date.")
        if start_date >= end_date:
            raise ValueError("Start date must be earlier than end date.")
        current = _snapshot(ticker,group,start_date,end_date,interval,horizon,model_name,params)
    except ValueError as error:
        validation_error = str(error)
    run_col, reset_col = st.columns([3,1])
    run = run_col.button(f"Run {model_name} forecast", type="primary", key="forecast_run_model")
    if reset_col.button("Clear forecast", key="forecast_clear"):
        for key in ["forecast_result","forecast_snapshot","forecast_export"]:
            st.session_state[key] = None
        st.session_state.forecast_history = pd.Series(dtype=float)
        st.session_state.forecast_data_cache = {}
        st.session_state.forecast_error = ""
    if run:
        st.session_state.forecast_error = validation_error
        if not validation_error:
            try:
                with st.spinner(f"Loading {ticker} and fitting {model_name}…"):
                    data_key = (ticker,str(start_date),str(end_date),interval)
                    cache = st.session_state.forecast_data_cache
                    if force_refresh or data_key not in cache:
                        loaded = load_price_series(ticker,start_date,end_date,interval)
                        # Bound per-session memory; never cache an unsuccessful load.
                        if len(cache) >= 5:
                            cache.pop(next(iter(cache)))
                        cache[data_key] = loaded
                    history = cache[data_key]
                    result = run_time_series_forecast(history,model_name,horizon,interval,group,params)
                    envelope = _forecast_export_envelope(history,result,current)
                    st.session_state.forecast_history = history
                    st.session_state.forecast_result = result
                    st.session_state.forecast_snapshot = current
                    st.session_state.forecast_export = envelope
            except Exception as error:
                st.session_state.forecast_error = str(error)
    if st.session_state.forecast_error:
        st.error(st.session_state.forecast_error)
    result = st.session_state.forecast_result
    saved = st.session_state.forecast_snapshot
    if result and st.session_state.forecast_export:
        if current != saved:
            st.info("Controls have changed. The report below is the previous successful run; run again to update it.")
        elif st.session_state.forecast_error:
            st.info("Showing the previous successful run. The latest attempt failed.")
        st.caption(f"Saved report: {saved['ticker']} · {saved['market']} · {saved['model']} · {saved['start']} to {saved['end']} · {saved['interval']} · {saved['horizon']} periods ahead")
        history = st.session_state.forecast_history
        fig = build_forecast_figure(history,result,saved["ticker"])
        render_forecast_results(fig,result["metrics"],result["forecast"],result["backtest"],st.session_state.forecast_export)
    else:
        st.info("Choose a market, symbol, history range and model, then run the forecast. No model runs automatically when controls change.")


__all__ = ["render_forecasts", "render_forecast_results", "build_forecast_figure"]
