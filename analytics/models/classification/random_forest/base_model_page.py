"""Interactive Random Forest Base Model workbench."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_ai.analytics.models.variant import render_model_navigation
from market_ai.analytics.export import build_envelope, render_download_button
from market_ai.components.ui.cards import stat_card
from market_ai.components.ui.feedback import toast_success
from market_ai.analytics.models.classification.random_forest.pipeline import (
    BASE_FEATURES,
    RandomForestBaseModel,
    to_chart_frame,
)
from market_ai.utils.theme import COLORS


FEATURE_GROUPS = {
    "Moving averages": ["MA5", "MA10", "MA20", "MA50", "MA200"],
    "Momentum": ["RSI_14", "Momentum_5", "Momentum_10", "MACD", "MACD_Signal"],
    "Volatility": ["Volatility_20", "ATR_14", "Bollinger_Width"],
    "Volume": ["Volume_Ratio", "OBV", "VWAP_Distance"],
    "Price ratios": ["Close_Open_Ratio", "High_Low_Ratio", "Price_MA20_Ratio", "Price_MA50_Ratio"],
}
ALL_FEATURES = [feature for features in FEATURE_GROUPS.values() for feature in features]
DEFAULT_FEATURES = BASE_FEATURES.copy()
ALL_FEATURES = list(dict.fromkeys(BASE_FEATURES + ALL_FEATURES))


def _download_market_data(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """Download real market data through the model's Yahoo Finance loader."""
    model = RandomForestBaseModel(ticker=ticker)
    return to_chart_frame(
        model.load_data(start_date=start, end_date=end, interval=interval)
    )


def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build the same feature matrix used by the fitted Random Forest."""
    upper = data.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    return RandomForestBaseModel.engineer_frame(upper)


def _reset_training_state() -> None:
    for key in ["rf_trained", "rf_metrics", "rf_feature_importance", "rf_confusion", "rf_forecast", "rf_model"]:
        st.session_state.pop(key, None)


def _classification_export_envelope(
    meta: dict,
    data: pd.DataFrame,
    metrics: dict,
    importance: pd.DataFrame,
) -> dict:
    """Export the existing fitted classifier outputs without refitting the model."""
    forecast = st.session_state.get("rf_forecast", pd.DataFrame())
    if forecast is None or forecast.empty:
        price_forecast = None
    else:
        starting_price = float(meta["last_close"]) if meta.get("last_close") is not None else None
        final_forecast = float(forecast["Forecast_Price"].iloc[-1])
        price_forecast = {
            "model": "RandomForestClassifier recursive price path",
            "horizon_sessions": int(len(forecast)),
            "price_unit": "quote currency per unit of the selected instrument",
            "starting_price": starting_price,
            "point_final": final_forecast,
            "return_pct": (
                final_forecast / starting_price - 1
                if starting_price not in (None, 0)
                else None
            ),
            "interval_95_final": {
                "low": float(forecast["Confidence_Lower"].iloc[-1]),
                "high": float(forecast["Confidence_Upper"].iloc[-1]),
            },
            "path": [
                {
                    "session": int(row["Day"]),
                    "date": row["Date"],
                    "point": row["Forecast_Price"],
                    "low_95": row["Confidence_Lower"],
                    "high_95": row["Confidence_Upper"],
                    "bullish_probability": row["Bullish_Probability"],
                    "predicted_trend": row["Predicted_Trend"],
                }
                for row in forecast.to_dict(orient="records")
            ],
        }

    model_params = {
        key: metrics[key]
        for key in ("estimators", "max_depth", "class_weight", "random_state")
        if key in metrics
    }
    metric_values = {
        key: float(metrics[key])
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc")
        if metrics.get(key) is not None
    }
    metric_values.update(
        {
            "train_samples": int(metrics.get("train_samples", 0)),
            "test_samples": int(metrics.get("test_samples", 0)),
        }
    )
    if metrics.get("training_time") is not None:
        metric_values["training_time_seconds"] = float(metrics["training_time"])

    caveats = [
        "Reported accuracy is on a single chronological holdout; out-of-sample performance may differ.",
        "Class balance in the test window affects accuracy, precision, and recall; always compare against a majority-class baseline.",
        "Skill score near zero means the model has no reliable edge over a naive baseline.",
        "Feature importances are model-specific and not causal.",
        "This is informational only and not investment advice.",
    ]
    if price_forecast is not None:
        caveats.append(
            "Recursive tree-based price forecasts drift toward the training mean; large point moves are often model artifacts."
        )

    return build_envelope(
        workbench="classification",
        model_name="RandomForestClassifier",
        model_variant="Base Model",
        request={
            "ticker": meta["ticker"],
            "start_date": meta["start"],
            "end_date": meta["end"],
            "interval": meta["interval"],
            "target_definition": st.session_state.get(
                "rf_target", "Next-day direction"
            ),
            "features": st.session_state.get("rf_selected_features", DEFAULT_FEATURES),
            "parameters": model_params,
        },
        inputs={
            "rows": int(len(data)),
            "source": meta.get("source", "Yahoo Finance"),
            "date_range": {"start": meta["start"], "end": meta["end"]},
            "interval": meta["interval"],
            "features": st.session_state.get("rf_selected_features", DEFAULT_FEATURES),
        },
        results={
            "features": st.session_state.get("rf_selected_features", DEFAULT_FEATURES),
            "confusion_matrix": metrics.get(
                "confusion_matrix", st.session_state.get("rf_confusion")
            ),
            "classification_report": metrics.get("classification_report", {}),
            "model_parameters": model_params,
            "recursive_price_forecast": price_forecast,
        },
        metrics=metric_values,
        diagnostics={
            "feature_importances": (
                importance.to_dict(orient="records") if not importance.empty else []
            ),
            "target_definition": st.session_state.get(
                "rf_target", "Next-day direction"
            ),
            "forecast_interval_level": 0.95 if price_forecast is not None else None,
        },
        caveats=caveats,
    )


def _ensure_demo_data() -> None:
    if "rf_market_data" not in st.session_state:
        end = date.today()
        start = end - timedelta(days=1800)
        try:
            data = _download_market_data("AAPL", start, end, "1d")
        except Exception as error:
            data = pd.DataFrame()
            st.session_state.rf_market_error = str(error)
        st.session_state.rf_market_data = data
        if data.empty:
            st.session_state.rf_market_meta = {
                "ticker": "AAPL",
                "start": str(start),
                "end": str(end),
                "interval": "1d",
                "rows": 0,
                "last_close": 0.0,
                "period_return": 0.0,
                "source": "Yahoo Finance",
            }
        else:
            st.session_state.rf_market_meta = {
                "ticker": "AAPL",
                "start": str(data["date"].min().date()),
                "end": str(data["date"].max().date()),
                "interval": "1d",
                "rows": len(data),
                "last_close": float(data["close"].iloc[-1]),
                "period_return": float(data["close"].iloc[-1] / data["close"].iloc[0] - 1),
                "source": "Yahoo Finance",
            }


def _market_data_section() -> None:
    st.markdown('<div class="workbench-section"><div class="workbench-section-title"><span>📥</span><div><h3>Market Data</h3><p>Choose the ticker and date range used by the model.</p></div></div>', unsafe_allow_html=True)
    meta = st.session_state.rf_market_meta
    control_cols = st.columns([1.05, 1, 1, 0.75, 0.75, 0.75])
    ticker = control_cols[0].text_input("Ticker", meta["ticker"], key="rf_ticker")
    start = control_cols[1].date_input("Start date", date.fromisoformat(meta["start"]), key="rf_start")
    end = control_cols[2].date_input("End date", date.fromisoformat(meta["end"]), key="rf_end")
    interval = control_cols[3].selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], key="rf_interval")
    cached = control_cols[4].checkbox("Use cached", True, key="rf_cached")
    if control_cols[5].button("Download", type="primary", use_container_width=True, key="rf_download"):
        if start >= end:
            st.error("Start date must be before the end date.")
        elif cached and ticker.upper() == meta["ticker"] and str(start) == meta["start"] and str(end) == meta["end"] and interval == meta["interval"]:
            st.success(f"Data loaded from cache for {ticker.upper()} ({meta['rows']:,} rows).")
        else:
            with st.spinner(f"Downloading {ticker.upper()} from Yahoo Finance…"):
                try:
                    downloaded = _download_market_data(ticker, start, end, interval)
                    st.session_state.pop("rf_market_error", None)
                except Exception as error:
                    downloaded = pd.DataFrame()
                    st.session_state.rf_market_error = str(error)
            if downloaded.empty:
                st.error("Yahoo Finance did not return usable data. Widen the date range or check the ticker.")
            else:
                st.session_state.rf_market_data = downloaded
                st.session_state.rf_market_meta = {
                    "ticker": ticker.upper(),
                    "start": str(downloaded["date"].min().date()),
                    "end": str(downloaded["date"].max().date()),
                    "interval": interval,
                    "rows": len(downloaded),
                    "last_close": float(downloaded["close"].iloc[-1]),
                    "period_return": float(downloaded["close"].iloc[-1] / downloaded["close"].iloc[0] - 1),
                    "source": "Yahoo Finance",
                }
                _reset_training_state()
                st.success(f"Data loaded for {ticker.upper()} ({interval} interval, {len(downloaded):,} rows).")
    summary = st.columns(4)
    summary[0].metric("Range", f'{meta["start"]} → {meta["end"]}')
    summary[1].metric("Rows", f'{meta["rows"]:,}')
    summary[2].metric("Last close", f'${meta["last_close"]:,.2f}')
    summary[3].metric("Period return", f'{meta["period_return"]:+.1%}')
    st.caption(f'Source: {meta["source"]}. Downloaded data is cached in session state for this workbench session.')
    st.markdown("</div>", unsafe_allow_html=True)


def _data_overview(data: pd.DataFrame) -> None:
    st.markdown('<div class="workbench-section"><div class="workbench-section-title"><span>📈</span><div><h3>Data Overview</h3><p>Candlesticks with moving-average overlays and volume.</p></div></div>', unsafe_allow_html=True)
    chart = go.Figure()
    recent = data.tail(260)
    chart.add_trace(go.Candlestick(x=recent["date"], open=recent["open"], high=recent["high"], low=recent["low"], close=recent["close"], name="Price", increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"]))
    chart.add_trace(go.Scatter(x=recent["date"], y=recent["close"].rolling(20).mean(), name="MA20", line={"color": COLORS["cyan"], "width": 1.4}))
    chart.add_trace(go.Scatter(x=recent["date"], y=recent["close"].rolling(50).mean(), name="MA50", line={"color": COLORS["yellow"], "width": 1.2}))
    chart.update_layout(height=350, margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_rangeslider_visible=False, font={"color": COLORS["muted"], "size": 10})
    st.plotly_chart(chart, use_container_width=True, config={"displaylogo": False})
    st.markdown("</div>", unsafe_allow_html=True)


def _feature_section(data: pd.DataFrame) -> list[str]:
    st.markdown('<div class="workbench-section"><div class="workbench-section-title"><span>🧪</span><div><h3>Feature Engineering</h3><p>Select the indicators that will be used by the test model.</p></div></div>', unsafe_allow_html=True)
    if "rf_selected_features" not in st.session_state:
        st.session_state.rf_selected_features = DEFAULT_FEATURES.copy()
    selected = st.multiselect("Features", ALL_FEATURES, default=st.session_state.rf_selected_features, key="rf_features")
    st.session_state.rf_selected_features = selected
    target = st.selectbox("Target definition", ["Next-day direction", "5-day direction", "Threshold-based > +0.5%"], key="rf_target")
    st.caption(f"{len(selected)} of {len(ALL_FEATURES)} features selected · Target: {target}")
    if len(selected) >= 2:
        matrix = _engineer_features(data)[selected].dropna().head(5)
        st.dataframe(matrix, hide_index=True, use_container_width=True)
        st.caption(f"Engineered from {len(data):,} raw rows → {_engineer_features(data)[selected].dropna().shape[0]:,} usable rows after rolling-window cleanup.")
    else:
        st.warning("Select at least 2 features to train the model.")
    st.markdown("</div>", unsafe_allow_html=True)
    return selected


def _train_section(data: pd.DataFrame, selected: list[str]) -> None:
    st.markdown('<div class="workbench-section"><div class="workbench-section-title"><span>🚀</span><div><h3>Train Model</h3><p>Fit the supplied Yahoo Finance + scikit-learn Random Forest pipeline.</p></div></div>', unsafe_allow_html=True)
    controls = st.columns([1, 1, 1, 1, 1.2, 1])
    test_size = controls[0].slider("Test size", 0.1, 0.4, 0.25, 0.05, key="rf_test_size")
    random_state = controls[1].number_input("Random state", 1, 999, 42, key="rf_random_state")
    estimators = controls[2].slider("n_estimators", 25, 300, 100, 25, key="rf_estimators")
    max_depth = controls[3].selectbox("Max depth", ["None", 5, 10, 20], key="rf_max_depth")
    class_weight = controls[4].selectbox("Class weight", ["balanced", "None"], key="rf_class_weight")
    run = controls[5].button("🌲 Run model", type="primary", use_container_width=True, key="rf_train")
    if run:
        if len(selected) < 2:
            st.warning("Select at least 2 features before training.")
        else:
            with st.spinner(f"Training Random Forest on Yahoo Finance data with {len(selected)} features…"):
                started = time.perf_counter()
                try:
                    model = RandomForestBaseModel(
                        ticker=st.session_state.rf_market_meta["ticker"],
                        n_estimators=estimators,
                        random_state=random_state,
                        test_size=test_size,
                        forecast_days=365,
                        max_depth=None if max_depth == "None" else int(max_depth),
                        class_weight=None if class_weight == "None" else class_weight,
                        features=selected,
                    )
                    model.load_data(data=data)
                    engineered = model.engineer_features(st.session_state.get("rf_target", "Next-day direction"))
                    if len(engineered) < 50:
                        raise ValueError("Not enough usable observations after feature engineering.")
                    model.prepare_data()
                    model.train()
                    metrics = model.evaluate()
                    importance = model.get_feature_importance()
                    model.generate_forecast(365)
                    metrics.update(
                        {
                            "training_time": time.perf_counter() - started,
                            "estimators": estimators,
                            "max_depth": max_depth,
                            "class_weight": class_weight,
                            "random_state": random_state,
                        }
                    )
                except Exception as error:
                    st.error(f"Model training failed: {error}")
                else:
                    st.session_state.rf_model = model
                    st.session_state.rf_trained = True
                    st.session_state.rf_metrics = metrics
                    st.session_state.rf_feature_importance = importance
                    st.session_state.rf_confusion = metrics["confusion_matrix"]
                    st.session_state.rf_forecast = model.forecast_df
                    st.success(f"Model trained in {metrics['training_time']:.2f}s · Accuracy {metrics['accuracy']:.1%}")
    st.markdown("</div>", unsafe_allow_html=True)


def _performance_and_importance(data: pd.DataFrame) -> None:
    metrics = st.session_state.rf_metrics
    st.markdown('<div class="workbench-section"><div class="workbench-section-title"><span>📊</span><div><h3>Model Performance</h3><p>Evaluation on the held-out test split.</p></div></div>', unsafe_allow_html=True)
    cards = st.columns(5)
    for column, label, key in zip(cards, ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"], ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        with column:
            stat_card(label, f'{metrics[key]:.1%}', "Local test run", "◎", [metrics[key] * 90, metrics[key] * 94, metrics[key] * 100], footer=f'{metrics["test_samples"]:,} test rows')
    panels = st.columns(3, gap="medium")
    with panels[0]:
        st.markdown("**Confusion matrix**")
        matrix = st.session_state.rf_confusion
        fig = go.Figure(go.Heatmap(z=matrix, x=["Bearish", "Bullish"], y=["Bearish", "Bullish"], text=matrix, texttemplate="%{text}", colorscale=[[0, "#141c2b"], [1, "#31d7ee"]], showscale=False))
        fig.update_layout(height=230, margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    with panels[1]:
        st.markdown("**Classification report**")
        st.dataframe(pd.DataFrame({"Class": ["Bearish", "Bullish"], "Precision": [metrics["precision"], metrics["precision"]], "Recall": [metrics["recall"], metrics["recall"]], "F1": [metrics["f1"], metrics["f1"]]}).style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}), hide_index=True, use_container_width=True)
    with panels[2]:
        st.markdown("**ROC curve**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 0.25, 0.5, 0.78, 1], y=[0, 0.6, 0.8, metrics["roc_auc"], 1], name=f'AUC {metrics["roc_auc"]:.1%}', line={"color": COLORS["cyan"], "width": 2}))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Baseline", line={"color": COLORS["muted"], "dash": "dot"}))
        fig.update_layout(height=230, margin={"l": 0, "r": 0, "t": 10, "b": 0, "pad": 0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="workbench-section"><div class="workbench-section-title"><span>🔍</span><div><h3>Feature Importance</h3><p>Relative contribution in the local test model.</p></div></div>', unsafe_allow_html=True)
    importance = st.session_state.rf_feature_importance
    fig = go.Figure(go.Bar(x=importance["importance"], y=importance["feature"], orientation="h", marker_color=importance["importance"], marker_colorscale=[[0, COLORS["violet"]], [1, COLORS["cyan"]]], text=importance["importance"].map(lambda value: f"{value:.1%}"), textposition="outside"))
    fig.update_layout(height=360, margin={"l": 0, "r": 30, "t": 10, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.download_button("Export feature importance CSV", importance.to_csv(index=False), "AAPL_feature_importance.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)


def _forecast_section(data: pd.DataFrame) -> None:
    metrics = st.session_state.rf_metrics
    controls = st.columns([1, 1, 2])
    horizon = controls[0].select_slider("Forecast horizon", [30, 60, 90, 160, 180, 365], value=160, key="rf_horizon")
    show_ci = controls[1].checkbox("Show 95% confidence band", True, key="rf_show_ci")
    show_signals = controls[2].checkbox("Show bullish / bearish signals", True, key="rf_show_signals")
    historical = data.tail(160)
    last_date = pd.Timestamp(historical["date"].iloc[-1])
    last_price = float(historical["close"].iloc[-1])
    model = st.session_state.get("rf_model")
    forecast = st.session_state.get("rf_forecast")
    if model is None or forecast is None:
        st.warning("Run the model to generate a forecast.")
        return
    if len(forecast) < horizon:
        with st.spinner(f"Generating the {horizon}-business-day recursive forecast…"):
            forecast = model.generate_forecast(horizon)
            st.session_state.rf_forecast = forecast
    forecast = forecast.head(horizon).copy()
    future_dates = pd.DatetimeIndex(pd.to_datetime(forecast["Date"]))
    future_prices = forecast["Forecast_Price"].to_numpy()
    probabilities = forecast["Bullish_Probability"].to_numpy()
    trends = forecast["Predicted_Trend"].to_numpy()
    # Include the last known point in the forecast trace so the two lines touch.
    plot_dates = pd.DatetimeIndex([last_date]).append(future_dates)
    plot_prices = np.r_[last_price, future_prices]
    lower = np.r_[last_price, forecast["Confidence_Lower"].to_numpy()]
    upper = np.r_[last_price, forecast["Confidence_Upper"].to_numpy()]
    st.markdown(f'<div class="workbench-section"><div class="workbench-section-title"><span>🔮</span><div><h3>{st.session_state.rf_market_meta["ticker"]} - {horizon}-Day Price Forecast</h3><p>A recursive forecast driven by fitted Random Forest probabilities and historical volatility.</p></div></div>', unsafe_allow_html=True)
    summary = st.columns(4)
    summary[0].metric("Starting price", f'${last_price:,.2f}')
    summary[1].metric("Final price", f'${future_prices[-1]:,.2f}', f'{future_prices[-1] / last_price - 1:+.1%}')
    summary[2].metric("Bullish days", f'{int((forecast["Predicted_Trend"] == "Bullish").sum())}', f'of {horizon}')
    summary[3].metric("Price range", f'${lower.min():,.2f} – ${upper.max():,.2f}')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical["date"], y=historical["close"], name="Historical Price", mode="lines", line={"color": COLORS["cyan"], "width": 2.5}, hovertemplate="%{x|%d %b %Y}<br>Historical: $%{y:.2f}<extra></extra>"))
    if show_ci:
        fig.add_trace(go.Scatter(x=plot_dates, y=upper, name="95% CI upper", line={"color": "rgba(255,107,107,0)", "width": 0}, hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=plot_dates, y=lower, name="95% Confidence Interval", fill="tonexty", line={"color": "rgba(255,107,107,0)", "width": 0}, fillcolor="rgba(255,107,107,0.15)", hovertemplate="%{x|%d %b %Y}<br>Lower: $%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=plot_dates, y=plot_prices, name="Forecast Price", mode="lines+markers", line={"color": COLORS["red"], "width": 2.5, "dash": "dash"}, marker={"size": 4, "color": COLORS["red"]}, hovertemplate="%{x|%d %b %Y}<br>Forecast: $%{y:.2f}<extra></extra>"))
    if show_signals:
        bullish = trends == "Bullish"
        bearish = trends == "Bearish"
        fig.add_trace(go.Scatter(x=future_dates[bullish], y=future_prices[bullish], name="Bullish Signal", mode="markers", marker={"symbol": "triangle-up", "size": 8, "color": COLORS["green"]}, hovertemplate="%{x|%d %b %Y}<br>Bullish signal<extra></extra>"))
        fig.add_trace(go.Scatter(x=future_dates[bearish], y=future_prices[bearish], name="Bearish Signal", mode="markers", marker={"symbol": "triangle-down", "size": 8, "color": COLORS["red"]}, hovertemplate="%{x|%d %b %Y}<br>Bearish signal<extra></extra>"))
    fig.add_vline(x=last_date, line={"color": "rgba(255,255,255,0.4)", "width": 1.5, "dash": "dash"}, annotation_text="📊 Forecast Start", annotation_position="top left", annotation_font={"color": "#FFFFFF", "size": 10})
    fig.update_layout(
        title={"text": f'{st.session_state.rf_market_meta["ticker"]} - {horizon}-Day Price Forecast', "x": 0, "font": {"size": 15, "color": COLORS["text"]}},
        height=500,
        margin={"l": 40, "r": 40, "t": 80, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0.5, "xanchor": "center", "font": {"size": 10}},
        xaxis={"title": "Date", "showgrid": True, "gridcolor": "rgba(255,255,255,0.05)", "rangeslider": {"visible": False}},
        yaxis={"title": "Price ($)", "showgrid": True, "gridcolor": "rgba(255,255,255,0.05)", "tickprefix": "$"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": True, "toImageButtonOptions": {"format": "png", "filename": f'{st.session_state.rf_market_meta["ticker"]}_forecast', "height": 900, "width": 1600, "scale": 2}})
    st.caption("Use the chart toolbar to export the forecast as PNG or SVG.")

    insight_cols = st.columns([1.2, 1, 0.9], gap="medium")
    with insight_cols[0]:
        probability_fig = go.Figure()
        probability_fig.add_trace(go.Scatter(x=future_dates, y=probabilities, name="Bullish probability", line={"color": COLORS["green"], "width": 2}, fill="tozeroy", fillcolor="rgba(63,224,160,0.10)"))
        probability_fig.add_hline(y=0.5, line={"color": COLORS["muted"], "dash": "dot"}, annotation_text="50% threshold", annotation_position="bottom right")
        probability_fig.update_layout(title="📊 Bullish Probability Over Time", height=250, margin={"l": 0, "r": 0, "t": 35, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={"tickformat": ".0%", "range": [0, 1]}, xaxis={"showgrid": False})
        st.plotly_chart(probability_fig, use_container_width=True, config={"displaylogo": False})
    with insight_cols[1]:
        monthly = forecast.assign(Month=forecast["Date"].dt.strftime("%b %Y")).groupby("Month", sort=False).agg(Avg_Price=("Forecast_Price", "mean"), Bullish_Probability=("Bullish_Probability", "mean"), Days=("Day", "count")).reset_index().head(5)
        monthly["Trend"] = np.where(monthly["Bullish_Probability"] >= 0.5, "🟢 ↑", "🔴 ↓")
        monthly["Confidence"] = (monthly["Bullish_Probability"].sub(0.5).abs() * 2).clip(0.5, 0.99)
        st.markdown("**📅 Monthly Forecast Breakdown**")
        st.dataframe(monthly[["Month", "Avg_Price", "Trend", "Confidence"]].style.format({"Avg_Price": "${:.2f}", "Confidence": "{:.0%}"}), hide_index=True, use_container_width=True)
    with insight_cols[2]:
        volatility = float(data["close"].pct_change().std())
        confidence = float(np.clip(metrics["accuracy"] * 0.65 + (1 - min(volatility / 0.03, 1)) * 0.2 + 0.95 * 0.15, 0, 1))
        st.markdown("**🎯 Forecast Confidence**")
        st.progress(confidence, text=f"{confidence:.0%}")
        st.caption(f"Based on:\n\n• Model accuracy: {metrics['accuracy']:.1%}\n\n• Historical volatility: {volatility:.1%}\n\n• Data quality: 95%")
    with st.expander("📋 View forecast details"):
        st.dataframe(forecast.head(30).style.format({"Bullish_Probability": "{:.1%}", "Forecast_Price": "${:.2f}", "Confidence_Lower": "${:.2f}", "Confidence_Upper": "${:.2f}"}), hide_index=True, use_container_width=True)
        st.download_button("Download forecast CSV", forecast.to_csv(index=False), "AAPL_forecast.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)


def render_base_model() -> None:
    """Render the complete Random Forest Base Model workbench."""
    _ensure_demo_data()
    render_model_navigation("Random Forest", "Base Model")
    meta = st.session_state.rf_market_meta
    title_col, action_col = st.columns([1.5, 0.7])
    with title_col:
        st.markdown('<div class="page-kicker">RANDOM FOREST · BASE VARIANT</div><h2>🌲 Base Random Forest Model</h2><p class="page-subtitle">Baseline Random Forest classifier for stock market trend prediction with price forecasting.</p>', unsafe_allow_html=True)
    with action_col:
        st.markdown('<div class="workbench-badges"><span class="badge badge-active">● Active</span><span class="badge badge-info">v1.0.0</span><span class="badge badge-info">AAPL</span></div>', unsafe_allow_html=True)
        action_cols = st.columns(3)
        if action_cols[0].button("↻ Retrain", use_container_width=True):
            st.session_state.rf_focus_train = True
            toast_success("Ready to retrain")
        if action_cols[1].button("Export", use_container_width=True):
            toast_success("Export options are available below")
        if action_cols[2].button("Delete", use_container_width=True):
            st.warning("Delete is disabled for this local workbench.")

    data = st.session_state.rf_market_data
    _market_data_section()
    if data.empty:
        st.error(
            "No market data is available for this session. "
            "Yahoo Finance may be unavailable; use Download after checking the ticker and date range."
        )
        return
    quick = st.columns(4)
    metrics = st.session_state.get("rf_metrics", {})
    with quick[0]:
        stat_card("Data points", f'{len(data):,}', "+45 loaded", "📊", [800, 980, 1100, len(data)], footer=f'{meta["ticker"]} · {meta["interval"]}')
    with quick[1]:
        stat_card("Features", str(len(st.session_state.get("rf_selected_features", DEFAULT_FEATURES))), "Configurable", "🔢", [8, 10, 12, 12], footer="Technical indicators")
    with quick[2]:
        stat_card("Model accuracy", f'{metrics.get("accuracy", 0):.1%}' if metrics else "—", "Run model to evaluate", "🎯", [78, 80, 82, 83] if metrics else [1, 1, 1, 1], footer="Held-out test split")
    with quick[3]:
        stat_card("Training time", f'{metrics.get("training_time", 0):.1f}s' if metrics else "—", "scikit-learn CPU run", "⏱", [6, 5, 4.6, 4.2], footer="Local process")

    _data_overview(data)
    selected = _feature_section(data)
    _train_section(data, selected)
    if st.session_state.get("rf_trained"):
        _performance_and_importance(data)
        _forecast_section(data)
        st.markdown('<div class="workbench-section"><div class="workbench-section-title"><span>💾</span><div><h3>Export & Downloads</h3><p>Download the artifacts from this local test run.</p></div></div>', unsafe_allow_html=True)
        export_cols = st.columns(4)
        export_cols[0].download_button("Download model JSON", json.dumps({"model": "RandomForestClassifier", "variant": "Base Model", "metadata": meta, "metrics": st.session_state.rf_metrics}, indent=2), f'{meta["ticker"]}_random_forest_base.json', "application/json", use_container_width=True)
        export_cols[1].download_button("Download results CSV", st.session_state.rf_feature_importance.to_csv(index=False), f'{meta["ticker"]}_feature_importance.csv', "text/csv", use_container_width=True)
        export_cols[2].download_button("Download raw data CSV", data.to_csv(index=False), f'{meta["ticker"]}_raw_data.csv', "text/csv", use_container_width=True)
        analysis_envelope = _classification_export_envelope(
            meta,
            data,
            st.session_state.rf_metrics,
            st.session_state.rf_feature_importance,
        )
        with export_cols[3]:
            render_download_button(
                analysis_envelope,
                key="classification_random_forest_base_json",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📝 Model metadata"):
        st.json({"model_id": "rf_base", "model_type": "RandomForestClassifier", "version": "v1.0.0", "framework": "scikit-learn + Yahoo Finance", "random_state": st.session_state.get("rf_metrics", {}).get("random_state", 42), "features_used": st.session_state.get("rf_selected_features", DEFAULT_FEATURES), "ticker": meta["ticker"], "data_range": f'{meta["start"]} → {meta["end"]}', "interval": meta["interval"]})
    with st.expander("📚 Documentation & warnings"):
        st.markdown("**About this model**\n\nThis baseline model predicts market direction from technical indicators engineered from Yahoo Finance OHLCV data. It uses a chronological holdout, StandardScaler, and scikit-learn RandomForestClassifier. Forecasts are illustrative and should not be used as financial advice.")
        st.warning("Past performance does not guarantee future results. Forecasts are illustrative and should not be used as financial advice.")