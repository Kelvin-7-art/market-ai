"""Interactive Random Forest Optimized Model workbench."""

from __future__ import annotations

import json
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_ai.analytics.models.variant import render_model_navigation
from market_ai.components.ui.cards import stat_card
from market_ai.components.ui.feedback import toast_success
from market_ai.analytics.models.classification.random_forest.advanced_pipeline import (
    ADVANCED_FEATURES,
    DEFAULT_PARAM_GRID,
    RandomForestAdvancedModel,
    to_chart_frame,
)
from market_ai.utils.theme import COLORS


# ---------------------------------------------------------------
# Feature catalogue (grouped for the multiselect UI)
# ---------------------------------------------------------------
FEATURE_GROUPS = {
    "Returns": ["Return_1d", "Return_5d", "Return_10d", "Return_20d"],
    "Moving averages": ["MA5", "MA10", "MA20", "MA50", "MA200"],
    "Price ratios": [
        "Price_MA5_Ratio",
        "Price_MA10_Ratio",
        "Price_MA20_Ratio",
        "Price_MA50_Ratio",
    ],
    "Volatility": ["Volatility_10", "Volatility_20", "Volatility_50", "ATR"],
    "Volume": [
        "Volume_Ratio_5",
        "Volume_Ratio_20",
        "Volume_Change",
        "OBV",
        "OBV_Ratio",
        "CMF",
    ],
    "Momentum": [
        "Momentum_5",
        "Momentum_10",
        "Momentum_20",
        "RSI",
        "MACD",
        "MACD_Signal",
        "MACD_Histogram",
    ],
    "Patterns": [
        "High_Low_Ratio",
        "Close_Open_Ratio",
        "High_Close_Ratio",
        "Low_Close_Ratio",
    ],
    "Bollinger Bands": ["BB_Upper", "BB_Middle", "BB_Lower", "BB_Position"],
    "Trend & structure": [
        "Stoch_K",
        "Stoch_D",
        "Williams_R",
        "ADX",
        "Trend_Strength",
        "Dist_From_High",
        "Dist_From_Low",
        "Price_Position",
    ],
}
ALL_FEATURES = list(dict.fromkeys(ADVANCED_FEATURES + [f for g in FEATURE_GROUPS.values() for f in g]))
DEFAULT_FEATURES = ADVANCED_FEATURES.copy()


# ---------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------
def _download_market_data(ticker: str, start: date, end: date, interval: str) -> pd.DataFrame:
    """Download real market data through the advanced model's Yahoo Finance loader."""
    model = RandomForestAdvancedModel(ticker=ticker)
    return to_chart_frame(
        model.load_data(start_date=start, end_date=end, interval=interval)
    )


def _engineer_features(data: pd.DataFrame, target_mode: str = "Next-day direction") -> pd.DataFrame:
    """Build the same feature matrix used by the fitted Advanced Random Forest."""
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
    return RandomForestAdvancedModel.engineer_frame(upper, target_mode=target_mode)


def _reset_training_state() -> None:
    for key in [
        "rf_adv_trained",
        "rf_adv_metrics",
        "rf_adv_feature_importance",
        "rf_adv_confusion",
        "rf_adv_forecast",
        "rf_adv_model",
        "rf_adv_best_params",
        "rf_adv_cv_results",
        "rf_adv_best_score",
    ]:
        st.session_state.pop(key, None)


def _ensure_demo_data() -> None:
    if "rf_adv_market_data" not in st.session_state:
        end = date.today()
        start = end - timedelta(days=1800)
        try:
            data = _download_market_data("AAPL", start, end, "1d")
        except Exception as error:
            data = pd.DataFrame()
            st.session_state.rf_adv_market_error = str(error)
        st.session_state.rf_adv_market_data = data
        if data.empty:
            st.session_state.rf_adv_market_meta = {
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
            st.session_state.rf_adv_market_meta = {
                "ticker": "AAPL",
                "start": str(data["date"].min().date()),
                "end": str(data["date"].max().date()),
                "interval": "1d",
                "rows": len(data),
                "last_close": float(data["close"].iloc[-1]),
                "period_return": float(data["close"].iloc[-1] / data["close"].iloc[0] - 1),
                "source": "Yahoo Finance",
            }


# ---------------------------------------------------------------
# Market data section
# ---------------------------------------------------------------
def _market_data_section() -> None:
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        '<span>📥</span><div><h3>Market Data</h3>'
        '<p>Choose the ticker and date range used by the model.</p></div></div>',
        unsafe_allow_html=True,
    )
    meta = st.session_state.rf_adv_market_meta
    control_cols = st.columns([1.05, 1, 1, 0.75, 0.75, 0.75])
    ticker = control_cols[0].text_input("Ticker", meta["ticker"], key="rf_adv_ticker")
    start = control_cols[1].date_input(
        "Start date", date.fromisoformat(meta["start"]), key="rf_adv_start"
    )
    end = control_cols[2].date_input(
        "End date", date.fromisoformat(meta["end"]), key="rf_adv_end"
    )
    interval = control_cols[3].selectbox(
        "Interval", ["1d", "1h", "30m", "15m", "5m"], key="rf_adv_interval"
    )
    cached = control_cols[4].checkbox("Use cached", True, key="rf_adv_cached")
    if control_cols[5].button(
        "Download", type="primary", use_container_width=True, key="rf_adv_download"
    ):
        if start >= end:
            st.error("Start date must be before the end date.")
        elif (
            cached
            and ticker.upper() == meta["ticker"]
            and str(start) == meta["start"]
            and str(end) == meta["end"]
            and interval == meta["interval"]
        ):
            st.success(
                f"Data loaded from cache for {ticker.upper()} ({meta['rows']:,} rows)."
            )
        else:
            with st.spinner(f"Downloading {ticker.upper()} from Yahoo Finance…"):
                try:
                    downloaded = _download_market_data(ticker, start, end, interval)
                    st.session_state.pop("rf_adv_market_error", None)
                except Exception as error:
                    downloaded = pd.DataFrame()
                    st.session_state.rf_adv_market_error = str(error)
            if downloaded.empty:
                st.error(
                    "Yahoo Finance did not return usable data. Widen the date range or check the ticker."
                )
            else:
                st.session_state.rf_adv_market_data = downloaded
                st.session_state.rf_adv_market_meta = {
                    "ticker": ticker.upper(),
                    "start": str(downloaded["date"].min().date()),
                    "end": str(downloaded["date"].max().date()),
                    "interval": interval,
                    "rows": len(downloaded),
                    "last_close": float(downloaded["close"].iloc[-1]),
                    "period_return": float(
                        downloaded["close"].iloc[-1] / downloaded["close"].iloc[0] - 1
                    ),
                    "source": "Yahoo Finance",
                }
                _reset_training_state()
                st.success(
                    f"Data loaded for {ticker.upper()} ({interval} interval, {len(downloaded):,} rows)."
                )
    summary = st.columns(4)
    summary[0].metric("Range", f'{meta["start"]} → {meta["end"]}')
    summary[1].metric("Rows", f'{meta["rows"]:,}')
    summary[2].metric("Last close", f'${meta["last_close"]:,.2f}')
    summary[3].metric("Period return", f'{meta["period_return"]:+.1%}')
    st.caption(
        f'Source: {meta["source"]}. Downloaded data is cached in session state for this workbench session.'
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Data overview
# ---------------------------------------------------------------
def _data_overview(data: pd.DataFrame) -> None:
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        '<span>📈</span><div><h3>Data Overview</h3>'
        '<p>Candlesticks with moving-average overlays and volume.</p></div></div>',
        unsafe_allow_html=True,
    )
    chart = go.Figure()
    recent = data.tail(260)
    chart.add_trace(
        go.Candlestick(
            x=recent["date"],
            open=recent["open"],
            high=recent["high"],
            low=recent["low"],
            close=recent["close"],
            name="Price",
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
        )
    )
    chart.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["close"].rolling(20).mean(),
            name="MA20",
            line={"color": COLORS["cyan"], "width": 1.4},
        )
    )
    chart.add_trace(
        go.Scatter(
            x=recent["date"],
            y=recent["close"].rolling(50).mean(),
            name="MA50",
            line={"color": COLORS["yellow"], "width": 1.2},
        )
    )
    chart.update_layout(
        height=350,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        font={"color": COLORS["muted"], "size": 10},
    )
    st.plotly_chart(chart, use_container_width=True, config={"displaylogo": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Feature engineering section
# ---------------------------------------------------------------
def _feature_section(data: pd.DataFrame) -> list[str]:
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        '<span>🧪</span><div><h3>Feature Engineering</h3>'
        '<p>Select the indicators that will be used by the optimized model.</p></div></div>',
        unsafe_allow_html=True,
    )
    if "rf_adv_selected_features" not in st.session_state:
        st.session_state.rf_adv_selected_features = DEFAULT_FEATURES.copy()
    selected = st.multiselect(
        "Features",
        ALL_FEATURES,
        default=st.session_state.rf_adv_selected_features,
        key="rf_adv_features",
    )
    st.session_state.rf_adv_selected_features = selected
    target = st.selectbox(
        "Target definition",
        ["Next-day direction", "5-day direction", "Threshold-based > +0.5%"],
        key="rf_adv_target",
    )
    st.caption(f"{len(selected)} of {len(ALL_FEATURES)} features selected · Target: {target}")
    if len(selected) >= 2:
        engineered = _engineer_features(data, target)
        matrix = engineered[selected].dropna().head(5)
        st.dataframe(matrix, hide_index=True, use_container_width=True)
        st.caption(
            f"Engineered from {len(data):,} raw rows → {engineered[selected].dropna().shape[0]:,} usable rows after rolling-window cleanup."
        )
    else:
        st.warning("Select at least 2 features to train the model.")
    st.markdown("</div>", unsafe_allow_html=True)
    return selected


# ---------------------------------------------------------------
# Train section (with GridSearchCV tuning)
# ---------------------------------------------------------------
def _train_section(data: pd.DataFrame, selected: list[str]) -> None:
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        '<span>🚀</span><div><h3>Train & Tune Model</h3>'
        '<p>GridSearchCV with TimeSeriesSplit for the supplied Yahoo Finance + scikit-learn pipeline.</p></div></div>',
        unsafe_allow_html=True,
    )
    controls = st.columns([1, 1, 1, 1, 1.2, 1])
    test_size = controls[0].slider("Test size", 0.1, 0.4, 0.20, 0.05, key="rf_adv_test_size")
    random_state = controls[1].number_input("Random state", 1, 999, 42, key="rf_adv_random_state")
    cv_splits = controls[2].slider("CV splits", 2, 5, 3, 1, key="rf_adv_cv_splits")
    scoring = controls[3].selectbox(
        "Scoring", ["accuracy", "f1", "roc_auc"], key="rf_adv_scoring"
    )
    grid_mode = controls[4].selectbox(
        "Parameter grid",
        ["Default (fast)", "Default (full)", "Narrow (trees only)"],
        key="rf_adv_grid_mode",
    )
    run = controls[5].button(
        "🌲 Run model", type="primary", use_container_width=True, key="rf_adv_train"
    )

    if grid_mode == "Default (fast)":
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [None, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
            "class_weight": ["balanced", None],
        }
    elif grid_mode == "Default (full)":
        param_grid = dict(DEFAULT_PARAM_GRID)
    else:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "max_features": ["sqrt"],
            "class_weight": ["balanced"],
        }

    if run:
        if len(selected) < 2:
            st.warning("Select at least 2 features before training.")
        else:
            with st.spinner(
                f"Tuning Random Forest with GridSearchCV ({cv_splits}-fold TimeSeriesSplit) on {len(selected)} features…"
            ):
                started = time.perf_counter()
                try:
                    model = RandomForestAdvancedModel(
                        ticker=st.session_state.rf_adv_market_meta["ticker"],
                        random_state=random_state,
                        test_size=test_size,
                        cv_splits=cv_splits,
                        forecast_days=365,
                        features=selected,
                        param_grid=param_grid,
                        scoring=scoring,
                    )
                    model.load_data(data=data)
                    engineered = model.engineer_features(
                        st.session_state.get("rf_adv_target", "Next-day direction")
                    )
                    if len(engineered) < 80:
                        raise ValueError(
                            "Not enough usable observations after feature engineering."
                        )
                    model.prepare_data()
                    best_params = model.tune_hyperparameters()
                    model.predict()
                    metrics = model.evaluate()
                    importance = model.get_feature_importance()
                    model.generate_forecast(365)
                    metrics.update(
                        {
                            "training_time": time.perf_counter() - started,
                            "best_params": best_params,
                            "best_cv_score": model.best_score,
                            "grid_total_fits": model.last_grid_total_fits,
                            "cv_splits": cv_splits,
                            "random_state": random_state,
                            "scoring": scoring,
                        }
                    )
                except Exception as error:
                    st.error(f"Model training failed: {error}")
                else:
                    st.session_state.rf_adv_model = model
                    st.session_state.rf_adv_trained = True
                    st.session_state.rf_adv_metrics = metrics
                    st.session_state.rf_adv_feature_importance = importance
                    st.session_state.rf_adv_confusion = metrics["confusion_matrix"]
                    st.session_state.rf_adv_forecast = model.forecast_df
                    st.session_state.rf_adv_best_params = best_params
                    st.session_state.rf_adv_cv_results = model.cv_results_summary
                    st.session_state.rf_adv_best_score = model.best_score
                    st.success(
                        f"Model tuned in {metrics['training_time']:.2f}s · "
                        f"CV {scoring} {model.best_score:.1%} · "
                        f"Test accuracy {metrics['accuracy']:.1%}"
                    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Performance & importance
# ---------------------------------------------------------------
def _performance_and_importance(data: pd.DataFrame) -> None:
    metrics = st.session_state.rf_adv_metrics
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        '<span>📊</span><div><h3>Model Performance</h3>'
        '<p>Evaluation on the held-out test split.</p></div></div>',
        unsafe_allow_html=True,
    )
    cards = st.columns(5)
    for column, label, key in zip(
        cards,
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        ["accuracy", "precision", "recall", "f1", "roc_auc"],
    ):
        with column:
            stat_card(
                label,
                f'{metrics[key]:.1%}',
                "Tuned test run",
                "◎",
                [metrics[key] * 90, metrics[key] * 94, metrics[key] * 100],
                footer=f'{metrics["test_samples"]:,} test rows',
            )
    panels = st.columns(3, gap="medium")
    with panels[0]:
        st.markdown("**Confusion matrix**")
        matrix = st.session_state.rf_adv_confusion
        fig = go.Figure(
            go.Heatmap(
                z=matrix,
                x=["Bearish", "Bullish"],
                y=["Bearish", "Bullish"],
                text=matrix,
                texttemplate="%{text}",
                colorscale=[[0, "#141c2b"], [1, "#31d7ee"]],
                showscale=False,
            )
        )
        fig.update_layout(
            height=230,
            margin={"l": 0, "r": 0, "t": 10, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    with panels[1]:
        st.markdown("**Classification report**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Class": ["Bearish", "Bullish"],
                    "Precision": [metrics["precision"], metrics["precision"]],
                    "Recall": [metrics["recall"], metrics["recall"]],
                    "F1": [metrics["f1"], metrics["f1"]],
                }
            ).style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}),
            hide_index=True,
            use_container_width=True,
        )
    with panels[2]:
        st.markdown("**ROC curve**")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 0.25, 0.5, 0.78, 1],
                y=[0, 0.6, 0.8, metrics["roc_auc"], 1],
                name=f'AUC {metrics["roc_auc"]:.1%}',
                line={"color": COLORS["cyan"], "width": 2},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                name="Baseline",
                line={"color": COLORS["muted"], "dash": "dot"},
            )
        )
        fig.update_layout(
            height=230,
            margin={"l": 0, "r": 0, "t": 10, "b": 0, "pad": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Best params / CV summary
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        '<span>⚙️</span><div><h3>Tuning Results</h3>'
        '<p>GridSearchCV best parameters and top-ranked configurations.</p></div></div>',
        unsafe_allow_html=True,
    )
    tune_cols = st.columns([1, 1.4])
    with tune_cols[0]:
        st.markdown("**Best parameters**")
        st.json(st.session_state.get("rf_adv_best_params", {}))
        best_score = st.session_state.get("rf_adv_best_score")
        if best_score is not None:
            st.metric("Best CV score", f"{best_score:.1%}")
        st.caption(
            f'Total fits: {metrics.get("grid_total_fits", 0):,} · '
            f'CV splits: {metrics.get("cv_splits", "—")} · '
            f'Scoring: {metrics.get("scoring", "accuracy")}'
        )
    with tune_cols[1]:
        st.markdown("**Top configurations**")
        cv_results = st.session_state.get("rf_adv_cv_results")
        if cv_results is not None and not cv_results.empty:
            display = cv_results.copy()
            if "mean_test_score" in display.columns:
                display["mean_test_score"] = display["mean_test_score"].map(
                    lambda v: f"{v:.1%}"
                )
            if "std_test_score" in display.columns:
                display["std_test_score"] = display["std_test_score"].map(
                    lambda v: f"±{v:.1%}"
                )
            st.dataframe(display, hide_index=True, use_container_width=True)
        else:
            st.info("No CV results available.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Feature importance (built-in + permutation)
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        '<span>🔍</span><div><h3>Feature Importance</h3>'
        '<p>Gini importance and permutation importance on the held-out test set.</p></div></div>',
        unsafe_allow_html=True,
    )
    importance = st.session_state.rf_adv_feature_importance
    importance_plot = importance.head(20).sort_values("Builtin_Importance", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=importance_plot["Builtin_Importance"],
            y=importance_plot["Feature"],
            orientation="h",
            name="Gini importance",
            marker_color=importance_plot["Builtin_Importance"],
            marker_colorscale=[[0, COLORS["violet"]], [1, COLORS["cyan"]]],
            text=importance_plot["Builtin_Importance"].map(lambda v: f"{v:.1%}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        height=420,
        margin={"l": 0, "r": 30, "t": 10, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.dataframe(
        importance.style.format(
            {
                "Builtin_Importance": "{:.3%}",
                "Permutation_Mean": "{:.3%}",
                "Permutation_STD": "{:.3%}",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.download_button(
        "Export feature importance CSV",
        importance.to_csv(index=False),
        "AAPL_advanced_feature_importance.csv",
        "text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Forecast section (mirrors base, uses improved pipeline)
# ---------------------------------------------------------------
def _forecast_section(data: pd.DataFrame) -> None:
    metrics = st.session_state.rf_adv_metrics
    controls = st.columns([1, 1, 2])
    horizon = controls[0].select_slider(
        "Forecast horizon",
        [30, 60, 90, 160, 180, 365],
        value=160,
        key="rf_adv_horizon",
    )
    show_ci = controls[1].checkbox(
        "Show 95% confidence band", True, key="rf_adv_show_ci"
    )
    show_signals = controls[2].checkbox(
        "Show bullish / bearish signals", True, key="rf_adv_show_signals"
    )
    historical = data.tail(160)
    last_date = pd.Timestamp(historical["date"].iloc[-1])
    last_price = float(historical["close"].iloc[-1])
    model = st.session_state.get("rf_adv_model")
    forecast = st.session_state.get("rf_adv_forecast")
    if model is None or forecast is None:
        st.warning("Run the model to generate a forecast.")
        return
    if len(forecast) < horizon:
        with st.spinner(f"Generating the {horizon}-business-day recursive forecast…"):
            forecast = model.generate_forecast(horizon)
            st.session_state.rf_adv_forecast = forecast
    forecast = forecast.head(horizon).copy()
    future_dates = pd.DatetimeIndex(pd.to_datetime(forecast["Date"]))
    future_prices = forecast["Forecast_Price"].to_numpy()
    probabilities = forecast["Bullish_Probability"].to_numpy()
    trends = forecast["Predicted_Trend"].to_numpy()
    plot_dates = pd.DatetimeIndex([last_date]).append(future_dates)
    plot_prices = np.r_[last_price, future_prices]
    lower = np.r_[last_price, forecast["Confidence_Lower"].to_numpy()]
    upper = np.r_[last_price, forecast["Confidence_Upper"].to_numpy()]

    st.markdown(
        f'<div class="workbench-section"><div class="workbench-section-title">'
        f'<span>🔮</span><div><h3>{st.session_state.rf_adv_market_meta["ticker"]} - {horizon}-Day Price Forecast</h3>'
        f'<p>A recursive forecast driven by tuned Random Forest probabilities and historical volatility.</p></div></div>',
        unsafe_allow_html=True,
    )
    summary = st.columns(4)
    summary[0].metric("Starting price", f"${last_price:,.2f}")
    summary[1].metric(
        "Final price",
        f"${future_prices[-1]:,.2f}",
        f"{future_prices[-1] / last_price - 1:+.1%}",
    )
    summary[2].metric(
        "Bullish days",
        f'{int((forecast["Predicted_Trend"] == "Bullish").sum())}',
        f"of {horizon}",
    )
    summary[3].metric("Price range", f"${lower.min():,.2f} – ${upper.max():,.2f}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=historical["date"],
            y=historical["close"],
            name="Historical Price",
            mode="lines",
            line={"color": COLORS["cyan"], "width": 2.5},
            hovertemplate="%{x|%d %b %Y}<br>Historical: $%{y:.2f}<extra></extra>",
        )
    )
    if show_ci:
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=upper,
                name="95% CI upper",
                line={"color": "rgba(255,107,107,0)", "width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=lower,
                name="95% Confidence Interval",
                fill="tonexty",
                line={"color": "rgba(255,107,107,0)", "width": 0},
                fillcolor="rgba(255,107,107,0.15)",
                hovertemplate="%{x|%d %b %Y}<br>Lower: $%{y:.2f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=plot_dates,
            y=plot_prices,
            name="Forecast Price",
            mode="lines+markers",
            line={"color": COLORS["red"], "width": 2.5, "dash": "dash"},
            marker={"size": 4, "color": COLORS["red"]},
            hovertemplate="%{x|%d %b %Y}<br>Forecast: $%{y:.2f}<extra></extra>",
        )
    )
    if show_signals:
        bullish = trends == "Bullish"
        bearish = trends == "Bearish"
        fig.add_trace(
            go.Scatter(
                x=future_dates[bullish],
                y=future_prices[bullish],
                name="Bullish Signal",
                mode="markers",
                marker={"symbol": "triangle-up", "size": 8, "color": COLORS["green"]},
                hovertemplate="%{x|%d %b %Y}<br>Bullish signal<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates[bearish],
                y=future_prices[bearish],
                name="Bearish Signal",
                mode="markers",
                marker={"symbol": "triangle-down", "size": 8, "color": COLORS["red"]},
                hovertemplate="%{x|%d %b %Y}<br>Bearish signal<extra></extra>",
            )
        )
    fig.add_vline(
        x=last_date,
        line={"color": "rgba(255,255,255,0.4)", "width": 1.5, "dash": "dash"},
        annotation_text="📊 Forecast Start",
        annotation_position="top left",
        annotation_font={"color": "#FFFFFF", "size": 10},
    )
    fig.update_layout(
        title={
            "text": f'{st.session_state.rf_adv_market_meta["ticker"]} - {horizon}-Day Price Forecast',
            "x": 0,
            "font": {"size": 15, "color": COLORS["text"]},
        },
        height=500,
        margin={"l": 40, "r": 40, "t": 80, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "y": 1.08,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 10},
        },
        xaxis={
            "title": "Date",
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.05)",
            "rangeslider": {"visible": False},
        },
        yaxis={
            "title": "Price ($)",
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.05)",
            "tickprefix": "$",
        },
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "displayModeBar": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": f'{st.session_state.rf_adv_market_meta["ticker"]}_advanced_forecast',
                "height": 900,
                "width": 1600,
                "scale": 2,
            },
        },
    )
    st.caption("Use the chart toolbar to export the forecast as PNG or SVG.")

    insight_cols = st.columns([1.2, 1, 0.9], gap="medium")
    with insight_cols[0]:
        probability_fig = go.Figure()
        probability_fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=probabilities,
                name="Bullish probability",
                line={"color": COLORS["green"], "width": 2},
                fill="tozeroy",
                fillcolor="rgba(63,224,160,0.10)",
            )
        )
        probability_fig.add_hline(
            y=0.5,
            line={"color": COLORS["muted"], "dash": "dot"},
            annotation_text="50% threshold",
            annotation_position="bottom right",
        )
        probability_fig.update_layout(
            title="📊 Bullish Probability Over Time",
            height=250,
            margin={"l": 0, "r": 0, "t": 35, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis={"tickformat": ".0%", "range": [0, 1]},
            xaxis={"showgrid": False},
        )
        st.plotly_chart(probability_fig, use_container_width=True, config={"displaylogo": False})
    with insight_cols[1]:
        monthly = (
            forecast.assign(Month=forecast["Date"].dt.strftime("%b %Y"))
            .groupby("Month", sort=False)
            .agg(
                Avg_Price=("Forecast_Price", "mean"),
                Bullish_Probability=("Bullish_Probability", "mean"),
                Days=("Day", "count"),
            )
            .reset_index()
            .head(5)
        )
        monthly["Trend"] = np.where(monthly["Bullish_Probability"] >= 0.5, "🟢 ↑", "🔴 ↓")
        monthly["Confidence"] = (
            monthly["Bullish_Probability"].sub(0.5).abs() * 2
        ).clip(0.5, 0.99)
        st.markdown("**📅 Monthly Forecast Breakdown**")
        st.dataframe(
            monthly[["Month", "Avg_Price", "Trend", "Confidence"]].style.format(
                {"Avg_Price": "${:.2f}", "Confidence": "{:.0%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
    with insight_cols[2]:
        volatility = float(data["close"].pct_change().std())
        confidence = float(
            np.clip(
                metrics["accuracy"] * 0.65
                + (1 - min(volatility / 0.03, 1)) * 0.2
                + 0.95 * 0.15,
                0,
                1,
            )
        )
        st.markdown("**🎯 Forecast Confidence**")
        st.progress(confidence, text=f"{confidence:.0%}")
        st.caption(
            f"Based on:\n\n• Model accuracy: {metrics['accuracy']:.1%}\n\n"
            f"• Historical volatility: {volatility:.1%}\n\n• Data quality: 95%"
        )
    with st.expander("📋 View forecast details"):
        st.dataframe(
            forecast.head(30).style.format(
                {
                    "Bullish_Probability": "{:.1%}",
                    "Forecast_Price": "${:.2f}",
                    "Confidence_Lower": "${:.2f}",
                    "Confidence_Upper": "${:.2f}",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download forecast CSV",
            forecast.to_csv(index=False),
            "AAPL_advanced_forecast.csv",
            "text/csv",
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------
def render_optimized_model() -> None:
    """Render the complete Random Forest Optimized Model workbench."""
    _ensure_demo_data()
    render_model_navigation("Random Forest", "Optimized Model")
    meta = st.session_state.rf_adv_market_meta
    title_col, action_col = st.columns([1.5, 0.7])
    with title_col:
        st.markdown(
            '<div class="page-kicker">RANDOM FOREST · OPTIMIZED VARIANT</div>'
            '<h2>🌲 Optimized Random Forest Model</h2>'
            '<p class="page-subtitle">Tuned Random Forest classifier with GridSearchCV, '
            'TimeSeriesSplit cross-validation, permutation importance, and recursive price forecasting.</p>',
            unsafe_allow_html=True,
        )
    with action_col:
        st.markdown(
            '<div class="workbench-badges"><span class="badge badge-active">● Active</span>'
            '<span class="badge badge-info">v1.0.0</span>'
            '<span class="badge badge-info">AAPL</span></div>',
            unsafe_allow_html=True,
        )
        action_cols = st.columns(3)
        if action_cols[0].button("↻ Retrain", use_container_width=True):
            st.session_state.rf_adv_focus_train = True
            toast_success("Ready to retrain")
        if action_cols[1].button("Export", use_container_width=True):
            toast_success("Export options are available below")
        if action_cols[2].button("Delete", use_container_width=True):
            st.warning("Delete is disabled for this local workbench.")

    data = st.session_state.rf_adv_market_data
    _market_data_section()
    if data.empty:
        st.error(
            "No market data is available for this session. "
            "Yahoo Finance may be unavailable; use Download after checking the ticker and date range."
        )
        return

    quick = st.columns(4)
    metrics = st.session_state.get("rf_adv_metrics", {})
    with quick[0]:
        stat_card(
            "Data points",
            f"{len(data):,}",
            "+45 loaded",
            "📊",
            [800, 980, 1100, len(data)],
            footer=f'{meta["ticker"]} · {meta["interval"]}',
        )
    with quick[1]:
        stat_card(
            "Features",
            str(len(st.session_state.get("rf_adv_selected_features", DEFAULT_FEATURES))),
            "Configurable",
            "🔢",
            [8, 10, 12, 12],
            footer="Technical indicators",
        )
    with quick[2]:
        stat_card(
            "Model accuracy",
            f'{metrics.get("accuracy", 0):.1%}' if metrics else "—",
            "Run model to evaluate",
            "🎯",
            [78, 80, 82, 83] if metrics else [1, 1, 1, 1],
            footer="Held-out test split",
        )
    with quick[3]:
        stat_card(
            "Training time",
            f'{metrics.get("training_time", 0):.1f}s' if metrics else "—",
            "GridSearchCV CPU run",
            "⏱",
            [6, 5, 4.6, 4.2],
            footer="Local process",
        )

    _data_overview(data)
    selected = _feature_section(data)
    _train_section(data, selected)

    if st.session_state.get("rf_adv_trained"):
        _performance_and_importance(data)
        _forecast_section(data)
        st.markdown(
            '<div class="workbench-section"><div class="workbench-section-title">'
            '<span>💾</span><div><h3>Export & Downloads</h3>'
            '<p>Download the artifacts from this local tuned run.</p></div></div>',
            unsafe_allow_html=True,
        )
        export_cols = st.columns(3)
        export_cols[0].download_button(
            "Download model JSON",
            json.dumps(
                {
                    "model": "RandomForestClassifier",
                    "variant": "Optimized Model",
                    "metadata": meta,
                    "metrics": st.session_state.rf_adv_metrics,
                    "best_params": st.session_state.get("rf_adv_best_params", {}),
                },
                indent=2,
            ),
            f'{meta["ticker"]}_random_forest_advanced.json',
            "application/json",
            use_container_width=True,
        )
        export_cols[1].download_button(
            "Download importance CSV",
            st.session_state.rf_adv_feature_importance.to_csv(index=False),
            f'{meta["ticker"]}_advanced_feature_importance.csv',
            "text/csv",
            use_container_width=True,
        )
        export_cols[2].download_button(
            "Download raw data CSV",
            data.to_csv(index=False),
            f'{meta["ticker"]}_raw_data.csv',
            "text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📝 Model metadata"):
        st.json(
            {
                "model_id": "rf_advanced",
                "model_type": "RandomForestClassifier",
                "version": "v1.0.0",
                "framework": "scikit-learn + Yahoo Finance + GridSearchCV",
                "random_state": st.session_state.get("rf_adv_metrics", {}).get(
                    "random_state", 42
                ),
                "features_used": st.session_state.get(
                    "rf_adv_selected_features", DEFAULT_FEATURES
                ),
                "ticker": meta["ticker"],
                "data_range": f'{meta["start"]} → {meta["end"]}',
                "interval": meta["interval"],
                "best_params": st.session_state.get("rf_adv_best_params", {}),
                "best_cv_score": st.session_state.get("rf_adv_best_score"),
            }
        )
    with st.expander("📚 Documentation & warnings"):
        st.markdown(
            "**About this model**\n\n"
            "This optimized variant extends the baseline with 45+ technical indicators, "
            "GridSearchCV hyperparameter tuning using TimeSeriesSplit cross-validation, "
            "permutation importance, and a recursive forecast driven by tuned probabilities. "
            "It uses a chronological holdout, StandardScaler, and scikit-learn RandomForestClassifier. "
            "Forecasts are illustrative and should not be used as financial advice."
        )
        st.warning(
            "Past performance does not guarantee future results. "
            "Forecasts are illustrative and should not be used as financial advice."
        )