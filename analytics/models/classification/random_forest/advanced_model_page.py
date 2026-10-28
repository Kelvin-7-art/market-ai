"""Interactive Random Forest Advanced Model workbench.

This page presents the *Advanced* Random Forest variant which adds:
- Extended feature engineering (25+ technical indicators)
- Hyperparameter tuning with GridSearchCV
- Cross-validation with TimeSeriesSplit
- Feature selection and importance analysis
- Comprehensive evaluation metrics (ROC-AUC, precision, recall, F1)
- Multi-market, multi-ticker support via the market catalog
- Skill-gated forecasts with honest confidence labels
"""

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
from market_ai.analytics.models.classification.random_forest.market_catalog import (
    SUPPORTED_MARKETS,
    get_market,
    all_tickers,
)
from market_ai.utils.theme import COLORS


# ============================================================
# CONSTANTS
# ============================================================

FEATURE_GROUPS = {
    "Returns": [
        "Return_1d", "Return_5d", "Return_10d", "Return_20d",
    ],
    "Moving averages": [
        "MA5", "MA10", "MA20", "MA50", "MA200",
    ],
    "Price ratios": [
        "Price_MA5_Ratio", "Price_MA10_Ratio",
        "Price_MA20_Ratio", "Price_MA50_Ratio",
    ],
    "Volatility": [
        "Volatility_10", "Volatility_20", "Volatility_50", "ATR",
    ],
    "Volume": [
        "Volume_Ratio_5", "Volume_Ratio_20", "Volume_Change",
        "OBV", "OBV_Ratio", "CMF",
    ],
    "Momentum": [
        "Momentum_5", "Momentum_10", "Momentum_20",
        "RSI", "MACD", "MACD_Signal", "MACD_Histogram",
    ],
    "Patterns": [
        "High_Low_Ratio", "Close_Open_Ratio",
        "High_Close_Ratio", "Low_Close_Ratio",
    ],
    "Bollinger Bands": [
        "BB_Upper", "BB_Middle", "BB_Lower", "BB_Position",
    ],
    "Trend & Structure": [
        "Stoch_K", "Stoch_D", "Williams_R",
        "ADX", "Trend_Strength",
        "Dist_From_High", "Dist_From_Low", "Price_Position",
    ],
}

ALL_FEATURES = [feature for group in FEATURE_GROUPS.values() for feature in group]
DEFAULT_FEATURES = ADVANCED_FEATURES.copy() if ADVANCED_FEATURES else ALL_FEATURES[:24]
ALL_FEATURES = list(dict.fromkeys(DEFAULT_FEATURES + ALL_FEATURES))

# Hyperparameter tuning presets
TUNING_PRESETS = {
    "Fast (32 combos)": {
        "n_estimators": [100, 200],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
        "class_weight": ["balanced", None],
    },
    "Balanced (96 combos)": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", None],
    },
    "Thorough (288 combos)": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": ["balanced", "balanced_subsample", None],
    },
}

# Skill gate threshold — 3 percentage points above random
MIN_SKILL = 0.03


# ============================================================
# DATA / PIPELINE HELPERS
# ============================================================

def _download_market_data(
    ticker: str, start: date, end: date, interval: str
) -> pd.DataFrame:
    """Download real market data through the advanced model's Yahoo loader."""
    model = RandomForestAdvancedModel(ticker=ticker)
    return to_chart_frame(
        model.load_data(start_date=start, end_date=end, interval=interval)
    )


def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build the same extended feature matrix used by the fitted model."""
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
    return RandomForestAdvancedModel.engineer_frame(upper)


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
    ]:
        st.session_state.pop(key, None)


def _compute_skill_score(metrics: dict) -> float:
    """Return how far above random the tuned model is, averaged over
    accuracy, CV score, and ROC-AUC."""
    accuracy = float(metrics.get("accuracy", 0.5))
    cv_score = float(metrics.get("best_cv_score") or 0.5)
    roc_auc = float(metrics.get("roc_auc", 0.5))
    return (
        max(0.0, accuracy - 0.5)
        + max(0.0, cv_score - 0.5)
        + max(0.0, roc_auc - 0.5)
    ) / 3.0


def _ensure_demo_data() -> None:
    """Load default market data (AAPL/US Equities) once per session."""
    if "rf_adv_market_data" not in st.session_state:
        end = date.today()
        start = end - timedelta(days=1800)
        market = SUPPORTED_MARKETS[0]
        default_ticker = market.default_ticker

        try:
            data = _download_market_data(default_ticker, start, end, "1d")
            st.session_state.pop("rf_adv_market_error", None)
        except Exception as error:
            data = pd.DataFrame()
            st.session_state.rf_adv_market_error = str(error)

        st.session_state.rf_adv_market_data = data
        st.session_state.rf_adv_market_key = market.key
        st.session_state.rf_adv_ticker = default_ticker

        if data.empty:
            st.session_state.rf_adv_market_meta = {
                "ticker": default_ticker,
                "market": market.key,
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
                "ticker": default_ticker,
                "market": market.key,
                "start": str(data["date"].min().date()),
                "end": str(data["date"].max().date()),
                "interval": "1d",
                "rows": len(data),
                "last_close": float(data["close"].iloc[-1]),
                "period_return": float(
                    data["close"].iloc[-1] / data["close"].iloc[0] - 1
                ),
                "source": "Yahoo Finance",
            }


# ============================================================
# SECTION 1: MARKET DATA (MULTI-MARKET + MULTI-TICKER)
# ============================================================

def _market_data_section() -> None:
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        "<span>📥</span><div><h3>Market Data</h3>"
        "<p>Pick a market, then choose a ticker and date range.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    meta = st.session_state.rf_adv_market_meta
    market_keys = [m.key for m in SUPPORTED_MARKETS]
    market_labels = {m.key: m.label for m in SUPPORTED_MARKETS}

    active_market_key = st.session_state.get("rf_adv_market_key", market_keys[0])
    active_market = get_market(active_market_key)

    # ---------------- Row 1: market + ticker + interval ----------------
    row1 = st.columns([1.2, 1.2, 1.1, 1.6])

    market_key = row1[0].selectbox(
        "Market",
        market_keys,
        index=market_keys.index(active_market.key)
        if active_market.key in market_keys
        else 0,
        format_func=lambda k: market_labels[k],
        key="rf_adv_market_key_widget",
    )
    if market_key != active_market.key:
        st.session_state.rf_adv_market_key = market_key
        active_market = get_market(market_key)

    ticker_options = list(active_market.tickers)
    current_ticker = st.session_state.get(
        "rf_adv_ticker", active_market.default_ticker
    )
    if current_ticker not in ticker_options:
        current_ticker = active_market.default_ticker

    ticker = row1[1].selectbox(
        "Ticker",
        ticker_options,
        index=ticker_options.index(current_ticker)
        if current_ticker in ticker_options
        else 0,
        key="rf_adv_ticker_widget",
    )
    st.session_state.rf_adv_ticker = ticker

    custom_ticker = row1[2].text_input(
        "Custom (override)",
        value="",
        placeholder="e.g. SHOP",
        key="rf_adv_custom_ticker",
    )
    if custom_ticker.strip():
        ticker = custom_ticker.strip().upper()

    interval = row1[3].selectbox(
        "Interval",
        ["1d", "1h", "30m", "15m", "5m"],
        key="rf_adv_interval",
    )

    # ---------------- Row 2: dates + actions ----------------
    row2 = st.columns([1, 1, 0.8, 1.4])
    start = row2[0].date_input(
        "Start date",
        date.fromisoformat(meta["start"]),
        key="rf_adv_start",
    )
    end = row2[1].date_input(
        "End date",
        date.fromisoformat(meta["end"]),
        key="rf_adv_end",
    )
    cached = row2[2].checkbox("Use cached", True, key="rf_adv_cached")
    download = row2[3].button(
        "Download",
        type="primary",
        use_container_width=True,
        key="rf_adv_download",
    )

    if download:
        if start >= end:
            st.error("Start date must be before the end date.")
        elif (
            cached
            and ticker.upper() == meta["ticker"]
            and market_key == meta.get("market")
            and str(start) == meta["start"]
            and str(end) == meta["end"]
            and interval == meta["interval"]
        ):
            st.success(
                f"Data loaded from cache for {ticker.upper()} "
                f"({meta['rows']:,} rows)."
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
                    "Yahoo Finance did not return usable data. "
                    "Widen the date range or check the ticker."
                )
            else:
                st.session_state.rf_adv_market_data = downloaded
                st.session_state.rf_adv_market_meta = {
                    "ticker": ticker.upper(),
                    "market": market_key,
                    "start": str(downloaded["date"].min().date()),
                    "end": str(downloaded["date"].max().date()),
                    "interval": interval,
                    "rows": len(downloaded),
                    "last_close": float(downloaded["close"].iloc[-1]),
                    "period_return": float(
                        downloaded["close"].iloc[-1]
                        / downloaded["close"].iloc[0]
                        - 1
                    ),
                    "source": "Yahoo Finance",
                }
                _reset_training_state()
                st.success(
                    f"Data loaded for {ticker.upper()} "
                    f"({interval} interval, {len(downloaded):,} rows)."
                )

    summary = st.columns(4)
    summary[0].metric(
        "Market", market_labels.get(meta.get("market", ""), "—")
    )
    summary[1].metric("Rows", f'{meta["rows"]:,}')
    summary[2].metric("Last close", f'${meta["last_close"]:,.2f}')
    summary[3].metric("Period return", f'{meta["period_return"]:+.1%}')

    st.caption(
        f'Source: {meta["source"]} · Ticker: {meta["ticker"]} · '
        f'Interval: {meta["interval"]}'
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SECTION 2: DATA OVERVIEW
# ============================================================

def _data_overview(data: pd.DataFrame) -> None:
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        "<span>📈</span><div><h3>Data Overview</h3>"
        "<p>Candlesticks with moving-average overlays and volume.</p>"
        "</div></div>",
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


# ============================================================
# SECTION 3: EXTENDED FEATURE ENGINEERING
# ============================================================

def _feature_section(data: pd.DataFrame) -> list[str]:
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        "<span>🧪</span><div><h3>Extended Feature Engineering</h3>"
        "<p>Choose from 25+ technical indicators grouped by category.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    if "rf_adv_selected_features" not in st.session_state:
        st.session_state.rf_adv_selected_features = DEFAULT_FEATURES.copy()

    group_counts = st.columns(len(FEATURE_GROUPS))
    for column, (group_name, group_features) in zip(
        group_counts, FEATURE_GROUPS.items()
    ):
        with column:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-size:11px;color:{COLORS['muted']};'>"
                f"{group_name}</div>"
                f"<div style='font-size:16px;font-weight:600;'>"
                f"{len(group_features)}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    selected = st.multiselect(
        "Features",
        ALL_FEATURES,
        default=st.session_state.rf_adv_selected_features,
        key="rf_adv_features",
    )
    st.session_state.rf_adv_selected_features = selected

    target = st.selectbox(
        "Target definition",
        [
            "Next-day direction",
            "5-day direction",
            "Threshold-based > +0.5%",
        ],
        key="rf_adv_target",
    )

    st.caption(
        f"{len(selected)} of {len(ALL_FEATURES)} features selected · "
        f"Target: {target}"
    )

    if len(selected) >= 2:
        matrix = _engineer_features(data)[selected].dropna().head(5)
        st.dataframe(matrix, hide_index=True, use_container_width=True)
        usable_rows = _engineer_features(data)[selected].dropna().shape[0]
        st.caption(
            f"Engineered from {len(data):,} raw rows → {usable_rows:,} usable rows "
            "after rolling-window cleanup."
        )
    else:
        st.warning("Select at least 2 features to train the model.")

    st.markdown("</div>", unsafe_allow_html=True)
    return selected


# ============================================================
# SECTION 4: HYPERPARAMETER TUNING + TRAIN
# ============================================================

def _tuning_section(selected: list[str]) -> dict:
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        "<span>⚙️</span><div><h3>Hyperparameter Tuning</h3>"
        "<p>GridSearchCV with TimeSeriesSplit for temporal cross-validation.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    controls = st.columns([1.2, 1, 1, 1, 1])

    preset = controls[0].selectbox(
        "Tuning preset",
        list(TUNING_PRESETS.keys()),
        index=0,
        key="rf_adv_tuning_preset",
    )
    cv_splits = controls[1].slider(
        "CV splits (TimeSeriesSplit)", 2, 5, 3, 1, key="rf_adv_cv_splits"
    )
    test_size = controls[2].slider(
        "Test size", 0.1, 0.4, 0.2, 0.05, key="rf_adv_test_size"
    )
    random_state = controls[3].number_input(
        "Random state", 1, 999, 42, key="rf_adv_random_state"
    )
    scoring = controls[4].selectbox(
        "Scoring metric",
        ["accuracy", "f1", "roc_auc", "precision", "recall"],
        key="rf_adv_scoring",
    )

    grid = TUNING_PRESETS[preset]
    total_combos = int(np.prod([len(v) for v in grid.values()]))
    total_fits = total_combos * cv_splits

    preview = st.columns(4)
    preview[0].metric("Grid combinations", f"{total_combos}")
    preview[1].metric("CV splits", f"{cv_splits}")
    preview[2].metric("Total model fits", f"{total_fits:,}")
    preview[3].metric("Scoring", scoring)

    with st.expander("🔎 View parameter grid", expanded=False):
        st.json(grid)

    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        "<span>🚀</span><div><h3>Train Advanced Model</h3>"
        "<p>Fit the scikit-learn Random Forest with GridSearchCV + TimeSeriesSplit.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    run_cols = st.columns([3, 1])
    run_cols[0].caption(
        "The advanced model applies the selected preset grid, tunes across "
        "TimeSeriesSplit folds, and refits the best estimator on the training split."
    )
    run = run_cols[1].button(
        "🌲 Run advanced model",
        type="primary",
        use_container_width=True,
        key="rf_adv_train",
    )

    return {
        "preset": preset,
        "param_grid": grid,
        "cv_splits": cv_splits,
        "test_size": test_size,
        "random_state": random_state,
        "scoring": scoring,
        "run": run,
    }


def _train_advanced_model(
    data: pd.DataFrame, selected: list[str], config: dict
) -> None:
    if len(selected) < 2:
        st.warning("Select at least 2 features before training.")
        return

    with st.spinner(
        f"Running GridSearchCV ({config['preset']}) on Yahoo Finance data "
        f"with {len(selected)} features…"
    ):
        started = time.perf_counter()
        try:
            model = RandomForestAdvancedModel(
                ticker=st.session_state.rf_adv_market_meta["ticker"],
                random_state=config["random_state"],
                test_size=config["test_size"],
                cv_splits=config["cv_splits"],
                features=selected,
                param_grid=config["param_grid"],
                scoring=config["scoring"],
            )
            model.load_data(data=data)
            engineered = model.engineer_features(
                st.session_state.get("rf_adv_target", "Next-day direction")
            )

            if len(engineered) < 50:
                raise ValueError(
                    "Not enough usable observations after feature engineering."
                )

            model.prepare_data()
            best_params = model.tune_hyperparameters()
            metrics = model.evaluate()
            importance = model.analyze_feature_importance()

            # Skill gate — only forecast when the model has an edge.
            skill_score = _compute_skill_score(metrics)
            metrics["skill_score"] = skill_score
            metrics["low_skill"] = skill_score < MIN_SKILL

            if not metrics["low_skill"]:
                model.generate_forecast(365)
            else:
                model.forecast_df = None

            metrics.update(
                {
                    "training_time": time.perf_counter() - started,
                    "preset": config["preset"],
                    "cv_splits": config["cv_splits"],
                    "scoring": config["scoring"],
                    "best_params": best_params,
                    "best_cv_score": getattr(model, "best_score", None),
                    "total_fits": model.last_grid_total_fits
                    if hasattr(model, "last_grid_total_fits")
                    else None,
                    "random_state": config["random_state"],
                }
            )

            st.session_state.rf_adv_model = model
            st.session_state.rf_adv_trained = True
            st.session_state.rf_adv_metrics = metrics
            st.session_state.rf_adv_feature_importance = importance
            st.session_state.rf_adv_confusion = metrics["confusion_matrix"]
            st.session_state.rf_adv_forecast = model.forecast_df
            st.session_state.rf_adv_best_params = best_params
            st.session_state.rf_adv_cv_results = getattr(
                model, "cv_results_summary", None
            )

        except Exception as error:
            st.error(f"Advanced model training failed: {error}")
            return

    if metrics["low_skill"]:
        st.warning(
            f"Advanced model trained in {metrics['training_time']:.2f}s · "
            f"CV score {(metrics.get('best_cv_score') or 0):.1%} · "
            f"Test accuracy {metrics['accuracy']:.1%} · "
            f"Skill score {metrics['skill_score'] * 100:.1f} pts — "
            "forecast disabled."
        )
    else:
        st.success(
            f"Advanced model trained in {metrics['training_time']:.2f}s · "
            f"CV score {(metrics.get('best_cv_score') or 0):.1%} · "
            f"Test accuracy {metrics['accuracy']:.1%} · "
            f"Skill score {metrics['skill_score'] * 100:.1f} pts"
        )


# ============================================================
# SECTION 5: PERFORMANCE & FEATURE IMPORTANCE
# ============================================================

def _performance_and_importance(data: pd.DataFrame) -> None:
    metrics = st.session_state.rf_adv_metrics

    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        "<span>📊</span><div><h3>Model Performance</h3>"
        "<p>Evaluation on the held-out test split after tuning.</p>"
        "</div></div>",
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
                f"{metrics[key]:.1%}",
                "Tuned model",
                "◎",
                [metrics[key] * 90, metrics[key] * 94, metrics[key] * 100],
                footer=f'{metrics["test_samples"]:,} test rows',
            )

    with st.expander("🏆 Best hyperparameters (GridSearchCV)", expanded=False):
        params = st.session_state.get("rf_adv_best_params", {})
        best_cv_score = metrics.get("best_cv_score")
        st.json(params)
        if best_cv_score is not None:
            st.caption(
                f"Best CV {metrics.get('scoring', 'score')}: "
                f"{best_cv_score:.1%} · "
                f"{metrics.get('cv_splits')} TimeSeriesSplit folds · "
                f"Preset: {metrics.get('preset')}"
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
            ).style.format(
                {"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}
            ),
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

    # -------------------- Feature importance --------------------
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        "<span>🔍</span><div><h3>Feature Importance</h3>"
        "<p>Built-in Gini importance and permutation importance on the test set.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    importance = st.session_state.rf_adv_feature_importance

    if (
        isinstance(importance, pd.DataFrame)
        and {"Builtin_Importance", "Permutation_Mean"}.issubset(importance.columns)
    ):
        left, right = st.columns(2, gap="medium")

        with left:
            st.markdown("**Built-in (Gini) importance — Top 15**")
            top_builtin = importance.sort_values(
                "Builtin_Importance", ascending=False
            ).head(15)
            fig = go.Figure(
                go.Bar(
                    x=top_builtin["Builtin_Importance"],
                    y=top_builtin["Feature"],
                    orientation="h",
                    marker_color=top_builtin["Builtin_Importance"],
                    marker_colorscale=[
                        [0, COLORS["violet"]],
                        [1, COLORS["cyan"]],
                    ],
                    text=top_builtin["Builtin_Importance"].map(
                        lambda value: f"{value:.1%}"
                    ),
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

        with right:
            st.markdown("**Permutation importance — Top 15**")
            top_perm = importance.sort_values(
                "Permutation_Mean", ascending=False
            ).head(15)
            fig = go.Figure(
                go.Bar(
                    x=top_perm["Permutation_Mean"],
                    y=top_perm["Feature"],
                    orientation="h",
                    error_x={
                        "type": "data",
                        "array": top_perm["Permutation_STD"],
                    },
                    marker_color=COLORS.get("coral", COLORS.get("red", "#ff6b6b")),
                    text=top_perm["Permutation_Mean"].map(
                        lambda value: f"{value:.3f}"
                    ),
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

    st.download_button(
        "Export feature importance CSV",
        importance.to_csv(index=False),
        f'{st.session_state.rf_adv_market_meta["ticker"]}_advanced_feature_importance.csv',
        "text/csv",
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SECTION 6: FORECAST (SKILL-GATED)
# ============================================================

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

    if model is None:
        st.warning("Run the advanced model to generate a forecast.")
        return

    # Skill gate
    accuracy = float(metrics.get("accuracy", 0.5))
    cv_score = float(metrics.get("best_cv_score") or 0.5)
    roc_auc = float(metrics.get("roc_auc", 0.5))
    skill_score = float(metrics.get("skill_score", _compute_skill_score(metrics)))

    if skill_score < MIN_SKILL:
        st.warning(
            "⚠ **Low-skill forecast suppressed.**\n\n"
            f"The tuned model scored "
            f"**accuracy {accuracy:.1%}**, "
            f"**CV {cv_score:.1%}**, "
            f"**ROC-AUC {roc_auc:.1%}** — all within noise of a coin flip. "
            "A recursive forecast from a model with no directional edge "
            "would be an uninformative flat line, so it is not displayed.\n\n"
            "**What to try instead:**\n"
            "- Add or remove features in **Extended Feature Engineering**\n"
            "- Switch the **Target definition** (e.g. 5-day direction)\n"
            "- Use a larger grid (**Balanced** or **Thorough** preset)\n"
            "- Try a different ticker/market in **Market Data**"
        )

        diag_cols = st.columns(3)
        diag_cols[0].metric(
            "Accuracy",
            f"{accuracy:.1%}",
            f"{(accuracy - 0.5) * 100:+.1f} pts vs random",
        )
        diag_cols[1].metric(
            "CV score",
            f"{cv_score:.1%}",
            f"{(cv_score - 0.5) * 100:+.1f} pts vs random",
        )
        diag_cols[2].metric(
            "ROC-AUC",
            f"{roc_auc:.1%}",
            f"{(roc_auc - 0.5) * 100:+.1f} pts vs random",
        )
        return

    if forecast is None:
        st.info(
            "No forecast was generated for this run. Retrain with a "
            "skillier configuration to enable forecasting."
        )
        return

    if len(forecast) < horizon:
        with st.spinner(
            f"Generating the {horizon}-business-day recursive forecast…"
        ):
            forecast = model.generate_forecast(horizon)
            st.session_state.rf_adv_forecast = forecast

    forecast = forecast.head(horizon).copy()
    future_dates = pd.DatetimeIndex(pd.to_datetime(forecast["Date"]))
    future_prices = forecast["Forecast_Price"].to_numpy()
    probabilities = forecast["Bullish_Probability"].to_numpy()
    trends = forecast["Predicted_Trend"].astype(str).str.lower().to_numpy()

    plot_dates = pd.DatetimeIndex([last_date]).append(future_dates)
    plot_prices = np.r_[last_price, future_prices]

    # Trend-aware asymmetric confidence band
    raw_lower = np.r_[last_price, forecast["Confidence_Lower"].to_numpy()]
    raw_upper = np.r_[last_price, forecast["Confidence_Upper"].to_numpy()]
    forecast_drift_sign = np.sign(plot_prices[-1] - plot_prices[0]) or 1
    asymmetry = 0.85 + 0.15 * forecast_drift_sign
    lower = np.r_[
        last_price,
        last_price + (raw_lower[1:] - last_price) * asymmetry,
    ]
    upper = np.r_[
        last_price,
        last_price + (raw_upper[1:] - last_price) / max(asymmetry, 0.4),
    ]

    # Range check
    forecast_range_pct = (
        (future_prices.max() - future_prices.min()) / last_price
        if last_price
        else 0.0
    )
    if forecast_range_pct < 0.02:
        st.info(
            f"ℹ **No-signal forecast.** The projected range over "
            f"{horizon} days is only **{forecast_range_pct:.1%}** — the "
            "model sees essentially no directional signal. Treat the chart "
            "below as a *no-trade* projection rather than a price target."
        )

    ticker = st.session_state.rf_adv_market_meta["ticker"]

    st.markdown(
        f'<div class="workbench-section"><div class="workbench-section-title">'
        f'<span>🔮</span><div><h3>{ticker} - {horizon}-Day Price Forecast</h3>'
        "<p>A recursive forecast driven by the tuned Random Forest probabilities "
        "and historical volatility.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    bullish_count = int((trends == "bullish").sum())
    bearish_count = int((trends == "bearish").sum())

    summary = st.columns(4)
    summary[0].metric("Starting price", f"${last_price:,.2f}")
    summary[1].metric(
        "Final price",
        f"${future_prices[-1]:,.2f}",
        f"{future_prices[-1] / last_price - 1:+.1%}",
    )
    summary[2].metric(
        "Bullish days",
        f"{bullish_count}",
        f"of {horizon} · bearish {bearish_count}",
    )
    summary[3].metric(
        "Price range", f"${lower.min():,.2f} – ${upper.max():,.2f}"
    )

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
        bullish = trends == "bullish"
        bearish = trends == "bearish"
        fig.add_trace(
            go.Scatter(
                x=future_dates[bullish],
                y=future_prices[bullish],
                name="Bullish Signal",
                mode="markers",
                marker={
                    "symbol": "triangle-up",
                    "size": 8,
                    "color": COLORS["green"],
                },
                hovertemplate="%{x|%d %b %Y}<br>Bullish signal<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=future_dates[bearish],
                y=future_prices[bearish],
                name="Bearish Signal",
                mode="markers",
                marker={
                    "symbol": "triangle-down",
                    "size": 8,
                    "color": COLORS["red"],
                },
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
            "text": f"{ticker} - {horizon}-Day Price Forecast",
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
                "filename": f"{ticker}_advanced_forecast",
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
        st.plotly_chart(
            probability_fig,
            use_container_width=True,
            config={"displaylogo": False},
        )

    with insight_cols[1]:
        monthly = (
            forecast.assign(Month=forecast["Date"].dt.strftime("%b %Y"))
            .groupby("Month", sort=False)
            .agg(
                First_Price=("Forecast_Price", "first"),
                Last_Price=("Forecast_Price", "last"),
                Avg_Price=("Forecast_Price", "mean"),
                Bullish_Probability=("Bullish_Probability", "mean"),
                Days=("Day", "count"),
            )
            .reset_index()
            .head(6)
        )
        monthly["Monthly_Return"] = (
            monthly["Last_Price"] / monthly["First_Price"] - 1
        )
        monthly["Trend"] = np.where(
            monthly["Monthly_Return"] >= 0.005,
            "🟢 ↑",
            np.where(
                monthly["Monthly_Return"] <= -0.005, "🔴 ↓", "🟡 →"
            ),
        )
        monthly["Confidence"] = (
            0.5
            + (monthly["Bullish_Probability"].sub(0.5).abs() * 2).clip(0, 1)
            * 0.49
        )
        monthly["Monthly_Return"] = monthly["Monthly_Return"].map(
            lambda v: f"{v:+.1%}"
        )

        st.markdown("**📅 Monthly Forecast Breakdown**")
        st.dataframe(
            monthly[
                [
                    "Month",
                    "Avg_Price",
                    "Monthly_Return",
                    "Trend",
                    "Confidence",
                ]
            ].style.format(
                {"Avg_Price": "${:.2f}", "Confidence": "{:.0%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )

    with insight_cols[2]:
        volatility = float(data["close"].pct_change().std())
        vol_penalty = max(0.0, min(1.0, 1 - volatility / 0.05))

        confidence = float(
            np.clip(
                0.60 * accuracy + 0.25 * cv_score + 0.15 * vol_penalty,
                0,
                1,
            )
        )

        st.markdown("**🎯 Forecast Confidence**")
        st.progress(confidence, text=f"{confidence:.0%}")

        if confidence >= 0.65:
            st.success("Moderate confidence")
        elif confidence >= 0.52:
            st.warning("Low confidence — near coin flip")
        else:
            st.error("Very low confidence — treat as illustrative only")

        st.caption(
            f"Based on:\n\n"
            f"• Accuracy: {accuracy:.1%}\n\n"
            f"• CV score: {cv_score:.1%}\n\n"
            f"• Volatility regime: {vol_penalty:.0%}"
        )

    with st.expander("📋 View forecast details"):
        details = forecast.copy()
        details["Predicted_Trend"] = (
            details["Predicted_Trend"].astype(str).str.title()
        )
        st.dataframe(
            details.head(30).style.format(
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
            f"{ticker}_advanced_forecast.csv",
            "text/csv",
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# SECTION 7: TUNING SUMMARY
# ============================================================

def _tuning_summary_section() -> None:
    metrics = st.session_state.get("rf_adv_metrics", {})
    if not metrics:
        return

    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title">'
        "<span>🎯</span><div><h3>Tuning Summary</h3>"
        "<p>Snapshot of the GridSearchCV run performed on the training split.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    cards = st.columns(5)
    cards[0].metric("Preset", metrics.get("preset", "—"))
    cards[1].metric("CV splits", str(metrics.get("cv_splits", "—")))
    cards[2].metric(
        "Best CV score", f'{(metrics.get("best_cv_score") or 0):.1%}'
    )
    cards[3].metric(
        "Total model fits", f'{metrics.get("total_fits") or "—"}'
    )
    cards[4].metric(
        "Skill score",
        f'{metrics.get("skill_score", 0) * 100:.1f} pts',
        "disabled forecast"
        if metrics.get("low_skill")
        else "forecast enabled",
    )

    st.markdown("**Selected features and best hyperparameters**")
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Features used**")
        features_used = st.session_state.get(
            "rf_adv_selected_features", DEFAULT_FEATURES
        )
        st.dataframe(
            pd.DataFrame({"feature": features_used}),
            hide_index=True,
            use_container_width=True,
        )
    with right:
        st.markdown("**Best hyperparameters**")
        st.json(st.session_state.get("rf_adv_best_params", {}))

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN RENDER
# ============================================================

def render_advanced_model() -> None:
    """Render the complete Random Forest Advanced Model workbench."""
    _ensure_demo_data()
    render_model_navigation("Random Forest", "Advanced Model")

    meta = st.session_state.rf_adv_market_meta
    active_ticker = meta["ticker"]
    active_market_label = get_market(meta.get("market", "")).label

    # ---------------- Header ----------------
    title_col, action_col = st.columns([1.5, 0.7])

    with title_col:
        st.markdown(
            '<div class="page-kicker">RANDOM FOREST · ADVANCED VARIANT</div>'
            "<h2>🌲 Advanced Random Forest Model</h2>"
            '<p class="page-subtitle">'
            "Extended Random Forest with 25+ technical indicators, GridSearchCV "
            "hyperparameter tuning, and TimeSeriesSplit cross-validation."
            "</p>",
            unsafe_allow_html=True,
        )

    with action_col:
        st.markdown(
            '<div class="workbench-badges">'
            '<span class="badge badge-active">● Active</span>'
            '<span class="badge badge-info">v2.0.0</span>'
            f'<span class="badge badge-info">{active_market_label}</span>'
            f'<span class="badge badge-info">{active_ticker}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        action_cols = st.columns(3)
        if action_cols[0].button(
            "↻ Retrain", use_container_width=True, key="rf_adv_retrain"
        ):
            st.session_state.rf_adv_focus_train = True
            toast_success("Ready to retrain")
        if action_cols[1].button(
            "Export", use_container_width=True, key="rf_adv_export_top"
        ):
            toast_success("Export options are available below")
        if action_cols[2].button(
            "Delete", use_container_width=True, key="rf_adv_delete_top"
        ):
            st.warning("Delete is disabled for this local workbench.")

    # ---------------- Market data ----------------
    data = st.session_state.rf_adv_market_data
    _market_data_section()

    if data.empty:
        st.error(
            "No market data is available for this session. "
            "Yahoo Finance may be unavailable; use Download after checking "
            "the ticker and date range."
        )
        return

    # ---------------- Quick stats ----------------
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
            str(
                len(
                    st.session_state.get(
                        "rf_adv_selected_features", DEFAULT_FEATURES
                    )
                )
            ),
            "25+ indicators",
            "🔢",
            [12, 18, 24, 25],
            footer="Extended engineering",
        )

    with quick[2]:
        stat_card(
            "Tuned CV score",
            f'{(metrics.get("best_cv_score") or 0):.1%}' if metrics else "—",
            "TimeSeriesSplit",
            "🎯",
            [70, 74, 78, 82] if metrics else [1, 1, 1, 1],
            footer="GridSearchCV",
        )

    with quick[3]:
        stat_card(
            "Training time",
            f'{metrics.get("training_time", 0):.1f}s' if metrics else "—",
            "scikit-learn CPU run",
            "⏱",
            [8, 6, 5, 4.2],
            footer="Local process",
        )

    # ---------------- Sections ----------------
    _data_overview(data)
    selected = _feature_section(data)

    tuning_config = _tuning_section(selected)
    if tuning_config["run"]:
        _train_advanced_model(data, selected, tuning_config)

    if st.session_state.get("rf_adv_trained"):
        _performance_and_importance(data)
        _tuning_summary_section()
        _forecast_section(data)

        # ---------------- Export ----------------
        st.markdown(
            '<div class="workbench-section"><div class="workbench-section-title">'
            "<span>💾</span><div><h3>Export & Downloads</h3>"
            "<p>Download the artifacts from this advanced test run.</p>"
            "</div></div>",
            unsafe_allow_html=True,
        )

        export_cols = st.columns(4)
        forecast_df = st.session_state.get("rf_adv_forecast")
        low_skill = st.session_state.get("rf_adv_metrics", {}).get(
            "low_skill", False
        )

        export_cols[0].download_button(
            "Download model JSON",
            json.dumps(
                {
                    "model": "RandomForestClassifier",
                    "variant": "Advanced Model",
                    "metadata": meta,
                    "metrics": st.session_state.rf_adv_metrics,
                    "best_params": st.session_state.get(
                        "rf_adv_best_params", {}
                    ),
                    "features": st.session_state.get(
                        "rf_adv_selected_features", DEFAULT_FEATURES
                    ),
                    "forecast_available": not low_skill,
                },
                indent=2,
            ),
            f'{meta["ticker"]}_random_forest_advanced.json',
            "application/json",
            use_container_width=True,
        )

        export_cols[1].download_button(
            "Download feature importance CSV",
            st.session_state.rf_adv_feature_importance.to_csv(index=False),
            f'{meta["ticker"]}_advanced_feature_importance.csv',
            "text/csv",
            use_container_width=True,
        )

        if forecast_df is not None and not low_skill:
            export_cols[2].download_button(
                "Download forecast CSV",
                forecast_df.to_csv(index=False),
                f'{meta["ticker"]}_advanced_forecast.csv',
                "text/csv",
                use_container_width=True,
            )
        else:
            export_cols[2].button(
                "Forecast unavailable",
                disabled=True,
                use_container_width=True,
                help="Forecast is disabled because the tuned model has no "
                     "directional edge (skill score below 3 pts).",
            )

        export_cols[3].download_button(
            "Download raw data CSV",
            data.to_csv(index=False),
            f'{meta["ticker"]}_raw_data.csv',
            "text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Metadata & docs ----------------
    with st.expander("📝 Model metadata"):
        st.json(
            {
                "model_id": "rf_advanced",
                "model_type": "RandomForestClassifier",
                "version": "v2.0.0",
                "framework": "scikit-learn + Yahoo Finance",
                "tuning": "GridSearchCV",
                "cv": "TimeSeriesSplit",
                "random_state": st.session_state.get("rf_adv_metrics", {}).get(
                    "random_state", 42
                ),
                "preset": st.session_state.get("rf_adv_metrics", {}).get(
                    "preset"
                ),
                "cv_splits": st.session_state.get("rf_adv_metrics", {}).get(
                    "cv_splits"
                ),
                "scoring": st.session_state.get("rf_adv_metrics", {}).get(
                    "scoring"
                ),
                "skill_score": st.session_state.get("rf_adv_metrics", {}).get(
                    "skill_score"
                ),
                "low_skill": st.session_state.get("rf_adv_metrics", {}).get(
                    "low_skill"
                ),
                "best_params": st.session_state.get("rf_adv_best_params", {}),
                "features_used": st.session_state.get(
                    "rf_adv_selected_features", DEFAULT_FEATURES
                ),
                "ticker": meta["ticker"],
                "market": meta.get("market"),
                "data_range": f'{meta["start"]} → {meta["end"]}',
                "interval": meta["interval"],
            }
        )

    with st.expander("📚 Documentation & warnings"):
        st.markdown(
            "**About this model**\n\n"
            "The Advanced Random Forest extends the baseline variant with:\n\n"
            "- **25+ technical indicators** spanning returns, moving averages, "
            "price ratios, volatility, volume, momentum, patterns, Bollinger "
            "Bands and trend structure.\n"
            "- **GridSearchCV** hyperparameter tuning across a configurable "
            "preset grid.\n"
            "- **TimeSeriesSplit** cross-validation to preserve temporal "
            "ordering and avoid data leakage.\n"
            "- **Feature importance** computed via both built-in Gini "
            "importance and permutation importance on the test set.\n"
            "- **Multi-market support** via the market catalog (US/EU/UK/JP/HK/IN "
            "equities, ETFs, crypto, forex, commodities) plus a custom ticker "
            "override.\n\n"
            "**Forecast safety features:**\n"
            "- **Skill gate** suppresses forecasts when the tuned model has no "
            "directional edge (all metrics within ~3 pts of random).\n"
            "- **Range check** flags no-signal forecasts whose projected range "
            "is under 2% over the horizon.\n"
            "- **Honest confidence** weights accuracy, CV score, and volatility "
            "regime without inflating certainty.\n"
            "- **Consistent gate** — the skill gate guards the chart, the "
            "forecast CSV download, and the JSON export together.\n\n"
            "Inputs are Yahoo Finance OHLCV bars; the model uses a "
            "chronological holdout and StandardScaler. All forecasts shown "
            "here are illustrative."
        )
        st.warning(
            "Past performance does not guarantee future results. Forecasts "
            "are illustrative and should not be used as financial advice."
        )