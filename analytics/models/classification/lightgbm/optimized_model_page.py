"""Interactive LightGBM Optimized Model workbench.

Drop-in replacement for analytics/models/classification/lightgbm/optimized_model_page.py.
Uses the existing navigation and JSON export helpers. Training is explicit,
chronological, with independent session state for this variant.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from market_ai.analytics.models.variant import render_model_navigation
from market_ai.analytics.export import build_envelope, render_download_button
from market_ai.analytics.models.classification.random_forest.pipeline import (
    RandomForestBaseModel, to_chart_frame,
)
from market_ai.utils.theme import COLORS

# ============================================================
# CONFIGURATION AND STATE
# ============================================================
VARIANT = "Optimized Model"
DESCRIPTION = "LightGBM parameter search over expanding time-series validation folds. The final holdout is reserved for evaluation; optimized does not guarantee better performance."
PREFIX = "lgb_optimized_"
FEATURES = ["Return", "Price_MA5_Ratio", "Price_MA20_Ratio", "Price_MA50_Ratio",
            "Momentum_5", "Momentum_10", "Volatility_20", "RSI_14",
            "High_Low_Ratio", "Close_Open_Ratio", "Volume_Ratio"]
DEFAULT_FEATURES = ["Return", "Price_MA20_Ratio", "Momentum_5", "Volatility_20",
                    "RSI_14", "High_Low_Ratio", "Close_Open_Ratio"]
TARGETS = {"Next-session direction": (1, 0.0),
           "5-session direction": (5, 0.0),
           "Next-session return > +0.5%": (1, 0.005)}


def _reset_results() -> None:
    st.session_state.pop(PREFIX + "run", None)


def _signature(data: pd.DataFrame, config: dict) -> str:
    digest = hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values.tobytes())
    digest.update(json.dumps(config, sort_keys=True).encode())
    return digest.hexdigest()


# ============================================================
# DATA AND FEATURES
# ============================================================
def _download_market_data(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Reuse the app's Yahoo loader; daily bars avoid its intraday normalization."""
    model = RandomForestBaseModel(ticker=ticker)
    frame = to_chart_frame(model.load_data(start_date=start, end_date=end, interval="1d"))
    return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build trailing-only indicators without dropping rows or creating labels."""
    frame = data.copy().reset_index(drop=True)
    # Consistent raw OHLC price basis; do not mix adjusted close with raw open.
    close = pd.to_numeric(frame["close"], errors="coerce")
    frame["Return"] = close.pct_change(fill_method=None)
    for window in (5, 20, 50):
        frame[f"Price_MA{window}_Ratio"] = close / close.rolling(window).mean()
    frame["Momentum_5"] = close / close.shift(5) - 1
    frame["Momentum_10"] = close / close.shift(10) - 1
    frame["Volatility_20"] = frame["Return"].rolling(20).std()
    change = close.diff()
    gain = change.clip(lower=0).rolling(14).mean()
    loss = (-change.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100)
    frame["RSI_14"] = rsi.mask((loss == 0) & (gain == 0), 50)
    frame["High_Low_Ratio"] = frame["high"] / frame["low"].replace(0, np.nan)
    frame["Close_Open_Ratio"] = close / frame["open"].replace(0, np.nan)
    frame["Volume_Ratio"] = frame["volume"] / frame["volume"].rolling(20).mean().replace(0, np.nan)
    return frame.replace([np.inf, -np.inf], np.nan)


def _prepare_split(data: pd.DataFrame, features: list[str], target: str,
                   test_fraction: float) -> tuple:
    """Purge labels touching the test period; exclude unknown future outcomes."""
    frame = _engineer_features(data)
    horizon, threshold = TARGETS[target]
    future = frame["close"].shift(-horizon) / frame["close"] - 1
    frame["forward_return"] = future
    frame["label_end"] = np.arange(len(frame)) + horizon
    frame["target"] = (future > threshold).astype(int)
    valid = frame.dropna(subset=features + ["forward_return"])
    split = int(len(valid) * (1 - test_fraction))
    if split < 1 or split >= len(valid):
        raise ValueError("Insufficient usable rows for a chronological split.")
    test = valid.iloc[split:]
    train_candidates = valid.iloc[:split]
    train = train_candidates.loc[train_candidates["label_end"] < test.index[0]]
    if len(train) < 50 or len(test) < 20:
        raise ValueError("Need at least 50 training and 20 test rows after feature cleanup and the gap. Download a longer history or select fewer features.")
    if train["target"].nunique() < 2:
        raise ValueError("Training data contains only one target class. Use more history or a different target.")
    return frame, valid, train, test, len(train_candidates) - len(train)


# ============================================================
# FIT AND EVALUATE
# ============================================================
def _fit_model(data: pd.DataFrame, config: dict) -> dict:
    """Fit LightGBM with variant-specific selection restricted to training data."""
    features = config["features"]
    frame, valid, train, test, gap = _prepare_split(
        data, features, config["target"], config["test_fraction"])
    try:
        from lightgbm import LGBMClassifier, early_stopping, record_evaluation
    except ImportError as error:
        raise ValueError("Install LightGBM: python -m pip install 'lightgbm>=4.0'") from error
    from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
    from sklearn.metrics import log_loss
    features = config["features"]
    started = time.perf_counter()
    base_params = dict(
        objective="binary", metric="binary_logloss", importance_type="gain",
        num_leaves=config["num_leaves"], subsample_freq=1, deterministic=True, force_col_wise=True,
        n_estimators=config["n_estimators"], max_depth=config["max_depth"],
        learning_rate=config["learning_rate"], min_child_samples=config["min_child_samples"],
        subsample=config["subsample"], colsample_bytree=config["colsample_bytree"],
        reg_alpha=config["reg_alpha"], reg_lambda=config["reg_lambda"],
        n_jobs=config["n_jobs"], random_state=config["seed"], verbosity=-1,
    )
    def create_model(rows: pd.DataFrame, params: dict):
        if rows["target"].nunique() != 2:
            raise ValueError("An internal training window has only one class. Download more history or change the target.")
        counts = rows["target"].value_counts()
        weight = float(counts[0] / counts[1]) if config["balance"] else 1.0
        params = dict(params)
        if params["max_depth"] > 0:
            params["num_leaves"] = min(params["num_leaves"], 2 ** params["max_depth"])
        return LGBMClassifier(**params, scale_pos_weight=weight)

    diagnostics = {"variant": VARIANT, "selection_method": "User configuration", "cv_results": []}
    if VARIANT == "Advanced Model":
        # Early stopping uses a purged validation tail inside training, never test.
        split = int(len(train) * 0.8)
        validation = train.iloc[split:]
        inner = train.iloc[:split]
        inner = inner.loc[inner["label_end"] < validation.index[0]]
        if len(inner) < 40 or len(validation) < 15:
            raise ValueError("Advanced early stopping needs more training history.")
        history = {}
        probe = create_model(inner, base_params)
        probe.fit(inner[features], inner["target"],
                  eval_set=[(validation[features], validation["target"])],
                  callbacks=[early_stopping(config["patience"]), record_evaluation(history)])
        base_params["n_estimators"] = int(probe.best_iteration_)
        diagnostics.update(selection_method="Purged internal validation + early stopping, then refit on outer training",
                           selected_trees=base_params["n_estimators"], validation_logloss=float(probe.best_score_["valid_0"]["binary_logloss"]),
                           inner_train_end=str(inner["date"].iloc[-1]), validation_start=str(validation["date"].iloc[0]),
                           internal_purged_rows=split - len(inner), learning_curve=history)
    elif VARIANT == "Optimized Model":
        horizon = TARGETS[config["target"]][0]
        folds = list(TimeSeriesSplit(n_splits=config["folds"], gap=horizon).split(train))
        candidates = list(ParameterSampler({
            "n_estimators": [75, 150, 250], "max_depth": [2, 3, 5],
            "learning_rate": [0.03, 0.07, 0.1], "num_leaves": [7, 15, 31], "min_child_samples": [10, 20, 40],
            "subsample": [0.7, 0.9, 1.0], "colsample_bytree": [0.7, 0.9, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0], "reg_lambda": [1.0, 5.0, 10.0],
        }, n_iter=config["trials"], random_state=config["seed"]))
        best_loss = float("inf")
        selected = None
        audit = []
        for candidate_id, candidate in enumerate(candidates, 1):
            losses = []
            for fold_id, (train_ids, val_ids) in enumerate(folds, 1):
                fold_train, fold_val = train.iloc[train_ids], train.iloc[val_ids]
                if len(fold_train) < 30 or fold_train["label_end"].max() >= fold_val.index[0]:
                    raise ValueError("Insufficient history for safe time-series folds. Reduce folds or download more data.")
                model = create_model(fold_train, {**base_params, **candidate})
                model.fit(fold_train[features], fold_train["target"])
                probability = model.predict_proba(fold_val[features])[:, 1]
                losses.append(float(log_loss(fold_val["target"], probability, labels=[0, 1])))
                if candidate_id == 1:
                    audit.append({"fold": fold_id, "train_end": str(fold_train["date"].iloc[-1]),
                                  "validation_start": str(fold_val["date"].iloc[0])})
            mean_loss = float(np.mean(losses))
            diagnostics["cv_results"].append({"candidate": candidate_id, "mean_logloss": mean_loss,
                                               "std_logloss": float(np.std(losses)), "parameters": candidate})
            if mean_loss < best_loss:
                best_loss, selected = mean_loss, candidate
        base_params.update(selected)
        diagnostics.update(selection_method="Randomized search on expanding time-series folds within training only",
                           selected_parameters=selected, cv_logloss=best_loss, folds=audit)
    model = create_model(train, base_params)
    model.fit(train[features], train["target"])
    fit_seconds = time.perf_counter() - started
    diagnostics["final_parameters"] = model.get_params()
    started = time.perf_counter()
    predictions = model.predict(test[features])
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(test[features])[:, positive_index]
    predict_seconds = time.perf_counter() - started
    y = test["target"]
    majority = int(train["target"].mode().iloc[0])
    baseline = float(accuracy_score(y, np.full(len(y), majority)))
    accuracy = float(accuracy_score(y, predictions))
    metrics = {
        "accuracy": accuracy, "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)) if y.nunique() == 2 else None,
        "baseline_accuracy": baseline, "accuracy_lift_pp": (accuracy - baseline) * 100,
        "train_samples": len(train), "test_samples": len(test), "purged_rows": gap,
        "fit_seconds": fit_seconds, "test_prediction_seconds": predict_seconds,
    }
    output = test[["date", "close", "forward_return", "target"]].copy()
    output["prediction"] = predictions
    output["positive_probability"] = probabilities
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    latest = None
    # Predict only the actual last observation, never a silently stale row.
    row = frame.tail(1)
    if not row[features].isna().any(axis=None):
        latest = {"as_of": str(row["date"].iloc[0]),
                  "close": float(row["close"].iloc[0]),
                  "positive_probability": float(model.predict_proba(row[features])[0, positive_index]),
                  "prediction": int(model.predict(row[features])[0])}
    return {"selection": diagnostics, "model": model, "config": dict(config), "metrics": metrics,
            "predictions": output, "importance": importance, "latest": latest,
            "confusion": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
            "report": classification_report(y, predictions, labels=[0, 1],
                target_names=["Not above threshold", "Above threshold"], output_dict=True, zero_division=0),
            "train_end": str(train["date"].iloc[-1]), "test_start": str(test["date"].iloc[0]),
            "signature": _signature(data, config)}


# ============================================================
# PRESENTATION AND EXPORT
# ============================================================
def _data_overview(data: pd.DataFrame) -> None:
    st.subheader("Data overview")
    recent = data.tail(260)
    figure = go.Figure(go.Candlestick(x=recent["date"], open=recent["open"],
        high=recent["high"], low=recent["low"], close=recent["close"], name="OHLC",
        increasing_line_color=COLORS["green"], decreasing_line_color=COLORS["red"]))
    for window, color in [(20, COLORS["cyan"]), (50, COLORS["yellow"])]:
        figure.add_scatter(x=recent["date"], y=data["close"].rolling(window).mean().tail(260),
                          name=f"MA{window}", line={"color": color})
    figure.update_layout(height=350, xaxis_rangeslider_visible=False,
        margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True, key=PREFIX + "price")
    with st.expander("OHLCV data and descriptives"):
        st.dataframe(data.tail(100), hide_index=True, use_container_width=True)
        st.dataframe(data.select_dtypes(include="number").describe(), use_container_width=True)


def _render_results(run: dict, meta: dict, data: pd.DataFrame) -> None:
    metrics = run["metrics"]
    st.subheader("Holdout performance")
    columns = st.columns(5)
    for col, name in zip(columns, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        value = metrics[name]
        col.metric(name.replace("_", " ").upper(), "N/A" if value is None else f"{value:.1%}")
    st.caption("Precision, recall and F1 refer to the positive class: return above the selected threshold. AUC is unavailable for a single-class test window.")
    cols = st.columns(4)
    cols[0].metric("Train-majority baseline", f"{metrics['baseline_accuracy']:.1%}")
    cols[1].metric("Accuracy lift", f"{metrics['accuracy_lift_pp']:+.2f} pp")
    cols[2].metric("Training + selection", f"{metrics['fit_seconds']:.3f} s")
    cols[3].metric("Test rows", metrics["test_samples"])
    st.caption(f"Training ends {run['train_end']} · Test begins {run['test_start']} · {metrics['purged_rows']} boundary rows purged. Baseline class is chosen from training data only.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        st.dataframe(pd.DataFrame(run["confusion"],
            index=["Actual: not above", "Actual: above"],
            columns=["Predicted: not above", "Predicted: above"]), use_container_width=True)
    with right:
        chart = go.Figure(go.Bar(x=run["importance"]["importance"],
            y=run["importance"]["feature"], orientation="h"))
        chart.update_layout(title="Feature importance", height=300, yaxis={"autorange": "reversed"})
        st.plotly_chart(chart, use_container_width=True, key=PREFIX + "importance")
    st.caption("Gain-based feature importance describes this fitted model; it is not a causal effect.")
    with st.expander("Classification report and test predictions"):
        st.json(run["report"])
        st.dataframe(run["predictions"], hide_index=True, use_container_width=True)
    with st.expander("Model selection diagnostics"):
        st.json(run["selection"])
    st.subheader("Latest direction estimate")
    latest = run["latest"]
    if latest is None:
        st.warning("The latest row is missing selected features. No current estimate is shown.")
    else:
        horizon, threshold = TARGETS[run["config"]["target"]]
        st.metric(f"Probability of return > {threshold:.1%} over {horizon} session(s)",
                  f"{latest['positive_probability']:.1%}")
        st.caption(f"As of {latest['as_of']} · Closing price {latest['close']:,.4f} in instrument quote units. Uses the model fitted on the outer training period. Scores are uncalibrated, especially with class weighting enabled. Use completed daily bars only.")
    st.info("This classifier estimates direction. It does not produce a validated multi-day price path or a price confidence interval.")
    envelope = build_envelope(
        workbench="classification", model_name="LGBMClassifier", model_variant=VARIANT,
        request={**meta, **run["config"]}, inputs={"rows": len(data), "source": "Yahoo Finance", "price_basis": "raw OHLC"},
        results={"latest_direction": latest, "confusion_matrix": run["confusion"],
                 "classification_report": run["report"], "test_predictions": run["predictions"]},
        metrics=metrics, diagnostics={"selection": run["selection"], "feature_importances": run["importance"],
            "train_end": run["train_end"], "test_start": run["test_start"], "signature": run["signature"]},
        caveats=["Single chronological holdout; repeated tuning on it makes results optimistic.",
                 "Raw price data may contain corporate-action discontinuities; inspect the data.",
                 "A classification probability is not a price forecast or calibrated confidence interval.",
                 "The latest available bar may be incomplete during an open market session."])
    st.subheader("Export & downloads")
    render_download_button(envelope, key=PREFIX + "analysis_json")
    cols = st.columns(3)
    for col, label, frame, suffix in [
        (cols[0], "Test predictions CSV", run["predictions"], "predictions"),
        (cols[1], "Feature importance CSV", run["importance"], "importance"),
        (cols[2], "Raw data CSV", data, "data")]:
        col.download_button(label, frame.to_csv(index=False), f"lgb_optimized_{suffix}.csv",
                            "text/csv", key=PREFIX + suffix + "_csv")
    with st.expander("Model metadata"):
        st.json({"variant": VARIANT, "parameters": run["model"].get_params(), "data": meta})


# ============================================================
# PUBLIC PAGE ENTRY POINT
# ============================================================
def render_optimized_model() -> None:
    """Render the complete LightGBM Optimized Model page inside the existing Analytics router."""
    render_model_navigation("LightGBM", VARIANT)
    st.caption(f"Analytics / Models / LightGBM / {VARIANT}")
    st.title(f"LightGBM / {VARIANT}")
    st.write(DESCRIPTION)
    if st.button("Reset trained results", key=PREFIX + "reset"):
        _reset_results()
    st.subheader("Market data")
    with st.form(PREFIX + "data_form"):
        cols = st.columns(3)
        ticker = cols[0].text_input("Ticker", "AAPL", key=PREFIX + "ticker").strip().upper()
        start = cols[1].date_input("Start date", date.today() - timedelta(days=1095), key=PREFIX + "start")
        end = cols[2].date_input("End date (exclusive)", date.today(), key=PREFIX + "end")
        cached = st.checkbox("Use matching session-cached data", True, key=PREFIX + "cached")
        download = st.form_submit_button("Download daily data", type="primary")
    request = {"ticker": ticker, "start": str(start), "end": str(end), "interval": "1d"}
    if download:
        if not ticker or start >= end:
            st.error("Enter a ticker and a start date before the end date.")
        elif cached and st.session_state.get(PREFIX + "meta") == request:
            st.success("Using the matching session-cached dataset.")
        else:
            try:
                with st.spinner(f"Downloading {ticker}…"):
                    frame = _download_market_data(ticker, start, end)
                if frame.empty:
                    raise ValueError("No usable data returned.")
            except Exception as error:
                st.error(f"Download failed: {error}. Previously loaded data, if any, remains active.")
            else:
                st.session_state[PREFIX + "data"] = frame
                st.session_state[PREFIX + "meta"] = request
                _reset_results()
    data = st.session_state.get(PREFIX + "data")
    if data is None or data.empty:
        st.info("Download daily market data to begin. Examples: AAPL, MSFT, BTC-USD, EURUSD=X, GC=F.")
        return
    meta = st.session_state[PREFIX + "meta"]
    st.caption(f"Active dataset: {meta['ticker']} · Daily bars · {data['date'].min()} to {data['date'].max()} · {len(data):,} rows. Edit and submit Download to change it.")
    _data_overview(data)
    st.subheader("Features & training")
    features = st.multiselect("Features", FEATURES, default=DEFAULT_FEATURES, key=PREFIX + "features")
    target = st.selectbox("Target", list(TARGETS), key=PREFIX + "target")
    controls = st.columns(3)
    test_fraction = controls[0].slider("Holdout fraction", 0.15, 0.40, 0.25, 0.05, key=PREFIX + "test")
    jobs = controls[1].selectbox("CPU workers", [1, 2, 4], key=PREFIX + "jobs")
    seed = controls[2].number_input("Random seed", 0, 99999, 42, key=PREFIX + "seed")
    balance = st.checkbox("Weight positive class using training class counts", False, key=PREFIX + "balance")
    config = {"features": list(features), "target": target, "test_fraction": test_fraction,
              "n_jobs": jobs, "seed": int(seed), "balance": balance,
              "n_estimators": 100, "max_depth": 3, "learning_rate": 0.1,
              "num_leaves": 7, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0,
              "reg_alpha": 0.0, "reg_lambda": 1.0}
    if VARIANT == "Optimized Model":
        cols = st.columns(2)
        config["trials"] = cols[0].slider("Search candidates", 3, 12, 6, key=PREFIX + "trials")
        config["folds"] = cols[1].slider("Time-series validation folds", 2, 5, 3, key=PREFIX + "folds")
        st.caption(f"{config['trials'] * config['folds']} internal fits + 1 final fit. Candidates vary trees, depth, leaves, minimum leaf rows, learning rate, sampling and regularization. Selection minimizes validation log loss; the final test period is excluded from the search.")
    else:
        cols = st.columns(3)
        config["n_estimators"] = cols[0].slider("Maximum trees" if VARIANT == "Advanced Model" else "Trees", 50, 800, 400 if VARIANT == "Advanced Model" else 100, 50, key=PREFIX + "trees")
        config["max_depth"] = cols[1].slider("Maximum depth", 2, 10, 4 if VARIANT == "Advanced Model" else 3, key=PREFIX + "depth")
        config["learning_rate"] = cols[2].number_input("Learning rate", 0.01, 0.3, 0.05 if VARIANT == "Advanced Model" else 0.1, 0.01, key=PREFIX + "rate")
        config["num_leaves"] = st.slider("Maximum leaves per tree", 2, 63, 15 if VARIANT == "Advanced Model" else 7, key=PREFIX + "leaves")
        if VARIANT == "Advanced Model":
            cols = st.columns(3)
            config["min_child_samples"] = cols[0].slider("Minimum rows per leaf", 5, 100, 20, 5, key=PREFIX + "child")
            config["subsample"] = cols[1].slider("Row sampling", 0.5, 1.0, 0.8, 0.05, key=PREFIX + "rows")
            config["colsample_bytree"] = cols[2].slider("Feature sampling", 0.5, 1.0, 0.8, 0.05, key=PREFIX + "columns")
            cols = st.columns(3)
            config["reg_alpha"] = cols[0].number_input("L1 regularization", 0.0, 20.0, 0.1, 0.1, key=PREFIX + "alpha")
            config["reg_lambda"] = cols[1].number_input("L2 regularization", 0.0, 30.0, 5.0, 0.5, key=PREFIX + "lambda")
            config["patience"] = cols[2].slider("Early-stopping patience", 5, 50, 20, 5, key=PREFIX + "patience")
    st.caption("Leaf count is capped at 2 to the power of maximum depth. Row sampling is enabled when its fraction is below 1.")
    st.caption("Daily bars, trailing indicators, chronological test holdout and a purged boundary. Positive class means return above your target threshold. No future test labels are used for training, early stopping or parameter selection.")
    if features:
        with st.expander("Feature preview"):
            engineered = _engineer_features(data)[features].dropna()
            st.dataframe(engineered.tail(10), use_container_width=True)
            st.caption(f"{len(engineered):,} feature-complete rows before target cleanup and splitting.")
    if st.button("Run LightGBM", type="primary", key=PREFIX + "train"):
        _reset_results()
        if len(features) < 2:
            st.error("Select at least two features.")
        else:
            try:
                with st.spinner("Training LightGBM and evaluating the chronological holdout…"):
                    st.session_state[PREFIX + "run"] = _fit_model(data, config)
            except Exception as error:
                st.error(f"Training failed: {error}")
    run = st.session_state.get(PREFIX + "run")
    if run:
        if run["signature"] != _signature(data, config):
            st.warning("Settings have changed. Run the model again to display matching results.")
        else:
            _render_results(run, meta, data)


__all__ = ["render_optimized_model"]
