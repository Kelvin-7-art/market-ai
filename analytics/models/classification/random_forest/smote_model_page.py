"""Interactive Random Forest With SMOTE workbench.

Drop-in replacement for analytics/models/classification/random_forest/smote_model_page.py.
Uses the existing navigation and JSON export helpers. Training is explicit,
chronological, and isolated from the Base Model's session state.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import (
    balanced_accuracy_score, accuracy_score, classification_report, confusion_matrix, f1_score,
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
PREFIX = "rf_smote_"
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
# TRAINING-ONLY RESAMPLING
# ============================================================
def _resample_training(X: np.ndarray, y: pd.Series, ratio: float,
                       neighbors: int, seed: int) -> tuple:
    """Balance training vectors only; adapt k to the available minority rows."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as error:
        raise ValueError("Install the SMOTE dependency: python -m pip install imbalanced-learn") from error
    counts = y.value_counts().sort_index()
    if len(counts) != 2:
        raise ValueError("SMOTE requires both target classes in training data.")
    minority_count = int(counts.min())
    majority_count = int(counts.max())
    before = {str(int(label)): int(count) for label, count in counts.items()}
    desired = int(majority_count * ratio)
    details = {"counts_before": before, "requested_ratio": ratio,
               "requested_neighbors": neighbors, "effective_neighbors": None,
               "applied": False, "synthetic_rows": 0}
    if desired <= minority_count:
        details.update(counts_after=before.copy(),
                       note="SMOTE skipped: training balance already meets the requested ratio.")
        return X, y.to_numpy(), details
    if minority_count < 2:
        raise ValueError("SMOTE needs at least two minority-class training rows. Use more history or a different target.")
    effective_k = min(int(neighbors), minority_count - 1)
    minority_label = counts.idxmin()
    sampler = SMOTE(sampling_strategy={minority_label: desired},
                    k_neighbors=effective_k, random_state=seed)
    X_resampled, y_resampled = sampler.fit_resample(X, y.to_numpy())
    after = {str(int(label)): int(count) for label, count in pd.Series(y_resampled).value_counts().sort_index().items()}
    details.update(counts_after=after, effective_neighbors=effective_k,
                   applied=True, synthetic_rows=len(y_resampled) - len(y),
                   note=f"SMOTE applied to training rows only with k={effective_k}.")
    return X_resampled, y_resampled, details


# ============================================================
# FIT AND EVALUATE
# ============================================================
def _fit_smote_model(data: pd.DataFrame, config: dict) -> dict:
    """Scale and resample training data, then score the untouched holdout."""
    features = config["features"]
    frame, valid, train, test, gap = _prepare_split(
        data, features, config["target"], config["test_fraction"])
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"], max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"], max_features="sqrt",
        max_samples=0.8, bootstrap=True, n_jobs=config["n_jobs"],
        class_weight=config["class_weight"], random_state=config["seed"],
    )
    preparation_started = time.perf_counter()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    X_resampled, y_resampled, smote = _resample_training(
        X_train, train["target"], config["sampling_ratio"], config["smote_neighbors"], config["seed"])
    preparation_seconds = time.perf_counter() - preparation_started
    started = time.perf_counter()
    model.fit(X_resampled, y_resampled)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = model.predict(X_test)
    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, positive_index]
    predict_seconds = time.perf_counter() - started
    y = test["target"]
    majority = int(train["target"].mode().iloc[0])
    baseline = float(accuracy_score(y, np.full(len(y), majority)))
    accuracy = float(accuracy_score(y, predictions))
    # Compare against the same forest settings without synthetic samples.
    comparison_model = clone(model)
    comparison_model.fit(X_train, train["target"])
    comparison_prediction = comparison_model.predict(X_test)
    comparison = {
        "accuracy": float(accuracy_score(y, comparison_prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, comparison_prediction)),
        "f1": float(f1_score(y, comparison_prediction, zero_division=0)),
    }
    metrics = {
        "accuracy": accuracy, "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "resampled_train_samples": len(y_resampled), "preparation_seconds": preparation_seconds, "precision": float(precision_score(y, predictions, zero_division=0)),
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
                  "positive_probability": float(model.predict_proba(scaler.transform(row[features]))[0, positive_index]),
                  "prediction": int(model.predict(scaler.transform(row[features]))[0])}
    return {"model": model, "scaler": scaler, "smote": smote, "comparison": comparison, "config": dict(config), "metrics": metrics,
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
    st.subheader("Training class balance")
    smote = run["smote"]
    st.info(smote["note"])
    balance = pd.DataFrame({"Original training": smote["counts_before"],
                            "After SMOTE": smote["counts_after"]})
    balance.index = ["Class " + str(label) for label in balance.index]
    st.dataframe(balance, use_container_width=True)
    st.caption(f"{smote['synthetic_rows']:,} synthetic rows · Class 1 = above threshold; class 0 = not above. The test set is never resampled.")
    st.subheader("Holdout performance")
    columns = st.columns(5)
    for col, name in zip(columns, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        value = metrics[name]
        col.metric(name.replace("_", " ").upper(), "N/A" if value is None else f"{value:.1%}")
    st.caption("Precision, recall and F1 refer to the positive class: return above the selected threshold. AUC is unavailable for a single-class test window.")
    cols = st.columns(4)
    cols[0].metric("Train-majority baseline", f"{metrics['baseline_accuracy']:.1%}")
    cols[1].metric("Accuracy lift", f"{metrics['accuracy_lift_pp']:+.2f} pp")
    cols[2].metric("Forest fit time", f"{metrics['fit_seconds']:.3f} s")
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
    st.markdown("**Same-split comparison: with and without SMOTE**")
    comparison = pd.DataFrame({
        "Without SMOTE": run["comparison"],
        "With SMOTE": {key: metrics[key] for key in run["comparison"]},
    })
    st.dataframe(comparison.style.format("{:.1%}"), use_container_width=True)
    st.caption("Both forests use the same settings and original training/test boundaries. SMOTE may improve or worsen results; this comparison does not establish future performance.")
    st.caption("Impurity-based feature importance describes this fitted model; it is not a causal effect.")
    with st.expander("Classification report and test predictions"):
        st.json(run["report"])
        st.dataframe(run["predictions"], hide_index=True, use_container_width=True)
    st.subheader("Latest direction estimate")
    latest = run["latest"]
    if latest is None:
        st.warning("The latest row is missing selected features. No current estimate is shown.")
    else:
        horizon, threshold = TARGETS[run["config"]["target"]]
        st.metric(f"Uncalibrated model score: return > {threshold:.1%} over {horizon} session(s)",
                  f"{latest['positive_probability']:.1%}")
        st.caption(f"As of {latest['as_of']} · Closing price {latest['close']:,.4f} in instrument quote units. Uses the holdout-trained model, not a new full-history fit. SMOTE changes class frequencies; these scores are not calibrated real-market probabilities. Use completed daily bars only.")
    st.info("This classifier estimates direction. It does not produce a validated multi-day price path or a price confidence interval.")
    envelope = build_envelope(
        workbench="classification", model_name="RandomForestClassifier", model_variant="With SMOTE",
        request={**meta, **run["config"]}, inputs={"rows": len(data), "source": "Yahoo Finance", "price_basis": "raw OHLC"},
        results={"latest_direction": latest, "confusion_matrix": run["confusion"],
                 "classification_report": run["report"], "test_predictions": run["predictions"]},
        metrics=metrics, diagnostics={"smote": run["smote"], "without_smote": run["comparison"], "feature_importances": run["importance"],
            "train_end": run["train_end"], "test_start": run["test_start"], "signature": run["signature"]},
        caveats=["SMOTE interpolates indicator vectors and does not preserve market regimes or indicator constraints.",
                 "Scaling and SMOTE use original training data only. Test rows retain their natural class distribution.",
                 "Oversampling changes class frequencies; predicted probabilities are not calibrated.",
                 "Single chronological holdout; repeated tuning on it makes results optimistic.",
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
        col.download_button(label, frame.to_csv(index=False), f"rf_smote_{suffix}.csv",
                            "text/csv", key=PREFIX + suffix + "_csv")
    with st.expander("Model metadata"):
        st.json({"variant": "With SMOTE", "parameters": run["model"].get_params(), "data": meta})


# ============================================================
# PUBLIC PAGE ENTRY POINT
# ============================================================
def render_smote_model() -> None:
    """Render the complete With SMOTE page inside the existing Analytics router."""
    render_model_navigation("Random Forest", "With SMOTE")
    st.caption("Analytics / Models / Random Forest / With SMOTE")
    st.title("Random Forest / With SMOTE")
    st.write("Random Forest with training-only Synthetic Minority Over-sampling Technique (SMOTE), chronological holdout evaluation, and a comparison against the same forest without SMOTE.")
    st.caption("SMOTE creates synthetic feature vectors, not additional real market observations. It can mix historical regimes; assess its benefit on the untouched test period.")
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
    cols = st.columns(4)
    trees = cols[0].slider("Trees", 25, 300, 100, 25, key=PREFIX + "trees")
    depth = cols[1].slider("Maximum depth", 3, 20, 10, key=PREFIX + "depth")
    leaf = cols[2].slider("Minimum samples per leaf", 1, 20, 5, key=PREFIX + "leaf")
    test_fraction = cols[3].slider("Holdout fraction", 0.15, 0.40, 0.25, 0.05, key=PREFIX + "test")
    cols = st.columns(3)
    weight = cols[0].selectbox("Class weight", ["None", "balanced"], key=PREFIX + "weight")
    jobs = cols[1].selectbox("CPU workers", [1, 2, -1], help="-1 uses all available cores; 1 limits shared-host resource use.", key=PREFIX + "jobs")
    seed = cols[2].number_input("Random seed", 0, 99999, 42, key=PREFIX + "seed")
    smote_cols = st.columns(2)
    ratio = smote_cols[0].slider("Target minority / majority ratio", 0.25, 1.0, 1.0, 0.05, key=PREFIX + "ratio")
    neighbors = smote_cols[1].slider("SMOTE neighbors", 1, 10, 5, key=PREFIX + "neighbors")
    st.caption("Ratio 1.0 balances the training classes. If the requested ratio is already met, SMOTE is skipped. Neighbors are reduced automatically for small minority classes. Class weighting defaults to None to avoid automatically combining two balancing methods.")
    config = {"sampling_ratio": ratio, "smote_neighbors": neighbors, "features": list(features), "target": target, "n_estimators": trees,
              "max_depth": depth, "min_samples_leaf": leaf, "test_fraction": test_fraction,
              "class_weight": None if weight == "None" else weight, "n_jobs": jobs, "seed": int(seed)}
    st.caption("Daily sessions are observed rows, including weekends for crypto. Features use information through each bar's close. Trees use 80% bootstrap samples and square-root feature sampling; scaling is fitted on original training rows before SMOTE computes nearest neighbors.")
    if features:
        with st.expander("Feature preview"):
            engineered = _engineer_features(data)[features].dropna()
            st.dataframe(engineered.tail(10), use_container_width=True)
            st.caption(f"{len(engineered):,} feature-complete rows before target cleanup and splitting.")
    if st.button("🌲 Run SMOTE model", type="primary", key=PREFIX + "train"):
        _reset_results()
        if len(features) < 2:
            st.error("Select at least two features.")
        else:
            try:
                with st.spinner("Resampling training data and comparing both forests…"):
                    st.session_state[PREFIX + "run"] = _fit_smote_model(data, config)
            except Exception as error:
                st.error(f"Training failed: {error}")
    run = st.session_state.get(PREFIX + "run")
    if run:
        if run["signature"] != _signature(data, config):
            st.warning("Settings have changed. Run the model again to display matching results.")
        else:
            _render_results(run, meta, data)


__all__ = ["render_smote_model"]
