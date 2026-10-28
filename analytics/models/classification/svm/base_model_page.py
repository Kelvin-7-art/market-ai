"""Interactive SVM Base Model workbench.

Drop-in replacement for analytics/models/classification/svm/base_model_page.py.
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
VARIANT = "Base Model"
DESCRIPTION = "Scaled SVM baseline with configurable linear or RBF kernels."
PREFIX = "svm_base_"
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
    """Fit SVM with variant-specific selection restricted to training data."""
    features = config["features"]
    frame, valid, train, test, gap = _prepare_split(
        data, features, config["target"], config["test_fraction"])
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import ParameterGrid, ParameterSampler, TimeSeriesSplit
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.inspection import permutation_importance
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    started = time.perf_counter()
    base_params = dict(C=config["C"], kernel=config["kernel"], gamma=config["gamma"],
                       class_weight="balanced" if config["balance"] else None,
                       cache_size=200, max_iter=50000, tol=0.001)

    def fit_rows(rows: pd.DataFrame, params: dict):
        if rows["target"].nunique() != 2:
            raise ValueError("An internal training window has only one class. Use more history or a different target.")
        model = Pipeline([("scaler", StandardScaler()), ("svc", SVC(**params))])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                model.fit(rows[features], rows["target"])
            except ConvergenceWarning as error:
                raise ValueError("SVM did not converge within 50,000 iterations. Lower C or use a shorter history.") from error
        return model

    diagnostics = {"variant": VARIANT, "selection_method": "User configuration", "cv_results": []}
    if VARIANT != "Base Model":
        if VARIANT == "Tuned Model":
            split = int(len(train) * 0.8)
            validation = train.iloc[split:]
            inner = train.iloc[:split]
            inner = inner.loc[inner["label_end"] < validation.index[0]]
            folds = [(inner, validation)]
            candidates = list(ParameterGrid({"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.01, 0.1], "kernel": ["rbf"]}))
            method = "Nine-candidate RBF grid on purged internal validation"
        else:
            horizon = TARGETS[config["target"]][0]
            folds = [(train.iloc[a], train.iloc[b]) for a, b in TimeSeriesSplit(
                n_splits=config["folds"], gap=horizon).split(train)]
            candidates = list(ParameterSampler([
                {"kernel": ["linear"], "C": [0.01, 0.1, 1.0, 10.0]},
                {"kernel": ["rbf"], "C": [0.01, 0.1, 1.0, 10.0, 100.0], "gamma": ["scale", 0.001, 0.01, 0.1, 1.0]},
            ], n_iter=config["trials"], random_state=config["seed"]))
            method = "Randomized search on expanding time-series folds"
        audit = []
        for fold_train, fold_val in folds:
            if len(fold_train) < 30 or len(fold_val) < 15 or fold_train["label_end"].max() >= fold_val.index[0]:
                raise ValueError("Insufficient history for safe validation. Download more data or reduce folds.")
            if fold_val["target"].nunique() < 2:
                raise ValueError("A validation window has only one class. Use more history or another target for balanced-accuracy tuning.")
            audit.append({"train_end": str(fold_train["date"].iloc[-1]), "validation_start": str(fold_val["date"].iloc[0])})
        best_score, selected = -float("inf"), None
        for candidate_id, candidate in enumerate(candidates, 1):
            scores = []
            for fold_train, fold_val in folds:
                probe = fit_rows(fold_train, {**base_params, **candidate})
                scores.append(float(balanced_accuracy_score(fold_val["target"], probe.predict(fold_val[features]))))
            score = float(np.mean(scores))
            diagnostics["cv_results"].append({"candidate": candidate_id,
                "mean_balanced_accuracy": score, "std_balanced_accuracy": float(np.std(scores)), "parameters": candidate})
            if score > best_score:
                best_score, selected = score, candidate
        base_params.update(selected)
        diagnostics.update(selection_method=method, selected_parameters=selected,
                           validation_balanced_accuracy=best_score, folds=audit)
    model = fit_rows(train, base_params)
    fit_seconds = time.perf_counter() - started
    diagnostics["final_parameters"] = model.named_steps["svc"].get_params()
    diagnostics["support_vectors"] = model.named_steps["svc"].n_support_.tolist()
    diagnostics["scaling"] = "StandardScaler fitted within each training window"
    started = time.perf_counter()
    predictions = model.predict(test[features])
    scores = model.decision_function(test[features])
    predict_seconds = time.perf_counter() - started
    y = test["target"]
    majority = int(train["target"].mode().iloc[0])
    baseline = float(accuracy_score(y, np.full(len(y), majority)))
    accuracy = float(accuracy_score(y, predictions))
    metrics = {
        "accuracy": accuracy, "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)) if y.nunique() == 2 else None,
        "baseline_accuracy": baseline, "accuracy_lift_pp": (accuracy - baseline) * 100,
        "train_samples": len(train), "test_samples": len(test), "purged_rows": gap,
        "fit_seconds": fit_seconds, "test_prediction_seconds": predict_seconds,
    }
    output = test[["date", "close", "forward_return", "target"]].copy()
    output["prediction"] = predictions
    output["decision_score"] = scores
    permutation = permutation_importance(model, test[features], y, scoring="accuracy", n_repeats=5, random_state=config["seed"], n_jobs=1)
    importance = pd.DataFrame({"feature": features, "importance": permutation.importances_mean, "std": permutation.importances_std}).sort_values("importance", ascending=False)
    latest = None
    # Predict only the actual last observation, never a silently stale row.
    row = frame.tail(1)
    if not row[features].isna().any(axis=None):
        latest = {"as_of": str(row["date"].iloc[0]),
                  "close": float(row["close"].iloc[0]),
                  "decision_score": float(model.decision_function(row[features])[0]),
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
    st.caption("Permutation importance: mean loss of holdout accuracy after shuffling each feature (5 repeats). Negative values are possible. This is descriptive, not causal, and is never used for model selection.")
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
        st.metric("Predicted outcome", "Above threshold" if latest["prediction"] == 1 else "Not above threshold")
        st.metric("SVM decision score", f"{latest['decision_score']:+.3f}")
        st.caption(f"Target: return > {threshold:.1%} over {horizon} session(s). Positive scores favor the positive class; magnitude is not a probability. As of {latest['as_of']} · Close {latest['close']:,.4f}. Uses the model fitted on the outer training period and completed daily bars.")
    st.info("This classifier estimates direction. It does not produce a validated multi-day price path or a price confidence interval.")
    envelope = build_envelope(
        workbench="classification", model_name="SVC", model_variant=VARIANT,
        request={**meta, **run["config"]}, inputs={"rows": len(data), "source": "Yahoo Finance", "price_basis": "raw OHLC"},
        results={"latest_direction": latest, "confusion_matrix": run["confusion"],
                 "classification_report": run["report"], "test_predictions": run["predictions"]},
        metrics=metrics, diagnostics={"selection": run["selection"], "feature_importances": run["importance"],
            "train_end": run["train_end"], "test_start": run["test_start"], "signature": run["signature"]},
        caveats=["Single chronological holdout; repeated tuning on it makes results optimistic.",
                 "Raw price data may contain corporate-action discontinuities; inspect the data.",
                 "An SVM decision score is not a probability, price forecast or confidence interval.",
                 "The latest available bar may be incomplete during an open market session."])
    st.subheader("Export & downloads")
    render_download_button(envelope, key=PREFIX + "analysis_json")
    cols = st.columns(3)
    for col, label, frame, suffix in [
        (cols[0], "Test predictions CSV", run["predictions"], "predictions"),
        (cols[1], "Feature importance CSV", run["importance"], "importance"),
        (cols[2], "Raw data CSV", data, "data")]:
        col.download_button(label, frame.to_csv(index=False), f"svm_base_{suffix}.csv",
                            "text/csv", key=PREFIX + suffix + "_csv")
    with st.expander("Model metadata"):
        st.json({"variant": VARIANT, "parameters": run["selection"]["final_parameters"], "data": meta})


# ============================================================
# PUBLIC PAGE ENTRY POINT
# ============================================================
def render_base_model() -> None:
    """Render the complete SVM Base Model page inside the existing Analytics router."""
    render_model_navigation("SVM", VARIANT)
    st.caption(f"Analytics / Models / SVM / {VARIANT}")
    st.title(f"SVM / {VARIANT}")
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
    controls = st.columns(2)
    test_fraction = controls[0].slider("Holdout fraction", 0.15, 0.40, 0.25, 0.05, key=PREFIX + "test")
    seed = controls[1].number_input("Random seed", 0, 99999, 42, key=PREFIX + "seed")
    balance = st.checkbox("Balance classes using training class counts", False, key=PREFIX + "balance")
    config = {"features": list(features), "target": target, "test_fraction": test_fraction,
              "seed": int(seed), "balance": balance, "C": 1.0, "kernel": "rbf", "gamma": "scale"}
    if VARIANT == "Base Model":
        cols = st.columns(3)
        config["kernel"] = cols[0].selectbox("Kernel", ["rbf", "linear"], key=PREFIX + "kernel")
        config["C"] = cols[1].select_slider("C (larger means less regularization)", [0.01, 0.1, 1.0, 10.0, 100.0], value=1.0, key=PREFIX + "c")
        if config["kernel"] == "rbf":
            config["gamma"] = cols[2].selectbox("Gamma", ["scale", "auto", 0.001, 0.01, 0.1, 1.0], key=PREFIX + "gamma")
    elif VARIANT == "Tuned Model":
        st.caption("Nine RBF candidates: C = 0.1, 1, 10 and gamma = scale, 0.01, 0.1. Selection uses a purged validation tail inside training, then refits on the full training period.")
    else:
        cols = st.columns(2)
        config["trials"] = cols[0].slider("Search candidates", 3, 20, 8, key=PREFIX + "trials")
        config["folds"] = cols[1].slider("Time-series validation folds", 2, 5, 3, key=PREFIX + "folds")
        st.caption(f"{config['trials'] * config['folds']} internal fits + 1 final fit. Search compares linear and RBF kernels using mean validation balanced accuracy.")
    st.caption("Scaling is fitted only on each training window. Daily bars, trailing indicators, chronological holdout and a purged boundary. SVM outputs decision scores, not probabilities. Large histories can be slow; fits run sequentially with a 200 MB kernel cache and a convergence limit.")
    if features:
        with st.expander("Feature preview"):
            engineered = _engineer_features(data)[features].dropna()
            st.dataframe(engineered.tail(10), use_container_width=True)
            st.caption(f"{len(engineered):,} feature-complete rows before target cleanup and splitting.")
    if st.button("Run SVM", type="primary", key=PREFIX + "train"):
        _reset_results()
        if len(features) < 2:
            st.error("Select at least two features.")
        else:
            try:
                with st.spinner("Training SVM and evaluating the chronological holdout…"):
                    st.session_state[PREFIX + "run"] = _fit_model(data, config)
            except Exception as error:
                st.error(f"Training failed: {error}")
    run = st.session_state.get(PREFIX + "run")
    if run:
        if run["signature"] != _signature(data, config):
            st.warning("Settings have changed. Run the model again to display matching results.")
        else:
            _render_results(run, meta, data)


__all__ = ["render_base_model"]
