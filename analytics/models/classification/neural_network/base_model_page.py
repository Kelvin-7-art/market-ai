"""Interactive Neural Network Base Model workbench.

Drop-in replacement for analytics/models/classification/neural_network/base_model_page.py.
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
DESCRIPTION = "A configurable single-hidden-layer neural classifier with training-only feature scaling."
PREFIX = "nn_base_"
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
    """Fit Neural Network with variant-specific selection restricted to training data."""
    features = config["features"]
    frame, valid, train, test, gap = _prepare_split(
        data, features, config["target"], config["test_fraction"])
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
    from sklearn.metrics import log_loss
    from sklearn.inspection import permutation_importance
    from threadpoolctl import threadpool_limits

    started = time.perf_counter()
    params = dict(hidden_layer_sizes=tuple(config["layers"]), activation=config["activation"],
                  alpha=config["alpha"], learning_rate_init=config["learning_rate"],
                  solver="adam", batch_size=32, shuffle=False, early_stopping=False,
                  random_state=config["seed"])

    def fit_rows(rows: pd.DataFrame, options: dict, epochs: int, validation=None):
        if rows["target"].nunique() != 2:
            raise ValueError("A training window has only one class. Download more history or change target.")
        scaler = StandardScaler().fit(rows[features])
        x = scaler.transform(rows[features])
        options = {**options, "batch_size": min(32, len(rows))}
        network = MLPClassifier(**options)
        history = []
        best_loss, best_epoch, stale = float("inf"), 1, 0
        val_x = scaler.transform(validation[features]) if validation is not None else None
        # partial_fit preserves Adam state and runs exactly one epoch per call.
        with threadpool_limits(limits=1):
            for epoch in range(1, epochs + 1):
                network.partial_fit(x, rows["target"], classes=np.array([0, 1]))
                record = {"epoch": epoch, "training_loss": float(network.loss_)}
                if not np.isfinite(network.loss_):
                    raise ValueError("Training became numerically unstable. Reduce the learning rate.")
                if validation is not None:
                    loss = float(log_loss(validation["target"], network.predict_proba(val_x), labels=[0, 1]))
                    record["validation_logloss"] = loss
                    if loss < best_loss - 0.0001:
                        best_loss, best_epoch, stale = loss, epoch, 0
                    else:
                        stale += 1
                history.append(record)
                if validation is not None and stale >= config["patience"]:
                    break
        return Pipeline([("scaler", scaler), ("network", network)]), history, best_epoch, best_loss

    diagnostics = {"variant": VARIANT, "selection_method": "User configuration, fixed epochs", "cv_results": []}
    epochs = config["epochs"]
    if VARIANT == "Deep Model":
        split = int(len(train) * 0.8)
        validation = train.iloc[split:]
        inner = train.iloc[:split]
        inner = inner.loc[inner["label_end"] < validation.index[0]]
        if len(inner) < 40 or len(validation) < 15:
            raise ValueError("Deep Model needs more history for internal validation.")
        _, curve, epochs, loss = fit_rows(inner, params, epochs, validation)
        diagnostics.update(selection_method="Purged internal early stopping, then refit from scratch",
                           selected_epochs=epochs, validation_logloss=loss, validation_curve=curve,
                           inner_train_end=str(inner["date"].iloc[-1]), validation_start=str(validation["date"].iloc[0]))
    elif VARIANT == "Optimized Model":
        horizon = TARGETS[config["target"]][0]
        folds = [(train.iloc[a], train.iloc[b]) for a,b in TimeSeriesSplit(
            n_splits=config["folds"], gap=horizon).split(train)]
        audit = []
        for tr, va in folds:
            if len(tr) < 30 or len(va) < 15 or tr["label_end"].max() >= va.index[0]:
                raise ValueError("Insufficient history for safe time-series folds. Reduce folds or download more data.")
            audit.append({"train_end": str(tr["date"].iloc[-1]), "validation_start": str(va["date"].iloc[0])})
        candidates = list(ParameterSampler({"hidden_layer_sizes": [(32,), (64,32), (64,32,16)],
            "activation": ["relu", "tanh"], "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.0003, 0.001, 0.003]},
            n_iter=config["trials"], random_state=config["seed"]))
        best_loss, selected = float("inf"), None
        for candidate_id, candidate in enumerate(candidates, 1):
            losses = []
            for tr, va in folds:
                probe, _, _, _ = fit_rows(tr, {**params, **candidate}, epochs)
                losses.append(float(log_loss(va["target"], probe.predict_proba(va[features]), labels=[0, 1])))
            score = float(np.mean(losses))
            diagnostics["cv_results"].append({"candidate": candidate_id, "mean_logloss": score,
                "std_logloss": float(np.std(losses)), "parameters": candidate})
            if score < best_loss:
                best_loss, selected = score, candidate
        params.update(selected)
        diagnostics.update(selection_method="Randomized search on expanding time-series folds",
                           selected_parameters=selected, cv_logloss=best_loss, folds=audit)
    model, curve, _, _ = fit_rows(train, params, epochs)
    fit_seconds = time.perf_counter() - started
    diagnostics.update(final_parameters=model.named_steps["network"].get_params(),
                       final_epochs=epochs, training_curve=curve,
                       scaling="StandardScaler fitted within each training window")
    started = time.perf_counter()
    predictions = model.predict(test[features])
    scores = model.predict_proba(test[features])[:, 1]
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
    output["positive_probability"] = scores
    permutation = permutation_importance(model, test[features], y, scoring="accuracy", n_repeats=5, random_state=config["seed"], n_jobs=1)
    importance = pd.DataFrame({"feature": features, "importance": permutation.importances_mean, "std": permutation.importances_std}).sort_values("importance", ascending=False)
    latest = None
    # Predict only the actual last observation, never a silently stale row.
    row = frame.tail(1)
    if not row[features].isna().any(axis=None):
        latest = {"as_of": str(row["date"].iloc[0]),
                  "close": float(row["close"].iloc[0]),
                  "positive_probability": float(model.predict_proba(row[features])[0, 1]),
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
    with st.expander("Training loss curve"):
        st.line_chart(pd.DataFrame(run["selection"]["training_curve"]).set_index("epoch"))
    with st.expander("Model selection diagnostics"):
        st.json(run["selection"])
    st.subheader("Latest direction estimate")
    latest = run["latest"]
    if latest is None:
        st.warning("The latest row is missing selected features. No current estimate is shown.")
    else:
        horizon, threshold = TARGETS[run["config"]["target"]]
        st.metric("Predicted outcome", "Above threshold" if latest["prediction"] == 1 else "Not above threshold")
        st.metric("Positive-class probability", f"{latest['positive_probability']:.1%}")
        st.caption(f"Target: return > {threshold:.1%} over {horizon} session(s). Probabilities are uncalibrated model estimates. As of {latest['as_of']} · Close {latest['close']:,.4f}. Uses the model fitted on the outer training period and completed daily bars.")
    st.info("This classifier estimates direction. It does not produce a validated multi-day price path or a price confidence interval.")
    envelope = build_envelope(
        workbench="classification", model_name="MLPClassifier", model_variant=VARIANT,
        request={**meta, **run["config"]}, inputs={"rows": len(data), "source": "Yahoo Finance", "price_basis": "raw OHLC"},
        results={"latest_direction": latest, "confusion_matrix": run["confusion"],
                 "classification_report": run["report"], "test_predictions": run["predictions"]},
        metrics=metrics, diagnostics={"selection": run["selection"], "feature_importances": run["importance"],
            "train_end": run["train_end"], "test_start": run["test_start"], "signature": run["signature"]},
        caveats=["Single chronological holdout; repeated tuning on it makes results optimistic.",
                 "Raw price data may contain corporate-action discontinuities; inspect the data.",
                 "A model probability is not a price forecast or calibrated confidence interval.",
                 "The latest available bar may be incomplete during an open market session."])
    st.subheader("Export & downloads")
    render_download_button(envelope, key=PREFIX + "analysis_json")
    cols = st.columns(3)
    for col, label, frame, suffix in [
        (cols[0], "Test predictions CSV", run["predictions"], "predictions"),
        (cols[1], "Feature importance CSV", run["importance"], "importance"),
        (cols[2], "Raw data CSV", data, "data")]:
        col.download_button(label, frame.to_csv(index=False), f"nn_base_{suffix}.csv",
                            "text/csv", key=PREFIX + suffix + "_csv")
    with st.expander("Model metadata"):
        st.json({"variant": VARIANT, "parameters": run["selection"]["final_parameters"], "data": meta})


# ============================================================
# PUBLIC PAGE ENTRY POINT
# ============================================================
def render_base_model() -> None:
    """Render the complete Neural Network Base Model page inside the existing Analytics router."""
    render_model_navigation("Neural Network", VARIANT)
    st.caption(f"Analytics / Models / Neural Network / {VARIANT}")
    st.title(f"Neural Network / {VARIANT}")
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
    config = {"features": list(features), "target": target, "test_fraction": test_fraction,
              "seed": int(seed), "layers": [32], "activation": "relu", "alpha": 0.001,
              "learning_rate": 0.001, "epochs": 100, "patience": 10}
    config["epochs"] = st.slider("Maximum epochs" if VARIANT == "Deep Model" else "Training epochs", 20, 300, 100, 20, key=PREFIX + "epochs")
    if VARIANT == "Optimized Model":
        cols = st.columns(2)
        config["trials"] = cols[0].slider("Search candidates", 3, 12, 6, key=PREFIX + "trials")
        config["folds"] = cols[1].slider("Time-series validation folds", 2, 5, 3, key=PREFIX + "folds")
        st.caption(f"{config['trials'] * config['folds']} internal fits + 1 final fit. Search varies architecture, activation, L2 penalty and learning rate. Selection minimizes validation log loss.")
    else:
        architectures = {"32": [32], "64": [64], "128": [128]} if VARIANT == "Base Model" else {"64 / 32 / 16": [64,32,16], "128 / 64 / 32": [128,64,32], "128 / 64 / 32 / 16": [128,64,32,16]}
        chosen = st.selectbox("Hidden-layer widths", list(architectures), key=PREFIX + "layers")
        config["layers"] = architectures[chosen]
        cols = st.columns(3)
        config["activation"] = cols[0].selectbox("Activation", ["relu", "tanh"], key=PREFIX + "activation")
        config["alpha"] = cols[1].select_slider("L2 regularization", [0.0001, 0.001, 0.01, 0.1], value=0.001, key=PREFIX + "alpha")
        config["learning_rate"] = cols[2].select_slider("Learning rate", [0.0001, 0.0003, 0.001, 0.003], value=0.001, key=PREFIX + "rate")
        if VARIANT == "Deep Model":
            config["patience"] = st.slider("Early-stopping patience", 5, 30, 10, 5, key=PREFIX + "patience")
            st.caption("Epoch selection uses a purged validation tail inside training; the network is then refitted from scratch on the full training period for the selected epoch count.")
    st.caption("CPU multilayer perceptron (scikit-learn), Adam optimizer, training-only scaling, chronological holdout and purged boundaries. Training runs sequentially. Probabilities are uncalibrated.")
    if features:
        with st.expander("Feature preview"):
            engineered = _engineer_features(data)[features].dropna()
            st.dataframe(engineered.tail(10), use_container_width=True)
            st.caption(f"{len(engineered):,} feature-complete rows before target cleanup and splitting.")
    if st.button("Run Neural Network", type="primary", key=PREFIX + "train"):
        _reset_results()
        if len(features) < 2:
            st.error("Select at least two features.")
        else:
            try:
                with st.spinner("Training Neural Network and evaluating the chronological holdout…"):
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
