"""MLOps model detail view for the local model registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_ai.components.ui.feedback import toast_success
from market_ai.mlops.audit import read_audit
from market_ai.mlops.lifecycle import ALLOWED_TRANSITIONS, archive_model, promote_model
from market_ai.mlops.paths import PREDICTIONS_PATH
from market_ai.mlops.registry import register_version
from market_ai.mlops.runs import list_runs
from market_ai.utils.theme import COLORS


def _metric_value(model: dict[str, Any], key: str) -> float:
    return float(model.get("metrics", {}).get(key, 0.0))


def _read_predictions(model_id: str) -> pd.DataFrame:
    """Read local prediction monitoring records for one model."""
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame([row for row in rows if row.get("model_id") == model_id])


@st.dialog("Confirm model promotion")
def _promotion_dialog(model_id: str, target_stage: str) -> None:
    """Confirm a lifecycle stage transition."""
    st.warning(f"Promote this model to **{target_stage}**? The action will be added to the audit log.")
    if st.button(f"Confirm {target_stage}", type="primary", use_container_width=True):
        try:
            promote_model(model_id, target_stage)
            toast_success(f"Model promoted to {target_stage}")
            st.session_state.pop("mlops_selected_model_id", None)
            st.rerun()
        except ValueError as error:
            st.error(str(error))


def _overview(model: dict[str, Any]) -> None:
    snapshot = model.get("training_snapshot", {})
    meta_col, snapshot_col = st.columns(2, gap="large")
    with meta_col:
        st.markdown("**Model metadata**")
        st.dataframe(
            [
                {"Field": "Model ID", "Value": model.get("model_id", "—")},
                {"Field": "Model type", "Value": model.get("model_type", "—")},
                {"Field": "Framework", "Value": model.get("framework", "—")},
                {"Field": "Owner", "Value": model.get("owner", "—")},
                {"Field": "Stage", "Value": model.get("stage", "Development")},
                {"Field": "Tags", "Value": ", ".join(model.get("tags", []))},
            ],
            hide_index=True,
            use_container_width=True,
        )
    with snapshot_col:
        st.markdown("**Training data snapshot**")
        st.dataframe(
            [
                {"Field": "Dataset", "Value": snapshot.get("dataset", "—")},
                {"Field": "Shape", "Value": f'{snapshot.get("rows", "—"):,} rows × {snapshot.get("feature_count", "—")} features' if isinstance(snapshot.get("rows"), int) else "—"},
                {"Field": "Date range", "Value": snapshot.get("date_range", "—")},
                {"Field": "Features", "Value": ", ".join(model.get("features", []))},
                {"Field": "Last trained", "Value": model.get("last_trained", "—")},
            ],
            hide_index=True,
            use_container_width=True,
        )
    with st.expander("Hyperparameters"):
        latest = (model.get("versions") or [{}])[-1]
        st.json(latest.get("hyperparameters", {}))


def _metrics(model: dict[str, Any]) -> None:
    versions = model.get("versions", [])
    chart = go.Figure()
    chart.add_trace(go.Scatter(x=[version.get("version") for version in versions], y=[version.get("metrics", {}).get("accuracy", 0) for version in versions], name="Accuracy", mode="lines+markers", line={"color": COLORS["cyan"], "width": 2}))
    chart.add_trace(go.Scatter(x=[version.get("version") for version in versions], y=[version.get("metrics", {}).get("auc", 0) for version in versions], name="AUC", mode="lines+markers", line={"color": COLORS["violet"], "width": 2}))
    chart.update_layout(title="Metrics over versions", height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin={"l": 0, "r": 0, "t": 45, "b": 0}, yaxis={"range": [0.65, 1], "tickformat": ".0%"}, legend={"orientation": "h", "y": 1.12, "x": 0})
    st.plotly_chart(chart, use_container_width=True, config={"displaylogo": False})
    metrics = model.get("metrics", {})
    st.dataframe([{"Metric": key.upper(), "Value": value} for key, value in metrics.items()], hide_index=True, use_container_width=True)
    st.markdown("**Validation snapshot**")
    matrix = [[100, 31], [19, 152]]
    confusion = go.Figure(go.Heatmap(z=matrix, x=["Bearish", "Bullish"], y=["Bearish", "Bullish"], text=matrix, texttemplate="%{text}", colorscale=[[0, "#141c2b"], [1, COLORS["cyan"]]], showscale=False))
    confusion.update_layout(height=220, margin={"l": 0, "r": 0, "t": 10, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(confusion, use_container_width=True, config={"displaylogo": False})


def _artifacts(model: dict[str, Any]) -> None:
    st.dataframe(model.get("artifacts", []), hide_index=True, use_container_width=True)
    st.caption("Artifacts are local paths in the demo registry. They are metadata records, not uploaded binaries.")


def _lineage(model: dict[str, Any]) -> None:
    st.markdown("**Dataset → preprocessing → model → metrics**")
    st.code(
        f"""flowchart LR
    D[Dataset\\n{model.get("training_snapshot", {}).get("dataset", "market_factors_v1")}] --> P[Feature preprocessing\\n{len(model.get("features", []))} features]
    P --> M[{model.get("model_type", "Model")}\\n{model.get("model_id", "model")}]
    M --> E[Evaluation\\naccuracy {_metric_value(model, "accuracy"):.1%}]
    E --> A[Artifacts\\n{len(model.get("artifacts", []))} saved files]""",
        language="mermaid",
    )


def _versions(model: dict[str, Any]) -> None:
    versions = model.get("versions", [])
    st.dataframe(
        [{"Version": version.get("version"), "Created": version.get("created_at"), "Accuracy": version.get("metrics", {}).get("accuracy"), "AUC": version.get("metrics", {}).get("auc"), "Artifact": version.get("artifact_path")} for version in versions],
        hide_index=True,
        use_container_width=True,
    )
    options = [version.get("version") for version in versions]
    if len(options) >= 2:
        selected = st.multiselect("Compare versions", options, default=options[-2:], max_selections=2, key=f"version_compare_{model['model_id']}")
        if len(selected) == 2:
            comparison = []
            for version in versions:
                if version.get("version") in selected:
                    comparison.append({"Version": version.get("version"), "Accuracy": version.get("metrics", {}).get("accuracy"), "AUC": version.get("metrics", {}).get("auc"), "Hyperparameters": json.dumps(version.get("hyperparameters", {}))})
            st.dataframe(comparison, hide_index=True, use_container_width=True)
    with st.expander("Register new version"):
        artifact_path = st.text_input("Artifact path", f'artifacts/{model["model_id"]}/next/model.joblib')
        if st.button("Register version", type="primary", key=f"register_version_{model['model_id']}"):
            register_version(model["model_id"], model.get("metrics", {}), (versions[-1] if versions else {}).get("hyperparameters", {}), artifact_path)
            toast_success("New model version registered")
            st.rerun()


def _runs(model: dict[str, Any]) -> None:
    runs = list_runs(model.get("model_id"))
    st.dataframe(
        [{"Run ID": run.get("run_id"), "Started": run.get("started_at"), "Duration": f'{run.get("duration_seconds", 0):.1f}s', "Status": run.get("status"), "Accuracy": run.get("metrics", {}).get("accuracy"), "Git commit": run.get("git_commit", "—")} for run in runs],
        hide_index=True,
        use_container_width=True,
    )
    options = [run.get("run_id") for run in runs]
    if len(options) >= 2:
        selected = st.multiselect("Compare runs", options, max_selections=2, key=f"run_compare_{model['model_id']}")
        if len(selected) == 2:
            chosen = [run for run in runs if run.get("run_id") in selected]
            st.dataframe([{"Run": run.get("run_id"), "Status": run.get("status"), "Duration": run.get("duration_seconds"), "Metrics": json.dumps(run.get("metrics", {})), "Hyperparameters": json.dumps(run.get("hyperparameters", {}))} for run in chosen], hide_index=True, use_container_width=True)


def _monitoring(model: dict[str, Any]) -> None:
    predictions = _read_predictions(model.get("model_id", ""))
    if predictions.empty:
        st.info("No local prediction records are available for this model yet.")
        return
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    health = "Healthy"
    if float(predictions["psi"].max()) >= 0.2:
        health = "Critical"
    elif float(predictions["psi"].max()) >= 0.1:
        health = "Warning"
    tone = {"Healthy": "active", "Warning": "training", "Critical": "failed"}[health]
    st.markdown(f'<div class="monitoring-health"><span class="badge badge-{tone}">● {health}</span><span>Health is computed from local PSI drift thresholds.</span></div>', unsafe_allow_html=True)
    volume_col, drift_col = st.columns(2, gap="large")
    with volume_col:
        volume = go.Figure(go.Bar(x=predictions["timestamp"], y=predictions["volume"], marker_color=COLORS["cyan"]))
        volume.update_layout(title="Prediction volume over time", height=260, margin={"l": 0, "r": 0, "t": 40, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(volume, use_container_width=True, config={"displaylogo": False})
    with drift_col:
        drift = go.Figure(go.Scatter(x=predictions["timestamp"], y=predictions["psi"], mode="lines+markers", line={"color": COLORS["yellow"], "width": 2}, name="PSI"))
        drift.add_hline(y=0.1, line={"color": COLORS["yellow"], "dash": "dot"}, annotation_text="Warning")
        drift.add_hline(y=0.2, line={"color": COLORS["red"], "dash": "dot"}, annotation_text="Critical")
        drift.update_layout(title="Feature distribution drift (PSI)", height=260, margin={"l": 0, "r": 0, "t": 40, "b": 0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(drift, use_container_width=True, config={"displaylogo": False})
    st.dataframe(predictions[["timestamp", "volume", "psi", "health"]].tail(10), hide_index=True, use_container_width=True)


def render_model_detail(model: dict[str, Any]) -> None:
    """Render the MLOps detail view for one registered model."""
    header_col, actions_col = st.columns([1.5, 1])
    with header_col:
        st.markdown(f'<div class="page-kicker">MODEL DETAIL · {model.get("model_id", "").upper()}</div><h2>{model.get("name", "Model")}</h2><p class="page-subtitle">{model.get("description", "Registered model lifecycle and monitoring workspace.")}</p>', unsafe_allow_html=True)
    with actions_col:
        st.markdown(f'<div class="detail-stage"><span class="badge badge-info">{model.get("stage", "Development")}</span><span class="setting-description">Last trained {model.get("last_trained", "—")}</span></div>', unsafe_allow_html=True)
        allowed = sorted(ALLOWED_TRANSITIONS.get(str(model.get("stage", "Development")), set()))
        action_cols = st.columns(2)
        if allowed:
            target = action_cols[0].selectbox("Promote to", allowed, key=f"promote_target_{model['model_id']}", label_visibility="collapsed")
            if action_cols[0].button("Promote", key=f"promote_{model['model_id']}", use_container_width=True):
                _promotion_dialog(model["model_id"], target)
        if action_cols[1].button("Archive", key=f"archive_{model['model_id']}", use_container_width=True):
            try:
                archive_model(model["model_id"])
                toast_success("Model archived")
                st.session_state.pop("mlops_selected_model_id", None)
                st.rerun()
            except ValueError as error:
                st.error(str(error))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f'{_metric_value(model, "accuracy"):.1%}')
    metric_cols[1].metric("AUC", f'{_metric_value(model, "auc"):.1%}')
    metric_cols[2].metric("Versions", len(model.get("versions", [])))
    metric_cols[3].metric("Runs", len(list_runs(model.get("model_id"))))
    tabs = st.tabs(["Overview", "Metrics", "Artifacts", "Lineage", "Versions", "Runs", "Monitoring"])
    with tabs[0]:
        _overview(model)
    with tabs[1]:
        _metrics(model)
    with tabs[2]:
        _artifacts(model)
    with tabs[3]:
        _lineage(model)
    with tabs[4]:
        _versions(model)
    with tabs[5]:
        _runs(model)
    with tabs[6]:
        _monitoring(model)
    with st.expander("Recent audit events"):
        events = [event for event in read_audit(100) if event.get("entity_id") == model.get("model_id")]
        st.dataframe(events, hide_index=True, use_container_width=True)