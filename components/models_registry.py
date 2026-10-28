"""Top-level model registry page."""

from __future__ import annotations

import json

import streamlit as st

from market_ai.components.ui.cards import avatar_stack, badge, stat_card
from market_ai.components.ui.charts import circular_progress, radar_chart
from market_ai.components.ui.feedback import empty_state, toast_success
from market_ai.components.ui.inputs import search_bar
from market_ai.components.models_registry_detail import render_model_detail
from market_ai.mlops.audit import read_audit
from market_ai.mlops.paths import PREDICTIONS_PATH
from market_ai.mlops.registry import load_registry
from market_ai.mlops.seed_demo import ensure_demo_seeded
from market_ai.utils.helpers import sparkline
from market_ai.utils.sample_data import model_data


def _registry_data() -> list[dict[str, object]]:
    rows = model_data().to_dict("records")
    metadata = [
        ("scikit-learn", "Classification", "v2.3.1", "Predict next-day direction from technical factors.", "Aisha", "4.2s"),
        ("Prophet", "Time Series", "v1.8.0", "Seasonality-aware market trend forecaster.", "Maya", "2.1s"),
        ("PyTorch", "Time Series", "v3.0.4", "Attention model for multi-horizon forecasts.", "Aisha", "8.8s"),
        ("PyTorch", "Time Series", "v1.4.2", "Patch-based transformer for noisy series.", "Ravi", "6.3s"),
        ("XGBoost", "Regression", "v2.1.0", "Fast factor model for cross-sectional returns.", "Kai", "3.4s"),
        ("TensorFlow", "NLP", "v4.2.0", "Financial language sentiment classifier.", "Maya", "5.9s"),
        ("XGBoost", "Regression", "v1.6.2", "Tree ensemble for portfolio risk scores.", "Kai", "3.8s"),
        ("scikit-learn", "Classification", "v2.0.1", "Baseline classifier for event studies.", "Ravi", "1.9s"),
        ("PyTorch", "Time Series", "v0.9.8", "Foundation model adapter for sparse data.", "Aisha", "12.2s"),
        ("TensorFlow", "Regression", "v1.2.5", "Long-range macro scenario model.", "Maya", "9.1s"),
        ("PyTorch", "Reinforcement", "v2.2.0", "Policy learner for execution decisions.", "Kai", "11.4s"),
        ("scikit-learn", "Classification", "v1.9.3", "Lightweight market state classifier.", "Ravi", "2.6s"),
    ]
    result = []
    for row, details in zip(rows, metadata):
        framework, category, version, description, author, training_time = details
        result.append({**row, "framework": framework, "category": category, "version": version, "description": description, "author": author, "training_time": training_time, "tags": [category, framework]})
    return result


def _models() -> list[dict[str, object]]:
    if "registry_models" not in st.session_state:
        st.session_state.registry_models = _registry_data()
    return st.session_state.registry_models


@st.dialog("Deploy a new model")
def _deploy_dialog() -> None:
    """Collect basic deployment metadata."""
    name = st.text_input("Model name", placeholder="e.g. Rates Regime v1")
    framework = st.selectbox("Framework", ["scikit-learn", "XGBoost", "PyTorch", "TensorFlow"])
    environment = st.radio("Environment", ["Staging", "Production"], horizontal=True)
    if st.button("Deploy model", type="primary", use_container_width=True):
        if not name.strip():
            st.warning("Add a model name to continue.")
            return
        toast_success(f"{name.strip()} queued for {environment.lower()}")
        st.rerun()


def _model_card(model: dict[str, object], index: int) -> None:
    """Render a registry model card with actions."""
    name = str(model["name"])
    selected = name in st.session_state.get("selected_registry_models", [])
    framework_icon = {"scikit-learn": "🐍", "XGBoost": "ϟ", "PyTorch": "◇", "TensorFlow": "◎"}.get(str(model["framework"]), "◇")
    tone = {"Active": "active", "Training": "training", "Inactive": "inactive"}.get(str(model["status"]), "info")
    st.markdown(
        f"""
        <div class="registry-card {"registry-selected" if selected else ""}">
          <div class="registry-top"><span class="framework-icon">{framework_icon}</span><div>{badge(str(model["status"]), tone)} <span class="version">{model["version"]}</span> <span class="registry-star">☆</span></div></div>
          <div class="registry-name-row"><div><h3>{name}</h3><span class="setting-description">{model["framework"]} · {model["category"]}</span></div>{circular_progress(float(model["accuracy"]))}</div>
          <p class="registry-description">{model["description"]}</p>
          <div class="registry-metrics"><span><b>{model["f1"]:.1f}</b> F1</span><span><b>{model["precision"]:.1f}</b> Precision</span><span><b>{model["recall"]:.1f}</b> Recall</span></div>
          {sparkline([float(model["accuracy"]) - 5, float(model["accuracy"]) - 3, float(model["accuracy"]) - 4, float(model["accuracy"]) - 1, float(model["accuracy"])], "#31D7EE")}
          <div class="registry-footer"><span>Trained {model["last_trained"]} · {model["training_time"]} · 1,247 samples</span>{avatar_stack([str(model["author"])])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    action_cols = st.columns([0.55, 1, 1, 1])
    with action_cols[0]:
        checked = st.checkbox("Compare", value=selected, key=f"select_registry_{name}", label_visibility="collapsed")
    if checked and name not in st.session_state.selected_registry_models:
        st.session_state.selected_registry_models.append(name)
    if not checked and name in st.session_state.selected_registry_models:
        st.session_state.selected_registry_models.remove(name)
    for col, label in zip(action_cols[1:], ["Predict", "Deploy", "View"]):
        with col:
            if st.button(label, key=f"registry_{label}_{name}_{index}", use_container_width=True):
                if label == "View":
                    match = next((item for item in load_registry() if item.get("name") == name), None)
                    if match:
                        st.session_state.mlops_selected_model_id = match["model_id"]
                        st.rerun()
                else:
                    toast_success(f"{label} action ready for {name}")


def _render_mlops_registry_table() -> None:
    """Render the persistent MLOps summary and registry table."""
    records = load_registry()
    if not records:
        st.info("The local MLOps registry has no records yet.")
        return
    predictions = []
    if PREDICTIONS_PATH.exists():
        predictions = [line for line in PREDICTIONS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    production = sum(record.get("stage") == "Production" for record in records)
    drifting = sum("Warning" in line or "Critical" in line for line in predictions)
    promotions = [event for event in read_audit(100) if event.get("action") == "model.stage_changed"]
    last_promotion = promotions[0].get("timestamp", "—") if promotions else "No promotions yet"
    st.markdown('<div class="mlops-summary-heading"><div><div class="page-kicker">MLOPS CONTROL PLANE</div><h3>Lifecycle, versions, and health</h3></div><span class="setting-description">Local file-backed registry</span></div>', unsafe_allow_html=True)
    summary = st.columns(4)
    summary[0].metric("Total models", len(records))
    summary[1].metric("In production", production)
    summary[2].metric("Drifting", drifting)
    summary[3].metric("Last promotion", last_promotion[:16].replace("T", " ") if last_promotion != "No promotions yet" else last_promotion)
    filter_cols = st.columns([1.5, 1, 1])
    query = filter_cols[0].text_input("MLOps search", placeholder="Search model ID, name, owner, or tag…", key="mlops_table_query")
    stage_filter = filter_cols[1].selectbox("MLOps stage", ["All stages", "Development", "Staging", "Production", "Archived"], key="mlops_stage_filter")
    sort_by = filter_cols[2].selectbox("MLOps sort", ["Last trained", "Name", "Stage", "Accuracy"], key="mlops_sort")
    visible = records
    if query:
        visible = [record for record in visible if query.lower() in json.dumps(record).lower()]
    if stage_filter != "All stages":
        visible = [record for record in visible if record.get("stage") == stage_filter]
    if sort_by == "Name":
        visible = sorted(visible, key=lambda record: str(record.get("name", "")))
    elif sort_by == "Stage":
        visible = sorted(visible, key=lambda record: str(record.get("stage", "")))
    elif sort_by == "Accuracy":
        visible = sorted(visible, key=lambda record: float(record.get("metrics", {}).get("accuracy", 0)), reverse=True)
    else:
        visible = sorted(visible, key=lambda record: str(record.get("last_trained", "")), reverse=True)
    st.dataframe(
        [
            {
                "Model ID": record.get("model_id"),
                "Name": record.get("name"),
                "Type": record.get("model_type"),
                "Version": (record.get("versions") or [{}])[-1].get("version", "—"),
                "Stage": record.get("stage"),
                "Metrics": f'Accuracy {_metric_accuracy(record):.1%} · AUC {float(record.get("metrics", {}).get("auc", 0)):.1%}',
                "Last Trained": record.get("last_trained"),
                "Owner": record.get("owner"),
                "Tags": ", ".join(record.get("tags", [])),
            }
            for record in visible
        ],
        hide_index=True,
        use_container_width=True,
    )
    if visible:
        st.markdown('<div class="caption">Open a model to inspect versions, runs, artifacts, lineage, and monitoring.</div>', unsafe_allow_html=True)
        open_cols = st.columns(min(4, len(visible)))
        for index, record in enumerate(visible):
            with open_cols[index % len(open_cols)]:
                if st.button(f'Open {record.get("name")}', key=f'mlops_open_{record.get("model_id")}', use_container_width=True):
                    st.session_state.mlops_selected_model_id = record.get("model_id")
                    st.rerun()


def _metric_accuracy(record: dict[str, object]) -> float:
    """Read an accuracy metric from a registry record."""
    return float(record.get("metrics", {}).get("accuracy", 0))


def render_models_registry() -> None:
    """Render the standalone model registry."""
    ensure_demo_seeded()
    selected_model_id = st.session_state.get("mlops_selected_model_id")
    if selected_model_id:
        selected_model = next((record for record in load_registry() if record.get("model_id") == selected_model_id), None)
        if selected_model:
            if st.button("← Back to model registry", key="back_to_mlops_registry"):
                st.session_state.pop("mlops_selected_model_id", None)
                st.rerun()
            render_model_detail(selected_model)
            return
        st.session_state.pop("mlops_selected_model_id", None)
    models = _models()
    st.markdown(
        '<div class="registry-hero"><div><div class="page-kicker">MODEL OPERATIONS · REGISTRY</div><h2>Every model, one source of truth.</h2><p>Track versions, validation metrics and deployment readiness across your research stack.</p></div><div class="registry-hero-actions"></div></div>',
        unsafe_allow_html=True,
    )
    action_cols = st.columns([1, 1, 1, 0.7])
    with action_cols[0]:
        if st.button("＋ Deploy new model", type="primary", use_container_width=True):
            _deploy_dialog()
    with action_cols[1]:
        if st.button("Import model", use_container_width=True):
            toast_success("Import flow ready")
    with action_cols[2]:
        if st.button("View docs", use_container_width=True):
            toast_success("Documentation link ready")
    with action_cols[3]:
        if st.button("↻ Refresh", use_container_width=True):
            toast_success("Registry refreshed")

    st.markdown('<div class="mlops-summary-panel">', unsafe_allow_html=True)
    _render_mlops_registry_table()
    st.markdown("</div><div class=\"section-gap\"></div>", unsafe_allow_html=True)
    st.markdown('<div class="stat-badges"><span>12 Active</span><span>3 Training</span><span>2 Pending deploy</span><span>5 Archived</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    stats = st.columns(4)
    with stats[0]:
        stat_card("Average accuracy", "87.3%", "+2.1% this month", "◎", [82, 83, 84, 84, 86, 87], footer="Across registered models")
    with stats[1]:
        stat_card("Average training time", "4.2s", "−0.8s this month", "ϟ", [7, 6, 5.8, 5, 4.7, 4.2], tone="positive", footer="Best: 1.9s")
    with stats[2]:
        stat_card("Deployments this week", "8", "+3 vs. last week", "↗", [2, 3, 3, 4, 5, 8], footer="2 pending approval")
    with stats[3]:
        stat_card("Total model size", "245 MB", "24.5% of quota", "◫", [120, 135, 160, 180, 215, 245], tone="neutral", footer="1 GB workspace quota")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    filter_cols = st.columns([1.3, 0.78, 0.78, 0.78, 0.78, 0.78])
    with filter_cols[0]:
        query = search_bar("Search registry", "registry_query", "Search name, tag, author…")
    with filter_cols[1]:
        category = st.selectbox("Category", ["All categories", "Classification", "Regression", "Time Series", "NLP", "Reinforcement", "Clustering"], label_visibility="collapsed")
    with filter_cols[2]:
        status = st.selectbox("Status", ["All statuses", "Active", "Training", "Inactive"], label_visibility="collapsed")
    with filter_cols[3]:
        framework = st.selectbox("Framework", ["All frameworks", "scikit-learn", "XGBoost", "LightGBM", "PyTorch", "TensorFlow", "JAX"], label_visibility="collapsed")
    with filter_cols[4]:
        sort_by = st.selectbox("Sort", ["Recently used", "Accuracy high → low", "Accuracy low → high", "Name"], label_visibility="collapsed")
    with filter_cols[5]:
        view_mode = st.selectbox("View", ["Grid", "Table", "List", "Compact"], label_visibility="collapsed")

    visible = models
    if query:
        visible = [model for model in visible if query.lower() in str(model).lower()]
    if category != "All categories":
        visible = [model for model in visible if model["category"] == category]
    if status != "All statuses":
        visible = [model for model in visible if model["status"] == status]
    if framework != "All frameworks":
        visible = [model for model in visible if model["framework"] == framework]
    if sort_by == "Name":
        visible = sorted(visible, key=lambda model: str(model["name"]).lower())
    elif sort_by == "Accuracy low → high":
        visible = sorted(visible, key=lambda model: float(model["accuracy"]))
    else:
        visible = sorted(visible, key=lambda model: float(model["accuracy"]), reverse=True)

    if "selected_registry_models" not in st.session_state:
        st.session_state.selected_registry_models = []
    selected = [model for model in models if model["name"] in st.session_state.selected_registry_models]
    if selected:
        bulk_cols = st.columns([1.1, 0.8, 0.8, 0.8, 0.8, 2])
        bulk_cols[0].markdown(f'<div class="bulk-actions"><strong>{len(selected)} selected</strong></div>', unsafe_allow_html=True)
        for col, label in zip(bulk_cols[1:5], ["Deploy", "Export", "Archive", "Delete"]):
            if col.button(label, key=f"bulk_{label}", use_container_width=True):
                toast_success(f"{label} action ready")
        if bulk_cols[5].button("Clear selection", key="clear_registry_selection", use_container_width=True):
            st.session_state.selected_registry_models = []
            st.rerun()
    if len(selected) >= 2:
        st.markdown('<div class="section-gap"></div><div class="panel"><div class="panel-title"><h3>Comparison workspace</h3><span>Selected models</span></div>', unsafe_allow_html=True)
        radar_col, delta_col = st.columns([1, 1])
        with radar_col:
            radar_chart(["Accuracy", "Precision", "Recall", "F1"], {str(model["name"]): [float(model["accuracy"]), float(model["precision"]), float(model["recall"]), float(model["f1"])] for model in selected[:3]})
        with delta_col:
            st.dataframe([{ "Model": model["name"], "Accuracy": model["accuracy"], "F1": model["f1"], "Framework": model["framework"] } for model in selected[:3]], hide_index=True, use_container_width=True, column_config={"Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=100, format="%.1f%%")})
        st.markdown("</div><div class=\"section-gap\"></div>", unsafe_allow_html=True)

    st.markdown(f'<div class="panel-title"><h3>Model registry</h3><span>{len(visible)} results · {view_mode.lower()} view</span></div>', unsafe_allow_html=True)
    if not visible:
        empty_state("◇", "Your registry is empty", "Import or train a model to populate this workspace.")
        return
    if view_mode == "Table":
        st.dataframe(
            [{"Model": model["name"], "Version": model["version"], "Category": model["category"], "Framework": model["framework"], "Accuracy": model["accuracy"], "F1": model["f1"], "Status": model["status"], "Author": model["author"], "Samples": 1247} for model in visible],
            hide_index=True,
            use_container_width=True,
            column_config={"Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=100, format="%.1f%%"), "F1": st.column_config.NumberColumn("F1", format="%.1f")},
        )
    elif view_mode == "List":
        for index, model in enumerate(visible):
            st.markdown(f'<div class="registry-list-row"><span class="framework-icon">{str(model["framework"])[:1]}</span><strong>{model["name"]}</strong><span class="version">{model["version"]}</span><span>{model["category"]}</span><span class="positive">{model["accuracy"]:.1f}%</span><span>{model["status"]}</span></div>', unsafe_allow_html=True)
            if st.button("View", key=f"list_view_{index}", use_container_width=False):
                toast_success(f'Viewing {model["name"]}')
    else:
        columns = 4 if view_mode == "Compact" else 3
        card_cols = st.columns(columns, gap="medium")
        for index, model in enumerate(visible):
            with card_cols[index % columns]:
                _model_card(model, index)
            if index % columns == columns - 1:
                st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    queue_col, analytics_col = st.columns([1, 1], gap="large")
    with queue_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Training queue</h3><span>3 running · 5 queued</span></div>', unsafe_allow_html=True)
        running = [("LSTM-v3", 78, "4 min"), ("XGBoost-Premium", 65, "8 min"), ("RandomForest-v5", 42, "12 min")]
        for queue_index, (name, progress, eta) in enumerate(running):
            st.markdown(f'<div class="queue-row"><div><strong>{name}</strong><span>ETA {eta}</span></div><div class="queue-progress"><div class="progress-track"><div class="progress-fill" style="width:{progress}%"></div></div><small>{progress}%</small></div></div>', unsafe_allow_html=True)
            queue_actions = st.columns([1, 1, 3])
            if queue_actions[0].button("Cancel", key=f"cancel_training_{queue_index}"):
                toast_success(f"{name} cancelled")
            if queue_actions[1].button("Logs", key=f"logs_training_{queue_index}"):
                toast_success(f"Logs opened for {name}")
        st.markdown('<div class="subheading">Queued next</div>', unsafe_allow_html=True)
        st.caption("Transformer-XL · GRU-Sentiment · SVM-Intraday · LightGBM-Trend · MLP-Classifier")
        st.markdown("</div>", unsafe_allow_html=True)
    with analytics_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Registry signals</h3><span>Operations overview</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="registry-signal"><span>Training success rate</span><strong class="positive">96.4%</strong></div><div class="registry-signal"><span>Average inference speed</span><strong>18 ms</strong></div><div class="registry-signal"><span>Models needing attention</span><strong class="neutral">2</strong></div><div class="registry-signal"><span>Most used this month</span><strong>Transformer v3</strong></div>', unsafe_allow_html=True)
        if st.button("View performance analytics", key="registry_analytics", use_container_width=True):
            toast_success("Analytics view ready")
        st.markdown("</div>", unsafe_allow_html=True)