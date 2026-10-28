"""Dashboard and model registry pages."""

from __future__ import annotations

import streamlit as st

from market_ai.components.cards import render_feature_grid, render_market_snapshot
from market_ai.components.charts import (
    render_confidence_heatmap,
    render_distribution_donut,
    render_model_performance,
    render_portfolio_chart,
    render_signal_chart,
)
from market_ai.components.metrics import render_kpi_row
from market_ai.utils.helpers import html_card, status_badge
from market_ai.utils.sample_data import activity_items, model_data, mover_items


def render_dashboard() -> None:
    """Render the full Analytics dashboard."""
    render_kpi_row()
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    chart_col, rail_col = st.columns([1.7, 1], gap="large")
    with chart_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Portfolio performance</h3><span>Benchmark-relative · USD</span></div>', unsafe_allow_html=True)
        periods = ["1D", "1W", "1M", "3M", "1Y", "ALL"]
        period_cols = st.columns(len(periods))
        selected_period = st.session_state.get("portfolio_period", "3M")
        for col, period in zip(period_cols, periods):
            with col:
                if st.button(period, key=f"period_{period}", use_container_width=True):
                    st.session_state.portfolio_period = period
                    st.rerun()
        render_portfolio_chart(selected_period)
        st.markdown("</div>", unsafe_allow_html=True)
    with rail_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        render_market_snapshot()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    left_col, right_col = st.columns([1.15, 1], gap="large")
    with left_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Top movers</h3><span>Today · 09:42 UTC</span></div>', unsafe_allow_html=True)
        for item in mover_items():
            st.markdown(
                f'<div class="mover-row"><div><span class="ticker">{item["ticker"]}</span><span class="ticker-name">{item["name"]}</span></div><div class="move-value {item["tone"]}">{item["change"]}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Recent activity</h3><span>Research stream</span></div>', unsafe_allow_html=True)
        for item in activity_items():
            st.markdown(
                f'<div class="timeline-row"><div><span class="ticker">{item["title"]}</span><small>{item["detail"]}</small></div><small>{item["time"]}</small></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="panel-title"><h3>Model performance</h3><span>Validation accuracy · all active experiments</span></div>', unsafe_allow_html=True)
    render_model_performance(model_data())
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    heat_col, donut_col = st.columns([1.55, 1], gap="large")
    with heat_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Prediction confidence</h3><span>Last 20 weeks</span></div>', unsafe_allow_html=True)
        render_confidence_heatmap()
        st.markdown("</div>", unsafe_allow_html=True)
    with donut_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Model distribution</h3><span>By family</span></div>', unsafe_allow_html=True)
        render_distribution_donut(model_data())
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    render_feature_grid()
    st.markdown("</div>", unsafe_allow_html=True)


def _model_card(row: object, selected: bool) -> None:
    """Render one model registry card."""
    name = row["name"]
    status = row["status"]
    st.markdown(
        f"""
        <div class="model-card">
          <div class="model-card-head"><div><div class="eyebrow">{row["category"]}</div><h4>{name}</h4><div class="model-type">{row["owner"]} · {row["last_trained"]}</div></div>{status_badge(status)}</div>
          <div class="model-accuracy">{row["accuracy"]:.1f}% <span class="caption">accuracy</span></div>
          <div class="progress-track"><div class="progress-fill" style="width:{row["accuracy"]}%"></div></div>
          <div class="model-stats"><span>P {row["precision"]:.1f}</span><span>R {row["recall"]:.1f}</span><span>F1 {row["f1"]:.1f}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_models() -> None:
    """Render the searchable model registry."""
    data = model_data()
    search_col, status_col, category_col, sort_col = st.columns([1.5, 1, 1, 1])
    with search_col:
        query = st.text_input("Search models", placeholder="Search by model or owner…", label_visibility="collapsed")
    with status_col:
        status_filter = st.selectbox("Status", ["All statuses", "Active", "Training", "Inactive"], label_visibility="collapsed")
    with category_col:
        category_filter = st.selectbox("Category", ["All categories"] + sorted(data["category"].unique().tolist()), label_visibility="collapsed")
    with sort_col:
        sort_by = st.selectbox("Sort", ["Accuracy", "Name", "Features"], label_visibility="collapsed")

    filtered = data.copy()
    if query:
        filtered = filtered[filtered.apply(lambda row: query.lower() in " ".join(row.astype(str)).lower(), axis=1)]
    if status_filter != "All statuses":
        filtered = filtered[filtered["status"] == status_filter]
    if category_filter != "All categories":
        filtered = filtered[filtered["category"] == category_filter]
    sort_column = {"Accuracy": "accuracy", "Name": "name", "Features": "features_count"}[sort_by]
    filtered = filtered.sort_values(sort_column, ascending=sort_by == "Name")

    st.markdown(f'<div class="caption" style="margin: 1rem 0 .75rem">{len(filtered)} models matching current view · select up to 3 for comparison</div>', unsafe_allow_html=True)
    selected_names = st.multiselect("Compare models", filtered["name"].tolist(), max_selections=3, label_visibility="collapsed", placeholder="Compare up to 3 models…")
    if selected_names:
        comparison = filtered[filtered["name"].isin(selected_names)]
        st.markdown('<div class="panel"><div class="panel-title"><h3>Model comparison</h3><span>Validation metrics</span></div>', unsafe_allow_html=True)
        st.dataframe(comparison[["name", "category", "accuracy", "precision", "recall", "f1", "training_time" if "training_time" in comparison.columns else "features_count"]], hide_index=True, use_container_width=True, column_config={"accuracy": st.column_config.ProgressColumn("Accuracy", format="%.1f%%", min_value=0, max_value=100), "precision": st.column_config.NumberColumn("Precision", format="%.1f"), "recall": st.column_config.NumberColumn("Recall", format="%.1f"), "f1": st.column_config.NumberColumn("F1", format="%.1f")})
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    if filtered.empty:
        html_card('<div class="empty-state"><div class="empty-icon">⌕</div><h3>No matching models</h3><p>Try a different search term or clear one of the filters.</p></div>')
        return

    card_cols = st.columns(3, gap="medium")
    for index, (_, row) in enumerate(filtered.iterrows()):
        with card_cols[index % 3]:
            _model_card(row, row["name"] in selected_names)
        if index % 3 == 2:
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    queue_col, signal_col = st.columns([1, 1], gap="large")
    with queue_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Training queue</h3><span>2 active jobs</span></div>', unsafe_allow_html=True)
        for model, progress, eta in [("LSTM", 72, "04:12 remaining"), ("CatBoost", 38, "12:48 remaining")]:
            st.markdown(f'<div class="mover-row"><div><span class="ticker">{model}</span><span class="ticker-name">Cross-validation run</span></div><div class="move-value neutral">{progress}% · {eta}</div></div><div class="progress-track"><div class="progress-fill" style="width:{progress}%"></div></div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with signal_col:
        st.markdown('<div class="panel"><div class="panel-title"><h3>Signal distribution</h3><span>Last 30 days</span></div>', unsafe_allow_html=True)
        render_signal_chart()
        st.markdown("</div>", unsafe_allow_html=True)