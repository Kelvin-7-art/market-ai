"""Chart primitives used by registry and settings pages."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from market_ai.utils.theme import COLORS


def circular_progress(value: float, label: str = "accuracy") -> str:
    """Return a circular progress ring using conic-gradient CSS."""
    return (
        f'<div class="circular-progress" style="--progress:{value * 3.6}deg">'
        f'<span>{value:.0f}%</span></div><div class="circular-label">{label}</div>'
    )


def radar_chart(labels: list[str], series: dict[str, list[float]]) -> None:
    """Render a compact radar chart for model comparison."""
    fig = go.Figure()
    palette = [COLORS["cyan"], COLORS["violet"], COLORS["blue"]]
    for index, (name, values) in enumerate(series.items()):
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                name=name,
                fill="toself",
                opacity=0.28,
                line={"color": palette[index % len(palette)], "width": 2},
            )
        )
    fig.update_layout(
        height=290,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, sans-serif", "color": COLORS["muted"], "size": 10},
        margin={"l": 20, "r": 20, "t": 10, "b": 10},
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "range": [70, 100], "gridcolor": COLORS["grid"]},
            "angularaxis": {"gridcolor": COLORS["grid"]},
        },
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})