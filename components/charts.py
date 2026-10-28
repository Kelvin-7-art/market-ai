"""Plotly chart components with a shared dark theme."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_ai.utils.sample_data import confidence_heatmap, portfolio_data, signal_distribution
from market_ai.utils.theme import COLORS


def _layout(fig: go.Figure, height: int = 290) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, sans-serif", "color": COLORS["muted"], "size": 10},
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        hoverlabel={"bgcolor": COLORS["panel_alt"], "font": {"color": COLORS["text"]}},
        legend={"orientation": "h", "y": 1.08, "x": 0, "font": {"size": 10}},
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, fixedrange=False)
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False, showline=False)
    return fig


def render_portfolio_chart(period: str = "3M") -> None:
    """Render portfolio performance vs benchmark."""
    periods = {"1D": 2, "1W": 7, "1M": 30, "3M": 90, "1Y": 252, "ALL": 252}
    data = portfolio_data(periods[period]).tail(periods[period])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["date"], y=data["Portfolio"], name="Portfolio", mode="lines", line={"color": COLORS["cyan"], "width": 2.5}, fill="tozeroy", fillcolor="rgba(49,215,238,0.07)", hovertemplate="%{y:.2f}<extra>Portfolio</extra>"))
    fig.add_trace(go.Scatter(x=data["date"], y=data["Benchmark"], name="Benchmark", mode="lines", line={"color": COLORS["violet"], "width": 1.6, "dash": "dot"}, hovertemplate="%{y:.2f}<extra>Benchmark</extra>"))
    _layout(fig, 310)
    fig.update_layout(legend={"orientation": "h", "y": 1.10, "x": 0})
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})


def render_model_performance(data: pd.DataFrame) -> None:
    """Render a horizontal model performance comparison."""
    ordered = data.sort_values("accuracy", ascending=True)
    fig = go.Figure(go.Bar(x=ordered["accuracy"], y=ordered["name"], orientation="h", marker={"color": ordered["accuracy"], "colorscale": [[0, COLORS["violet"]], [0.5, COLORS["blue"]], [1, COLORS["cyan"]]]}, hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    _layout(fig, 340)
    fig.update_layout(xaxis={"range": [80, 100], "ticksuffix": "%", "showgrid": True, "gridcolor": COLORS["grid"]}, yaxis={"title": ""})
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})


def render_signal_chart() -> None:
    """Render signal distribution."""
    data = signal_distribution()
    fig = go.Figure(go.Bar(x=data["signal"], y=data["count"], marker_color=[COLORS["green"], COLORS["red"], COLORS["blue"]], text=data["count"], textposition="outside", hovertemplate="%{x}: %{y}<extra></extra>"))
    _layout(fig, 220)
    fig.update_layout(yaxis={"visible": False}, xaxis={"showgrid": False})
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})


def render_confidence_heatmap() -> None:
    """Render a calendar-like prediction confidence heatmap."""
    data = confidence_heatmap()
    pivot = data.pivot_table(index="weekday", columns="date", values="confidence", aggfunc="mean")
    weekday_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = pivot.reindex(weekday_order)
    fig = go.Figure(go.Heatmap(z=pivot.values, x=[date.strftime("%d %b") for date in pivot.columns], y=pivot.index, colorscale=[[0, "#221a3e"], [0.5, COLORS["violet"]], [1, COLORS["cyan"]]], zmin=0.45, zmax=1, hovertemplate="%{y}, %{x}<br>Confidence: %{z:.1%}<extra></extra>", showscale=False))
    _layout(fig, 230)
    fig.update_layout(xaxis={"showticklabels": False, "showgrid": False}, yaxis={"showgrid": False})
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})


def render_distribution_donut(data: pd.DataFrame) -> None:
    """Render model distribution by category."""
    counts = data["category"].value_counts()
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.68, marker={"colors": [COLORS["cyan"], COLORS["blue"], COLORS["violet"], COLORS["green"], COLORS["yellow"], COLORS["red"]]}, textinfo="label+percent", textfont={"size": 9}, hovertemplate="%{label}: %{value}<extra></extra>"))
    _layout(fig, 260)
    fig.update_layout(showlegend=False, annotations=[{"text": f"<b>{len(data)}</b><br><span style='font-size:10px'>models</span>", "showarrow": False, "font": {"color": COLORS["text"], "size": 16}}])
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": False})