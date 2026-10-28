"""Card renderers for market snapshots and feature modules."""

from __future__ import annotations

import streamlit as st

from market_ai.utils.helpers import sparkline
from market_ai.utils.sample_data import market_snapshot


def render_market_snapshot() -> None:
    """Render the compact market snapshot rail."""
    st.markdown('<div class="panel-title"><h3>Market snapshot</h3><span>Live demo feed</span></div>', unsafe_allow_html=True)
    for item in market_snapshot():
        st.markdown(
            f"""
            <div class="mover-row">
              <div>{sparkline(item["points"], "#31D7EE")}<span class="ticker">{item["ticker"]}</span><span class="ticker-name">{item["symbol"]}</span></div>
              <div><div class="ticker" style="text-align:right">{item["price"]}</div><div class="move-value {item["direction"]}">{item["change"]}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero() -> None:
    """Render the primary assistant hero."""
    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">Institutional research workspace</div>
          <h2>AI Market Assistant</h2>
          <p>Conversational intelligence for financial markets, quantitative research, algorithmic trading, forecasting and portfolio analytics.</p>
          <div class="hero-meta">MODEL ROUTER · 18 ACTIVE MODELS · DATA LATENCY 118MS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_grid() -> None:
    """Render feature discovery cards."""
    features = [
        ("↗", "Time Series Forecasting", "Explore regime-aware predictions across your watchlists."),
        ("◇", "Deep Learning", "Compare neural architectures with reproducible validation."),
        ("▤", "News Intelligence", "Turn market-moving headlines into structured context."),
        ("◉", "FinBERT Sentiment", "Measure language signals across filings and media."),
        ("ϟ", "Trading Signals", "Surface high-conviction setups with explainable rules."),
        ("◎", "Reinforcement Learning", "Evaluate policy performance across market regimes."),
        ("▦", "Portfolio Analytics", "Understand attribution, risk and benchmark drift."),
        ("⌕", "Explainable AI", "Trace every prediction back to its key drivers."),
    ]
    st.markdown('<div class="panel-title"><h3>Research modules</h3><span>Explore the workspace</span></div>', unsafe_allow_html=True)
    columns = st.columns(4)
    for index, (icon, title, description) in enumerate(features):
        with columns[index % 4]:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{icon}</div><h4>{title}</h4><p>{description}</p></div>',
                unsafe_allow_html=True,
            )
        if index == 3:
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)