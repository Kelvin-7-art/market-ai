"""Financial news sentiment workbench for Analytics."""

from __future__ import annotations

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from market_ai.analytics.market_catalog import catalog_name, market_groups, tickers_for
from market_ai.analytics.sentiment.engine import (
    VARIANTS,
    aggregate_daily,
    correlate_with_returns,
    download_price_history,
    fetch_news,
    score_headlines,
    train_sentiment_forecast,
)
from market_ai.analytics.models.variant import render_model_navigation
from market_ai.components.ui.cards import stat_card
from market_ai.components.ui.feedback import toast_success
from market_ai.utils.theme import COLORS


def _ensure_sentiment_state() -> None:
    """Initialize page state without overwriting any existing session choices."""
    defaults = {
        "sent_market": "US equities",
        "sent_catalog_ticker": "AAPL",
        "sent_custom_override": False,
        "sent_custom_ticker": "",
        "sent_variant": VARIANTS[0],
        "sent_news_raw": pd.DataFrame(),
        "sent_news_source_status": "",
        "sent_price_history": pd.DataFrame(),
        "sent_results": {},
        "sent_model_status": "",
        "sent_news_error": "",
        "sent_price_error": "",
        "sent_daily": pd.DataFrame(),
        "sent_correlation_data": pd.DataFrame(),
        "sent_correlation": 0.0,
        "sent_forecast_df": pd.DataFrame(),
        "sent_forecast_horizon": 0,
        "sent_forecast_model": None,
        "sent_forecast_error": "",
        "sent_impact_strength": 0.08,
        "sent_impact_percent": 8,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _market_controls() -> str:
    """Render the shared market selector with a custom ticker override."""
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title"><span>🌐</span><div><h3>Market & News</h3><p>Select a market and ticker, then fetch current financial headlines.</p></div></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns([1.1, 1.4, 1.1, 1.0])
    market = columns[0].selectbox("Market", market_groups(), key="sent_market")
    entries = tickers_for(market)
    symbols = [entry["ticker"] for entry in entries]
    if st.session_state.get("sent_catalog_ticker") not in symbols:
        st.session_state.sent_catalog_ticker = symbols[0]
    selected = columns[1].selectbox(
        "Ticker",
        symbols,
        format_func=lambda symbol: f"{symbol} · {catalog_name(symbol)}",
        key="sent_catalog_ticker",
    )
    override = columns[2].checkbox("Custom ticker override", key="sent_custom_override")
    if override:
        custom = columns[3].text_input(
            "Custom ticker",
            value=st.session_state.get("sent_custom_ticker", ""),
            placeholder="e.g. BRK-B or ^GSPC",
            key="sent_custom_ticker",
        )
        ticker = custom.strip().upper() or selected
    else:
        columns[3].caption("Ticker source")
        columns[3].markdown(f"**{html.escape(selected)}**")
        ticker = selected
    fetch_col, limit_col, range_col = st.columns([1.2, 0.8, 1.0])
    limit = limit_col.selectbox("Headline limit", [10, 20, 30, 50], index=2, key="sent_headline_limit")
    lookback = range_col.selectbox("Price history", ["6 months", "1 year", "2 years"], index=1, key="sent_price_lookback")
    pressed = fetch_col.button(
        "Fetch News & Analyze",
        type="primary",
        use_container_width=True,
        key="sent_fetch_analyze",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    if pressed and ticker:
        _fetch_and_analyze(ticker, int(limit), lookback, st.session_state.sent_variant)
    return ticker


def _fetch_and_analyze(ticker: str, limit: int, lookback: str, variant: str) -> None:
    """Fetch headlines/prices once and score for the selected variant."""
    days = {"6 months": 183, "1 year": 365, "2 years": 730}[lookback]
    st.session_state.sent_news_error = ""
    st.session_state.sent_price_error = ""
    st.session_state.sent_forecast_error = ""
    st.session_state.sent_forecast_df = pd.DataFrame()
    with st.spinner(f"Fetching {ticker} headlines and market history…"):
        news, source_status = fetch_news(ticker, limit)
        st.session_state.sent_news_raw = news
        st.session_state.sent_news_source_status = source_status
        st.session_state.sent_active_ticker = ticker
        try:
            prices = download_price_history(ticker, days)
            st.session_state.sent_price_history = prices
        except Exception as error:
            st.session_state.sent_price_history = pd.DataFrame()
            st.session_state.sent_price_error = str(error)
        st.session_state.sent_results = {}
        if not news.empty:
            scored, status = score_headlines(news, variant)
            st.session_state.sent_results = {variant: scored}
            st.session_state.sent_model_status = status
            daily = aggregate_daily(scored)
            st.session_state.sent_daily = daily
            if not st.session_state.sent_price_history.empty:
                joined, correlation = correlate_with_returns(daily, st.session_state.sent_price_history)
                st.session_state.sent_correlation_data = joined
                st.session_state.sent_correlation = correlation
            else:
                st.session_state.sent_correlation_data = pd.DataFrame()
                st.session_state.sent_correlation = 0.0
        else:
            st.session_state.sent_daily = pd.DataFrame()
            st.session_state.sent_correlation_data = pd.DataFrame()
            st.session_state.sent_correlation = 0.0
            st.session_state.sent_model_status = ""
    if not news.empty:
        toast_success(f"Analyzed {len(news)} headlines for {ticker}")
    else:
        st.warning(source_status)


def _active_result(variant: str) -> pd.DataFrame:
    """Return cached active-variant scores, calculating them after a variant switch."""
    results = st.session_state.sent_results
    if variant not in results and not st.session_state.sent_news_raw.empty:
        with st.spinner(f"Scoring headlines with {variant}…"):
            try:
                scored, status = score_headlines(st.session_state.sent_news_raw, variant)
                results[variant] = scored
                st.session_state.sent_results = results
                st.session_state.sent_model_status = status
            except Exception as error:
                st.session_state.sent_news_error = str(error)
                return pd.DataFrame()
    result = results.get(variant, pd.DataFrame())
    if not result.empty:
        st.session_state.sent_model_status = str(result["status"].iloc[0])
        daily = aggregate_daily(result)
        st.session_state.sent_daily = daily
        prices = st.session_state.sent_price_history
        if not prices.empty:
            joined, correlation = correlate_with_returns(daily, prices)
            st.session_state.sent_correlation_data = joined
            st.session_state.sent_correlation = correlation
        else:
            st.session_state.sent_correlation_data = pd.DataFrame()
            st.session_state.sent_correlation = 0.0
    return result


def _summary_cards(headlines: pd.DataFrame, ticker: str, variant: str) -> None:
    """Render headline count, aggregate tone, confidence, and correlation."""
    if headlines.empty:
        values = ("0", "—", "—", "—")
        tones = ("No headlines loaded", "Awaiting analysis", "Awaiting analysis", "Need price overlap")
        points = ([0, 0, 0],) * 4
    else:
        mean_score = float(headlines["score"].mean())
        positive_share = float((headlines["sentiment"] == "Positive").mean())
        confidence = float(headlines["confidence"].mean())
        correlation = float(st.session_state.sent_correlation)
        correlation_value = (
            f"{correlation:+.2f}"
            if len(st.session_state.sent_correlation_data) >= 3
            else "—"
        )
        tone = "Positive" if mean_score >= 0.15 else "Negative" if mean_score <= -0.15 else "Neutral"
        values = (f"{len(headlines):,}", f"{mean_score:+.2f}", f"{confidence:.0%}", correlation_value)
        tones = (f"{ticker} · {variant}", f"{tone} · {positive_share:.0%} positive", "Mean model confidence", "Sentiment vs. daily return")
        points = (
            [1, max(1, len(headlines) * 0.4), max(1, len(headlines) * 0.8), len(headlines)],
            [0, mean_score * 0.4, mean_score * 0.8, mean_score],
            [0.5, confidence * 0.7, confidence * 0.9, confidence],
            [0, correlation * 0.4, correlation * 0.8, correlation],
        )
    columns = st.columns(4)
    labels = ["Headlines", "Mean sentiment", "Mean confidence", "Return correlation"]
    icons = ["📰", "◉", "◎", "↗"]
    for column, label, icon, value, delta, spark in zip(columns, labels, icons, values, tones, points):
        with column:
            stat_card(label, value, delta, icon, spark, footer="Current session")


def _headline_section(headlines: pd.DataFrame) -> None:
    """Render per-headline scores in a table and a time-ordered score chart."""
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title"><span>📰</span><div><h3>Headline Sentiment</h3><p>Each news item is scored independently before daily aggregation.</p></div></div>',
        unsafe_allow_html=True,
    )
    if headlines.empty:
        st.info("Fetch news to score headlines for the selected ticker.")
    else:
        chart = go.Figure()
        color_map = {"Positive": COLORS["green"], "Negative": COLORS["red"], "Neutral": COLORS["yellow"]}
        for label in ["Positive", "Neutral", "Negative"]:
            subset = headlines[headlines["sentiment"] == label]
            if not subset.empty:
                chart.add_trace(
                    go.Scatter(
                        x=subset["published"],
                        y=subset["score"],
                        mode="markers",
                        name=label,
                        marker={"size": 10, "color": color_map[label], "line": {"width": 1, "color": COLORS["panel"]}},
                        text=subset["title"],
                        customdata=np.c_[subset["source"], subset["confidence"]],
                        hovertemplate="%{x|%d %b %Y %H:%M}<br>%{text}<br>Score: %{y:+.2f}<br>Source: %{customdata[0]}<br>Confidence: %{customdata[1]:.0%}<extra></extra>",
                    )
                )
        chart.add_hline(y=0, line={"color": COLORS["muted"], "dash": "dot", "width": 1})
        chart.update_layout(
            height=300,
            margin={"l": 12, "r": 12, "t": 16, "b": 8},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["muted"], "size": 10},
            yaxis={"title": "Sentiment score", "range": [-1.05, 1.05], "gridcolor": COLORS["grid"]},
            xaxis={"title": "Publication time", "gridcolor": COLORS["grid"]},
            legend={"orientation": "h", "y": 1.08},
            hovermode="closest",
        )
        st.plotly_chart(chart, use_container_width=True, config={"displaylogo": False})
        display = headlines.sort_values("published", ascending=False).copy()
        display["Published"] = display["published"].dt.strftime("%Y-%m-%d %H:%M")
        display["Sentiment score"] = display["score"].map(lambda score: f"{score:+.3f}")
        display["Confidence"] = display["confidence"].map(lambda confidence: f"{confidence:.0%}")
        display["Headline tone"] = display["sentiment"]
        st.dataframe(
            display[["Published", "source", "Headline tone", "Sentiment score", "Confidence", "title"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "source": "Source",
                "title": st.column_config.TextColumn("Headline", width="large"),
            },
        )
        st.download_button(
            "Download headline sentiment CSV",
            headlines.to_csv(index=False),
            f'{st.session_state.get("sent_active_ticker", "market")}_sentiment_headlines.csv',
            "text/csv",
            key="sent_download_headlines",
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _daily_and_correlation(daily: pd.DataFrame) -> None:
    """Render daily sentiment and its relationship with realized returns."""
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title"><span>📊</span><div><h3>Daily Aggregation & Return Correlation</h3><p>Daily mean sentiment is compared with matched market returns.</p></div></div>',
        unsafe_allow_html=True,
    )
    if daily.empty:
        st.info("Daily aggregation and return correlation will appear after headline analysis.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    daily_col, corr_col = st.columns([1.1, 1], gap="large")
    with daily_col:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=daily["date"],
                y=daily["sentiment_score"],
                name="Mean sentiment",
                marker_color=[
                    COLORS["green"] if value >= 0.15 else COLORS["red"] if value <= -0.15 else COLORS["yellow"]
                    for value in daily["sentiment_score"]
                ],
                customdata=daily[["headlines", "confidence"]],
                hovertemplate="%{x|%d %b %Y}<br>Mean score: %{y:+.2f}<br>Headlines: %{customdata[0]}<br>Confidence: %{customdata[1]:.0%}<extra></extra>",
            )
        )
        fig.add_hline(y=0, line={"color": COLORS["muted"], "dash": "dot"})
        fig.update_layout(
            title="Daily headline sentiment",
            height=300,
            margin={"l": 5, "r": 8, "t": 40, "b": 8},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": COLORS["muted"], "size": 10},
            xaxis={"gridcolor": COLORS["grid"]},
            yaxis={"title": "Average score", "range": [-1.05, 1.05], "gridcolor": COLORS["grid"]},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        daily_display = daily.copy()
        daily_display["Sentiment score"] = daily_display["sentiment_score"].map(lambda score: f"{score:+.3f}")
        daily_display["Mean confidence"] = daily_display["confidence"].map(lambda confidence: f"{confidence:.0%}")
        st.dataframe(
            daily_display[["date", "headlines", "positive", "negative", "sentiment", "Sentiment score", "Mean confidence"]],
            hide_index=True,
            use_container_width=True,
        )
    with corr_col:
        correlation_data = st.session_state.sent_correlation_data
        correlation = float(st.session_state.sent_correlation)
        st.markdown("**Sentiment vs. daily return**")
        if len(correlation_data) < 3:
            st.info("At least three overlapping sentiment and price sessions are needed to estimate correlation.")
        else:
            scatter = go.Figure(
                go.Scatter(
                    x=correlation_data["sentiment_score"],
                    y=correlation_data["return"] * 100,
                    mode="markers",
                    marker={"size": 10, "color": COLORS["cyan"], "opacity": 0.82},
                    text=correlation_data["date"].dt.strftime("%Y-%m-%d"),
                    customdata=correlation_data["headlines"],
                    hovertemplate="%{text}<br>Sentiment: %{x:+.2f}<br>Daily return: %{y:+.2f}%<br>Headlines: %{customdata}<extra></extra>",
                )
            )
            scatter.add_hline(y=0, line={"color": COLORS["muted"], "dash": "dot"})
            scatter.add_vline(x=0, line={"color": COLORS["muted"], "dash": "dot"})
            scatter.update_layout(
                title=f"Pearson r = {correlation:+.3f}",
                height=300,
                margin={"l": 8, "r": 8, "t": 40, "b": 8},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["muted"], "size": 10},
                xaxis={"title": "Daily sentiment score", "range": [-1.05, 1.05], "gridcolor": COLORS["grid"]},
                yaxis={"title": "Daily return (%)", "gridcolor": COLORS["grid"]},
            )
            st.plotly_chart(scatter, use_container_width=True, config={"displaylogo": False})
            interpretation = "positive co-movement" if correlation > 0.15 else "negative co-movement" if correlation < -0.15 else "little linear co-movement"
            st.caption(f"Observed {interpretation} across {len(correlation_data)} matched sessions. Correlation is descriptive, not causal.")
    st.markdown("</div>", unsafe_allow_html=True)


def _forecast_section(ticker: str, headlines: pd.DataFrame) -> None:
    """Train the existing Random Forest and display a sentiment-adjusted forecast."""
    st.markdown(
        '<div class="workbench-section"><div class="workbench-section-title"><span>🔮</span><div><h3>Sentiment-Augmented Price Forecast</h3><p>Random Forest price path with a transparent headline-sentiment overlay.</p></div></div>',
        unsafe_allow_html=True,
    )
    if headlines.empty:
        st.info("Analyze headlines before running a sentiment-augmented forecast.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    mean_sentiment = float(headlines["score"].mean())
    control_cols = st.columns([0.8, 1.4, 1.2])
    horizon = control_cols[0].selectbox("Forecast horizon", [30, 60, 90, 180], index=2, key="sent_forecast_days")
    impact_percent = control_cols[1].slider(
        "Maximum sentiment adjustment",
        min_value=0,
        max_value=15,
        value=int(st.session_state.sent_impact_percent),
        step=1,
        format="%d%%",
        key="sent_impact_percent",
    )
    impact = impact_percent / 100
    st.session_state.sent_impact_strength = impact
    control_cols[2].metric("Current mean sentiment", f"{mean_sentiment:+.2f}")
    run = st.button("Run sentiment-augmented forecast", type="primary", key="sent_run_forecast")
    if run:
        with st.spinner(f"Training Random Forest and forecasting {ticker}…"):
            try:
                model, forecast = train_sentiment_forecast(
                    ticker,
                    mean_sentiment,
                    int(horizon),
                    impact_strength=float(impact),
                )
                st.session_state.sent_forecast_model = model
                st.session_state.sent_forecast_df = forecast
                st.session_state.sent_forecast_horizon = int(horizon)
                st.session_state.sent_forecast_error = ""
                toast_success("Sentiment-augmented forecast is ready")
            except Exception as error:
                st.session_state.sent_forecast_error = str(error)
                st.error(f"Forecast failed: {error}")
    if st.session_state.sent_forecast_error and not run:
        st.error(f"Forecast failed: {st.session_state.sent_forecast_error}")
    forecast = st.session_state.sent_forecast_df
    model = st.session_state.sent_forecast_model
    if not forecast.empty and model is not None:
        history = model.raw_data
        if history is not None:
            history = history.tail(100)
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=history["Date"],
                    y=history[model.price_col],
                    name="Historical close",
                    line={"color": COLORS["cyan"], "width": 2},
                )
            )
            future_dates = pd.DatetimeIndex(pd.to_datetime(forecast["Date"]))
            last_date = pd.Timestamp(history["Date"].iloc[-1])
            forecast_dates = pd.DatetimeIndex([last_date]).append(future_dates)
            base_prices = np.r_[float(history[model.price_col].iloc[-1]), forecast["Base_Forecast_Price"].to_numpy()]
            adjusted_prices = np.r_[float(history[model.price_col].iloc[-1]), forecast["Sentiment_Adjusted_Price"].to_numpy()]
            lower = np.r_[float(history[model.price_col].iloc[-1]), forecast["Confidence_Lower"].to_numpy() * (1 + forecast["Sentiment_Bias"].to_numpy())]
            upper = np.r_[float(history[model.price_col].iloc[-1]), forecast["Confidence_Upper"].to_numpy() * (1 + forecast["Sentiment_Bias"].to_numpy())]
            fig.add_trace(go.Scatter(x=forecast_dates, y=upper, name="Adjusted interval upper", line={"width": 0}, showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=forecast_dates, y=lower, name="Sentiment-adjusted 95% interval", fill="tonexty", line={"width": 0}, fillcolor="rgba(156,107,255,0.15)"))
            fig.add_trace(go.Scatter(x=forecast_dates, y=base_prices, name="Random Forest base forecast", line={"color": COLORS["yellow"], "dash": "dot", "width": 2}))
            fig.add_trace(go.Scatter(x=forecast_dates, y=adjusted_prices, name="Sentiment-adjusted forecast", line={"color": COLORS["violet"], "width": 2.5}))
            fig.add_vline(x=last_date, line={"color": COLORS["muted"], "dash": "dash"})
            fig.update_layout(
                title=f"{ticker} · {st.session_state.sent_forecast_horizon}-session forecast",
                height=420,
                margin={"l": 20, "r": 20, "t": 55, "b": 20},
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["muted"], "size": 10},
                hovermode="x unified",
                xaxis={"gridcolor": COLORS["grid"]},
                yaxis={"title": "Price", "gridcolor": COLORS["grid"]},
                legend={"orientation": "h", "y": 1.08},
            )
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "displayModeBar": True})
            start = float(history[model.price_col].iloc[-1])
            base_end = float(forecast["Base_Forecast_Price"].iloc[-1])
            adjusted_end = float(forecast["Sentiment_Adjusted_Price"].iloc[-1])
            metric_cols = st.columns(4)
            metric_cols[0].metric("Starting price", f"{start:,.2f}")
            metric_cols[1].metric("Base RF final", f"{base_end:,.2f}", f"{base_end / start - 1:+.1%}")
            metric_cols[2].metric("Sentiment-adjusted final", f"{adjusted_end:,.2f}", f"{adjusted_end / start - 1:+.1%}")
            metric_cols[3].metric("Sentiment overlay", f"{adjusted_end - base_end:+,.2f}")
            st.caption(
                "The sentiment overlay adjusts the Random Forest path by up to "
                f"{impact:.0%} using the current mean headline score ({mean_sentiment:+.2f}). "
                "It is a transparent scenario adjustment, not a causal estimate or investment advice."
            )
            st.download_button(
                "Download forecast CSV",
                forecast.to_csv(index=False),
                f"{ticker}_sentiment_forecast.csv",
                "text/csv",
                key="sent_download_forecast",
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_sentiment() -> None:
    """Render Analytics → Sentiment with Transformer, baseline, and ensemble variants."""
    _ensure_sentiment_state()
    render_model_navigation("Sentiment", "")
    variant = st.session_state.get("sent_variant", VARIANTS[0])
    if variant not in VARIANTS:
        variant = VARIANTS[0]
        st.session_state.sent_variant = variant
    ticker = _market_controls()
    active_ticker = st.session_state.get("sent_active_ticker", ticker)
    result = _active_result(variant)
    status = st.session_state.get("sent_model_status", "")

    st.markdown(
        f'<div class="model-route-breadcrumb">Analytics <span>/</span> <strong>Sentiment</strong> <span>/</span> {html.escape(variant)}</div>',
        unsafe_allow_html=True,
    )
    title_col, badge_col = st.columns([1.6, 0.9])
    with title_col:
        st.markdown(
            f'<div class="page-kicker">FINANCIAL NEWS · {html.escape(variant.upper())}</div>'
            f'<h2>📰 {html.escape(variant)} Sentiment</h2>'
            f'<p class="page-subtitle">Analyze financial headlines for {html.escape(active_ticker)} and connect daily news tone to market returns.</p>',
            unsafe_allow_html=True,
        )
    with badge_col:
        status_label = "FinBERT ready" if status == "FinBERT" else "Baseline active" if variant == "Traditional ML" else "Fallback active" if status else "Ready to analyze"
        tone = "active" if status == "FinBERT" or variant == "Traditional ML" else "info"
        st.markdown(
            '<div class="workbench-badges">'
            f'<span class="badge badge-{tone}">● {html.escape(status_label)}</span>'
            f'<span class="badge badge-info">{html.escape(active_ticker)}</span>'
            f'<span class="badge badge-info">{len(result)} headlines</span>'
            "</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.sent_news_source_status and st.session_state.sent_news_raw.empty:
        st.warning(st.session_state.sent_news_source_status)
    if st.session_state.sent_price_error:
        st.warning(f"Price history unavailable: {st.session_state.sent_price_error}")
    if st.session_state.sent_news_error:
        st.error(st.session_state.sent_news_error)
    if variant == VARIANTS[0] and result.empty:
        st.info(
            "FinBERT is the primary sentiment model. It loads the ProsusAI/finbert transformer on first use; "
            "if the optional Transformers runtime or model download is unavailable, the page will report and use an explicit finance-lexicon fallback."
        )
    if status and variant != "Traditional ML" and "unavailable" in status.lower():
        st.warning(status)
    elif status and variant != "Traditional ML" and "failed" in status.lower():
        st.warning(status)
    if variant == "Traditional ML":
        st.caption("Baseline stack: VADER compound score + TextBlob polarity + TF-IDF Logistic Regression.")
    elif variant == "Ensemble":
        st.caption("Blended output averages the FinBERT sentiment score and the Traditional ML score.")

    _summary_cards(result, active_ticker, variant)
    _headline_section(result)
    _daily_and_correlation(st.session_state.sent_daily if not result.empty else pd.DataFrame())
    _forecast_section(active_ticker, result)

    with st.expander("Model details & data source"):
        st.json(
            {
                "active_variant": variant,
                "primary_model": "ProsusAI/finbert" if variant == VARIANTS[0] else variant,
                "sentiment_range": "[-1, +1]",
                "market": st.session_state.sent_market,
                "ticker": active_ticker,
                "news_source": st.session_state.sent_news_source_status,
                "scoring_status": status,
                "correlation_observations": len(st.session_state.sent_correlation_data),
                "random_forest_forecast": not st.session_state.sent_forecast_df.empty,
            }
        )
    st.warning("News sentiment and model forecasts are informational only. They are not investment advice.")


__all__ = ["render_sentiment"]
