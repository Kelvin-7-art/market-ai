"""Deterministic sample data used by the frontend-only demo."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _rng(seed: int = 7) -> np.random.Generator:
    return np.random.default_rng(seed)


def market_snapshot() -> list[dict[str, object]]:
    """Return the watchlist snapshot shown in the right rail."""
    return [
        {"symbol": "S&P 500", "ticker": "SPX", "price": "5,648.40", "change": "+0.83%", "direction": "positive", "points": [12, 13, 12.5, 15, 16, 15.7, 18, 20, 21]},
        {"symbol": "NASDAQ", "ticker": "IXIC", "price": "17,683.98", "change": "+1.12%", "direction": "positive", "points": [10, 9, 11, 10.5, 13, 14, 16, 15.8, 18]},
        {"symbol": "Bitcoin", "ticker": "BTC/USD", "price": "64,284.12", "change": "-0.42%", "direction": "negative", "points": [18, 17, 18, 16, 15, 15.5, 14, 13, 12]},
        {"symbol": "Ethereum", "ticker": "ETH/USD", "price": "3,452.80", "change": "+2.06%", "direction": "positive", "points": [8, 9, 8.5, 11, 12, 12.4, 15, 16, 18]},
        {"symbol": "EUR / USD", "ticker": "EURUSD", "price": "1.1084", "change": "+0.18%", "direction": "positive", "points": [14, 14.4, 14.2, 15, 15.4, 15.1, 16, 16.8, 17]},
        {"symbol": "Gold", "ticker": "XAU", "price": "2,526.90", "change": "+0.55%", "direction": "positive", "points": [10, 11, 10.5, 12, 13, 13, 14, 15, 16]},
        {"symbol": "Crude Oil", "ticker": "WTI", "price": "75.88", "change": "-1.24%", "direction": "negative", "points": [18, 18, 17, 16, 17, 14, 14.5, 13, 12]},
    ]


def portfolio_data(periods: int = 252) -> pd.DataFrame:
    """Create a portfolio and benchmark series for the chart."""
    rng = _rng()
    dates = pd.date_range(end=datetime.now().date(), periods=periods, freq="D")
    portfolio_returns = rng.normal(0.00055, 0.008, periods)
    benchmark_returns = rng.normal(0.00035, 0.006, periods)
    portfolio = 100 * np.cumprod(1 + portfolio_returns)
    benchmark = 100 * np.cumprod(1 + benchmark_returns)
    return pd.DataFrame(
        {
            "date": dates,
            "Portfolio": portfolio,
            "Benchmark": benchmark,
            "daily_return": portfolio_returns,
        }
    )


def model_data() -> pd.DataFrame:
    """Return the model registry described in the brief."""
    rows = [
        ("ARIMA", "Time Series", "Active", 91.8, 92.1, 89.4, 90.7, 34, "12m ago", "Research"),
        ("Prophet", "Time Series", "Active", 89.6, 88.3, 91.2, 89.7, 18, "2d ago", "Research"),
        ("LSTM", "Neural", "Training", 94.8, 93.7, 94.1, 93.9, 248, "4m ago", "Quant Lab"),
        ("Transformer", "Neural", "Active", 96.2, 95.8, 94.9, 95.3, 412, "1d ago", "Quant Lab"),
        ("PatchTST", "Time Series", "Active", 95.4, 94.8, 95.2, 95.0, 380, "5h ago", "Signals"),
        ("Chronos", "Foundation", "Inactive", 93.9, 92.9, 93.3, 93.1, 520, "8d ago", "Research"),
        ("TimesFM", "Foundation", "Active", 94.1, 93.2, 94.6, 93.9, 460, "3d ago", "Signals"),
        ("XGBoost", "Tree-based", "Active", 92.7, 93.0, 91.6, 92.3, 72, "10h ago", "Risk"),
        ("LightGBM", "Tree-based", "Active", 91.9, 92.4, 90.8, 91.6, 54, "2d ago", "Risk"),
        ("CatBoost", "Tree-based", "Training", 90.8, 91.5, 89.7, 90.6, 82, "7m ago", "Risk"),
        ("FinBERT", "NLP", "Active", 88.6, 89.8, 87.6, 88.7, 180, "16h ago", "Sentiment"),
        ("PPO", "Reinforcement", "Active", 86.4, 87.2, 84.9, 86.0, 305, "4d ago", "Execution"),
    ]
    return pd.DataFrame(
        rows,
        columns=["name", "category", "status", "accuracy", "precision", "recall", "f1", "features_count", "last_trained", "owner"],
    )


def confidence_heatmap() -> pd.DataFrame:
    """Return a calendar-style grid of prediction confidence."""
    rng = _rng(12)
    dates = pd.date_range(end=datetime.now().date(), periods=140, freq="D")
    values = np.clip(rng.normal(0.83, 0.09, len(dates)), 0.48, 0.99)
    result = pd.DataFrame({"date": dates, "confidence": values})
    result["week"] = result["date"].dt.isocalendar().week.astype(int)
    result["weekday"] = result["date"].dt.day_name().str[:3]
    return result


def signal_distribution() -> pd.DataFrame:
    """Return demo signal counts by action."""
    return pd.DataFrame({"signal": ["Long", "Short", "Neutral"], "count": [42, 18, 67]})


def activity_items() -> list[dict[str, str]]:
    """Return recent research activity."""
    return [
        {"title": "Transformer v3 completed", "detail": "96.2% validation accuracy", "time": "8 min ago", "tone": "positive"},
        {"title": "Risk report generated", "detail": "Factor exposure within limits", "time": "41 min ago", "tone": "neutral"},
        {"title": "BTC/USD alert triggered", "detail": "Momentum crossed 70 RSI", "time": "2 hr ago", "tone": "warning"},
        {"title": "Dataset refreshed", "detail": "1.2m records indexed", "time": "4 hr ago", "tone": "neutral"},
    ]


def mover_items() -> list[dict[str, str]]:
    """Return top movers for the dashboard list."""
    return [
        {"ticker": "NVDA", "name": "NVIDIA Corp.", "change": "+4.82%", "tone": "positive"},
        {"ticker": "TSLA", "name": "Tesla Inc.", "change": "+3.17%", "tone": "positive"},
        {"ticker": "AAPL", "name": "Apple Inc.", "change": "+1.92%", "tone": "positive"},
        {"ticker": "XOM", "name": "Exxon Mobil", "change": "-2.08%", "tone": "negative"},
        {"ticker": "COIN", "name": "Coinbase", "change": "-3.44%", "tone": "negative"},
    ]