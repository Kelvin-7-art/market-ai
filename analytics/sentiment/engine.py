"""News sentiment adapters and data preparation for the Sentiment workbench."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from market_ai.analytics.models.classification.random_forest.pipeline import (
    BASE_FEATURES,
    RandomForestBaseModel,
    to_chart_frame,
)


VARIANTS = ["Transformer (FinBERT)", "Traditional ML", "Ensemble"]
POSITIVE_WORDS = {
    "beat", "beats", "bullish", "buy", "growth", "improve", "improved", "innovative",
    "outperform", "positive", "profit", "profits", "record", "strong", "surge", "upside",
}
NEGATIVE_WORDS = {
    "bearish", "cut", "cuts", "decline", "downgrade", "drop", "fraud", "loss", "losses",
    "miss", "misses", "negative", "risk", "slump", "weak", "warning", "downside",
}
def fetch_news(ticker: str, limit: int = 30) -> tuple[pd.DataFrame, str]:
    """Fetch recent Yahoo Finance news and normalize its nested response shape."""
    source = "Yahoo Finance news"
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception:
        raw_items = []
    if not raw_items:
        try:
            raw_items = yf.Search(f"{ticker} stock", news_count=limit).news or []
            source = "Yahoo Finance Search"
        except Exception as error:
            return _empty_news(ticker), f"Yahoo Finance news unavailable: {error}"
    rows: list[dict[str, Any]] = []
    for item in raw_items[:limit]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title") or item.get("title")
        if not title:
            continue
        provider = content.get("provider") or content.get("publisher", item.get("publisher", {}))
        if isinstance(provider, dict):
            provider = provider.get("displayName") or provider.get("name")
        canonical = content.get("canonicalUrl") or content.get("link") or item.get("link", {})
        if isinstance(canonical, dict):
            canonical = canonical.get("url")
        published = content.get("pubDate") or content.get("providerPublishTime") or item.get("providerPublishTime")
        if isinstance(published, (int, float)):
            published = pd.to_datetime(published, unit="s", errors="coerce")
        else:
            published = pd.to_datetime(published, errors="coerce")
        rows.append(
            {
                "title": str(title),
                "source": str(provider or source),
                "published": published,
                "url": str(canonical or ""),
            }
        )
    if not rows:
        return _empty_news(ticker), f"No recent Yahoo Finance headlines were returned for {ticker}."
    frame = pd.DataFrame(rows)
    frame["published"] = (
        pd.to_datetime(frame["published"], errors="coerce", utc=True)
        .fillna(pd.Timestamp.now(tz="UTC"))
        .dt.tz_convert(None)
    )
    frame["date"] = frame["published"].dt.normalize()
    frame["ticker"] = ticker.upper()
    return frame, source


def _empty_news(ticker: str) -> pd.DataFrame:
    """Return an empty but schema-stable frame; never invent news headlines."""
    return pd.DataFrame(columns=["title", "source", "published", "date", "url", "ticker"])


def _lexicon_score(text: str) -> float:
    words = {word.strip(".,:;!?()[]{}").lower() for word in text.split()}
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    return float(np.clip((positive - negative) / max(positive + negative, 1), -1, 1))


@lru_cache(maxsize=1)
def _traditional_classifier() -> tuple[TfidfVectorizer, LogisticRegression]:
    examples = [
        ("earnings beat expectations with strong revenue growth", 1),
        ("company reports record profit and raises outlook", 1),
        ("shares surge as demand remains resilient", 1),
        ("analysts see upside from improving margins", 1),
        ("revenue misses estimates and guidance is cut", 0),
        ("shares slump after a weak earnings update", 0),
        ("regulators warn of risk and possible losses", 0),
        ("downgrade raises concern about slowing demand", 0),
        ("investors hold shares ahead of results", 1),
        ("market waits for the next quarterly report", 0),
    ]
    texts, labels = zip(*examples)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    matrix = vectorizer.fit_transform(texts)
    classifier = LogisticRegression(random_state=42, max_iter=500)
    classifier.fit(matrix, labels)
    return vectorizer, classifier


def _traditional_score(text: str) -> tuple[float, float]:
    vader = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    blob = float(TextBlob(text).sentiment.polarity)
    vectorizer, classifier = _traditional_classifier()
    probability = float(classifier.predict_proba(vectorizer.transform([text]))[0, 1])
    logistic = (probability - 0.5) * 2
    score = float(np.clip(np.mean([vader, blob, logistic, _lexicon_score(text)]), -1, 1))
    confidence = float(np.clip(0.55 + abs(score) * 0.4, 0, 0.99))
    return score, confidence


@lru_cache(maxsize=1)
def _finbert_pipeline() -> tuple[Any | None, str]:
    """Load FinBERT only when requested; keep the page usable without torch."""
    try:
        from transformers import pipeline

        return pipeline("sentiment-analysis", model="ProsusAI/finbert", top_k=None), "FinBERT"
    except Exception as error:
        return None, f"FinBERT unavailable ({type(error).__name__}); finance lexicon fallback"


def _finbert_score(text: str) -> tuple[float, float, str]:
    classifier, status = _finbert_pipeline()
    if classifier is None:
        score = _lexicon_score(text)
        return score, float(np.clip(0.55 + abs(score) * 0.35, 0, 0.9)), status
    try:
        outputs = classifier(text[:1000], truncation=True)
        if outputs and isinstance(outputs[0], list):
            outputs = outputs[0]
        scores = {str(item["label"]).lower(): float(item["score"]) for item in outputs}
        score = scores.get("positive", 0.0) - scores.get("negative", 0.0)
        confidence = max(scores.values()) if scores else 0.5
        return float(np.clip(score, -1, 1)), float(confidence), status
    except Exception as error:
        return _lexicon_score(text), 0.5, f"FinBERT inference failed ({type(error).__name__}); fallback used"


def score_headlines(frame: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, str]:
    """Score headlines with the requested adapter and return normalized outputs."""
    result = frame.copy()
    scores: list[float] = []
    confidences: list[float] = []
    statuses: list[str] = []
    for title in result["title"].astype(str):
        if variant == "Traditional ML":
            score, confidence = _traditional_score(title)
            status = "VADER + TextBlob + Logistic Regression"
        elif variant == "Ensemble":
            traditional, traditional_confidence = _traditional_score(title)
            finbert, finbert_confidence, finbert_status = _finbert_score(title)
            score, confidence = (traditional + finbert) / 2, (traditional_confidence + finbert_confidence) / 2
            status = f"Ensemble: {finbert_status} + Traditional ML"
        else:
            score, confidence, status = _finbert_score(title)
        scores.append(float(np.clip(score, -1, 1)))
        confidences.append(float(np.clip(confidence, 0, 1)))
        statuses.append(status)
    result["score"] = scores
    result["confidence"] = confidences
    result["sentiment"] = np.select(
        [result["score"] >= 0.15, result["score"] <= -0.15],
        ["Positive", "Negative"],
        default="Neutral",
    )
    result["status"] = statuses
    return result, statuses[0] if statuses else "No sentiment adapter output"


def aggregate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headline scores by calendar day."""
    daily = (
        frame.groupby("date", as_index=False)
        .agg(
            sentiment_score=("score", "mean"),
            confidence=("confidence", "mean"),
            headlines=("title", "count"),
            positive=("sentiment", lambda values: int((values == "Positive").sum())),
            negative=("sentiment", lambda values: int((values == "Negative").sum())),
        )
        .sort_values("date")
    )
    daily["sentiment"] = np.select(
        [daily["sentiment_score"] >= 0.15, daily["sentiment_score"] <= -0.15],
        ["Positive", "Negative"],
        default="Neutral",
    )
    return daily


def download_price_history(ticker: str, lookback_days: int = 365) -> pd.DataFrame:
    """Download daily OHLCV history using the same loader as Random Forest."""
    end = pd.Timestamp.now().normalize()
    model = RandomForestBaseModel(ticker=ticker)
    return to_chart_frame(
        model.load_data(
            start_date=(end - pd.Timedelta(days=lookback_days)).date(),
            end_date=end.date(),
            interval="1d",
        )
    )


def correlate_with_returns(daily: pd.DataFrame, prices: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Join sentiment with next-session returns and calculate Pearson correlation."""
    returns = prices[["date", "close"]].copy()
    returns["return"] = returns["close"].pct_change()
    merged = daily.merge(returns[["date", "return"]], on="date", how="inner").dropna()
    correlation = float(merged["sentiment_score"].corr(merged["return"])) if len(merged) >= 3 else 0.0
    return merged, correlation if np.isfinite(correlation) else 0.0


def train_sentiment_forecast(
    ticker: str,
    sentiment_score: float,
    horizon: int = 90,
    impact_strength: float = 0.08,
) -> tuple[RandomForestBaseModel, pd.DataFrame]:
    """Train the existing Random Forest pipeline and apply a sentiment bias."""
    model = RandomForestBaseModel(
        ticker=ticker,
        n_estimators=75,
        random_state=42,
        test_size=0.25,
        forecast_days=horizon,
        features=BASE_FEATURES,
    )
    model.load_data(
        start_date=(pd.Timestamp.now() - pd.Timedelta(days=900)).date(),
        end_date=pd.Timestamp.now().date(),
        interval="1d",
    )
    model.engineer_features()
    model.prepare_data()
    model.train()
    model.evaluate()
    base = model.generate_forecast(horizon).copy()
    strength = float(np.clip(sentiment_score, -1, 1))
    ramp = 1 + strength * float(np.clip(impact_strength, 0, 0.15)) * (1 - np.exp(-base["Day"].to_numpy() / 24))
    base["Base_Forecast_Price"] = base["Forecast_Price"]
    base["Sentiment_Adjusted_Price"] = base["Forecast_Price"] * ramp
    base["Sentiment_Bias"] = ramp - 1
    return model, base