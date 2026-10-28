"""Pydantic request contract for analyst prompts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


FocusArea = Literal[
    "Model quality",
    "Feature drivers",
    "Forecast",
    "Risks",
    "Recommendations",
]


class MarketContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticker: str = "Unknown"
    market_label: str | None = None
    interval: str = "1d"
    start_date: str | None = None
    end_date: str | None = None
    last_close: float | None = None
    period_return: float | None = None


class TrainingContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: str | None = None
    preset: str | None = None
    cv_splits: int | None = None
    scoring: str | None = None
    best_params: dict | None = None
    parameters: dict | None = None
    training_time_seconds: float | None = None


class MetricsContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    best_cv_score: float | None = None
    skill_score: float | None = None
    low_skill: bool | None = None


class ForecastContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    available: bool = False
    horizon_days: int | None = None
    starting_price: float | None = None
    final_price: float | None = None
    total_return: float | None = None
    bullish_days: int | None = None
    bearish_days: int | None = None
    avg_bullish_probability: float | None = None


class ChatRequest(BaseModel):
    """Validated model context and user-selected interpretation settings."""

    model_config = ConfigDict(extra="ignore")

    task_type: Literal[
        "Forecast review",
        "Model diagnostic",
        "Feature analysis",
        "Risk assessment",
        "Custom question",
    ] = "Forecast review"
    tone: Literal["Analyst", "Executive", "Technical", "Casual"] = "Analyst"
    detail_level: Literal["Concise", "Standard", "Deep dive"] = "Standard"
    analysis_horizon_days: int | None = Field(default=None, ge=5, le=730)
    focus_areas: list[FocusArea] = Field(
        default_factory=lambda: [
            "Model quality",
            "Feature drivers",
            "Forecast",
            "Risks",
        ]
    )
    custom_question: str | None = None

    market: MarketContext = Field(default_factory=MarketContext)
    training: TrainingContext = Field(default_factory=TrainingContext)
    metrics: MetricsContext = Field(default_factory=MetricsContext)
    forecast: ForecastContext = Field(default_factory=ForecastContext)
    feature_importance_top: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    schema_version: str = "1.0"
