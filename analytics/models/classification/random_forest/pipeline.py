"""Yahoo Finance and scikit-learn pipeline for the Random Forest base model.

The Streamlit page owns presentation.  This module owns the model lifecycle
from the supplied implementation: download, engineer, prepare, train,
evaluate, and recursively forecast.
"""

from __future__ import annotations
from market_ai.utils.cloud_config import setting


from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = [
    "Return",
    "MA20",
    "MA50",
    "Volatility",
    "Price_MA20_Ratio",
    "Price_MA50_Ratio",
    "Volume_Ratio",
    "High_Low_Ratio",
    "Close_Open_Ratio",
    "Momentum_5",
    "Momentum_10",
]


def _flatten_yahoo_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance's single- and multi-ticker column formats."""
    result = frame.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            next((str(value) for value in column if str(value).lower() in {
                "open", "high", "low", "close", "adj close", "volume"
            }), str(column[0]))
            for column in result.columns
        ]
    rename: dict[str, str] = {}
    for column in result.columns:
        name = str(column).strip().lower().replace("_", " ")
        if name == "adj close":
            rename[column] = "Adj_Close"
        elif name == "close":
            rename[column] = "Close"
        elif name == "open":
            rename[column] = "Open"
        elif name == "high":
            rename[column] = "High"
        elif name == "low":
            rename[column] = "Low"
        elif name == "volume":
            rename[column] = "Volume"
    result = result.rename(columns=rename)
    if "Date" not in result.columns:
        index_name = str(result.index.name or "Date")
        result = result.reset_index().rename(columns={index_name: "Date"})
    result["Date"] = pd.to_datetime(result["Date"]).dt.tz_localize(None).dt.normalize()
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(f"Yahoo Finance data is missing required columns: {', '.join(missing)}")
    result = result.dropna(subset=required).sort_values("Date").reset_index(drop=True)
    for column in required + (["Adj_Close"] if "Adj_Close" in result.columns else []):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=required)


def to_chart_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert model data to the lowercase schema used by the Streamlit charts."""
    return frame.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj_Close": "adj_close",
        }
    )


class RandomForestBaseModel:
    """Baseline Random Forest classifier for stock-trend prediction."""

    def __init__(
        self,
        ticker: str = "AAPL",
        n_estimators: int = 100,
        random_state: int = 42,
        test_size: float = 0.25,
        forecast_days: int = 300,
        max_depth: int | None = None,
        class_weight: str | None = "balanced",
        features: list[str] | None = None,
    ) -> None:
        self.ticker = ticker.upper().strip()
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.test_size = float(test_size)
        self.forecast_days = int(forecast_days)
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.data: pd.DataFrame | None = None
        self.raw_data: pd.DataFrame | None = None
        self.price_col = "Close"
        self.model: RandomForestClassifier | None = None
        self.scaler = StandardScaler()
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None
        self.predictions: np.ndarray | None = None
        self.test_probabilities: np.ndarray | None = None
        self.features = features or BASE_FEATURES.copy()
        self.metrics: dict[str, Any] = {}
        self.feature_importance: pd.DataFrame | None = None
        self.forecast_df: pd.DataFrame | None = None
        self.last_price: float | None = None
        self.last_date: pd.Timestamp | None = None
        self.target_mode = "Next-day direction"

    def load_data(
        self,
        start_date: str | date | None = None,
        end_date: str | date | None = None,
        interval: str = "1d",
        data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Load and normalize Yahoo Finance data, or normalize supplied data."""
        if data is None:
            if start_date is None or end_date is None:
                raise ValueError("A start date and end date are required for Yahoo Finance downloads.")
            downloaded = yf.download(
                self.ticker,
                start=str(start_date),
                end=str(end_date),
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if downloaded.empty:
                raise ValueError(f"No Yahoo Finance data returned for {self.ticker}.")
            normalized = _flatten_yahoo_columns(downloaded)
        else:
            normalized = _flatten_yahoo_columns(
                data.rename(
                    columns={
                        "date": "Date",
                        "open": "Open",
                        "high": "High",
                        "low": "Low",
                        "close": "Close",
                        "volume": "Volume",
                    }
                )
                if "date" in data.columns
                else data
            )
        self.raw_data = normalized
        self.data = normalized.copy()
        self.price_col = "Adj_Close" if "Adj_Close" in normalized.columns else "Close"
        return normalized

    @staticmethod
    def engineer_frame(
        data: pd.DataFrame,
        target_mode: str = "Next-day direction",
    ) -> pd.DataFrame:
        """Create the supplied implementation's technical indicators."""
        frame = data.copy()
        price_col = "Adj_Close" if "Adj_Close" in frame.columns else "Close"
        prices = frame[price_col].astype(float)
        frame["Return"] = prices.pct_change()
        frame["MA5"] = prices.rolling(5).mean()
        frame["MA10"] = prices.rolling(10).mean()
        frame["MA20"] = prices.rolling(20).mean()
        frame["MA50"] = prices.rolling(50).mean()
        frame["MA200"] = prices.rolling(200).mean()
        frame["Volatility"] = frame["Return"].rolling(20).std()
        frame["Volatility_20"] = frame["Volatility"]
        gains = prices.diff().clip(lower=0).rolling(14).mean()
        losses = prices.diff().clip(upper=0).abs().rolling(14).mean().replace(0, np.nan)
        frame["RSI_14"] = 100 - (100 / (1 + gains / losses))
        frame["Price_MA20_Ratio"] = prices / frame["MA20"]
        frame["Price_MA50_Ratio"] = prices / frame["MA50"]
        frame["Volume_Ratio"] = frame["Volume"] / frame["Volume"].rolling(20).mean()
        frame["High_Low_Ratio"] = frame["High"] / frame["Low"]
        frame["Close_Open_Ratio"] = frame["Close"] / frame["Open"]
        frame["Momentum_5"] = prices / prices.shift(5) - 1
        frame["Momentum_10"] = prices / prices.shift(10) - 1
        frame["MACD"] = prices.ewm(span=12, adjust=False).mean() - prices.ewm(span=26, adjust=False).mean()
        frame["MACD_Signal"] = frame["MACD"].ewm(span=9, adjust=False).mean()
        frame["ATR_14"] = (frame["High"] - frame["Low"]).rolling(14).mean()
        frame["Bollinger_Width"] = prices.rolling(20).std() * 4
        frame["OBV"] = (np.sign(prices.diff()).fillna(0) * frame["Volume"]).cumsum()
        frame["VWAP_Distance"] = (
            prices - (prices * frame["Volume"]).rolling(20).sum() / frame["Volume"].rolling(20).sum()
        ) / prices
        if target_mode == "5-day direction":
            target_return = prices.shift(-5) / prices - 1
        elif target_mode == "Threshold-based > +0.5%":
            target_return = prices.shift(-1) / prices - 1
        else:
            target_return = prices.shift(-1) / prices - 1
        frame["Trend"] = np.where(
            target_return > (0.005 if target_mode == "Threshold-based > +0.5%" else 0),
            "bullish",
            "bearish",
        )
        return frame.dropna().reset_index(drop=True)

    def engineer_features(self, target_mode: str = "Next-day direction") -> pd.DataFrame:
        """Engineer features and create the configured classification target."""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        self.target_mode = target_mode
        engineered = self.engineer_frame(self.raw_data, target_mode)
        available = [feature for feature in self.features if feature in engineered.columns]
        if len(available) < 2:
            raise ValueError("At least two supported features are required.")
        self.features = available
        self.data = engineered
        return engineered

    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare a chronological split and fit the feature scaler on training data."""
        if self.data is None:
            raise ValueError("Features are not prepared. Call engineer_features() first.")
        X = self.data[self.features]
        y = self.data["Trend"]
        split_idx = int(len(self.data) * (1 - self.test_size))
        if split_idx < 20 or len(self.data) - split_idx < 10:
            raise ValueError("Not enough usable observations for the selected test size.")
        self.X_train = self.scaler.fit_transform(X.iloc[:split_idx])
        self.X_test = self.scaler.transform(X.iloc[split_idx:])
        self.y_train = y.iloc[:split_idx].to_numpy()
        self.y_test = y.iloc[split_idx:].to_numpy()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train(self) -> RandomForestClassifier:
        """Fit the Random Forest classifier."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=(1 if setting("CLOUD_MODE", "false").lower() == "true" else -1),
            max_depth=self.max_depth,
            class_weight=self.class_weight,
        )
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def predict(self) -> np.ndarray:
        """Predict held-out test labels and retain probabilities."""
        if self.model is None or self.X_test is None:
            raise ValueError("Model not trained. Call train() first.")
        self.predictions = self.model.predict(self.X_test)
        self.test_probabilities = self.model.predict_proba(self.X_test)
        return self.predictions

    def evaluate(self) -> dict[str, Any]:
        """Calculate metrics and the labeled confusion matrix."""
        if self.predictions is None:
            self.predict()
        assert self.y_test is not None
        assert self.model is not None
        labels = ["bearish", "bullish"]
        probabilities = self.test_probabilities
        bull_index = list(self.model.classes_).index("bullish") if "bullish" in self.model.classes_ else 0
        try:
            roc_auc = float(roc_auc_score(self.y_test == "bullish", probabilities[:, bull_index]))
        except (ValueError, TypeError):
            roc_auc = 0.5
        self.metrics = {
            "accuracy": float(accuracy_score(self.y_test, self.predictions)),
            "precision": float(precision_score(self.y_test, self.predictions, average="weighted", zero_division=0)),
            "recall": float(recall_score(self.y_test, self.predictions, average="weighted", zero_division=0)),
            "f1": float(f1_score(self.y_test, self.predictions, average="weighted", zero_division=0)),
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix(self.y_test, self.predictions, labels=labels).tolist(),
            "classification_report": {
                label: values
                for label, values in self._classification_rows(labels).items()
            },
            "train_samples": int(len(self.y_train) if self.y_train is not None else 0),
            "test_samples": int(len(self.y_test)),
        }
        return self.metrics

    def _classification_rows(self, labels: list[str]) -> dict[str, dict[str, float]]:
        assert self.y_test is not None and self.predictions is not None
        rows: dict[str, dict[str, float]] = {}
        for label in labels:
            actual = self.y_test == label
            predicted = self.predictions == label
            tp = int((actual & predicted).sum())
            fp = int((~actual & predicted).sum())
            fn = int((actual & ~predicted).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            rows[label] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1-score": float(2 * precision * recall / max(precision + recall, 1e-12)),
                "support": float(actual.sum()),
            }
        return rows

    def get_feature_importance(self) -> pd.DataFrame:
        """Return fitted Random Forest impurity-based feature importance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        self.feature_importance = pd.DataFrame(
            {"feature": self.features, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return self.feature_importance

    def generate_forecast(self, forecast_days: int | None = None) -> pd.DataFrame:
        """Generate a recursive business-day forecast from fitted model probabilities."""
        if self.model is None or self.data is None or self.raw_data is None:
            raise ValueError("Model and engineered data are required before forecasting.")
        days = int(forecast_days or self.forecast_days)
        self.last_date = pd.Timestamp(self.raw_data["Date"].iloc[-1])
        self.last_price = float(self.raw_data[self.price_col].iloc[-1])
        current_features = self.data[self.features].iloc[-1].astype(float).copy()
        returns = self.raw_data[self.price_col].pct_change().dropna()
        std_return = float(returns.std()) if not returns.empty else 0.015
        current_price = self.last_price
        current_date = self.last_date
        rows: list[dict[str, Any]] = []
        for day in range(1, days + 1):
            scaled = self.scaler.transform(pd.DataFrame([current_features], columns=self.features))
            prediction = str(self.model.predict(scaled)[0])
            probabilities = self.model.predict_proba(scaled)[0]
            classes = list(self.model.classes_)
            bull_index = classes.index("bullish") if "bullish" in classes else 0
            bull_probability = float(probabilities[bull_index])
            expected_return = (
                0.002 + 0.008 * bull_probability
                if prediction == "bullish"
                else -0.002 - 0.008 * (1 - bull_probability)
            )
            expected_return = float(np.clip(expected_return, -0.025, 0.025))
            next_price = current_price * (1 + expected_return)
            current_date = self._next_business_day(current_date)
            widening = min(0.75, 1.96 * max(std_return, 0.001) * 0.5 * np.sqrt(day))
            rows.append(
                {
                    "Day": day,
                    "Date": current_date,
                    "Forecast_Price": float(next_price),
                    "Predicted_Trend": prediction,
                    "Bullish_Probability": bull_probability,
                    "Confidence_Lower": float(next_price * (1 - widening)),
                    "Confidence_Upper": float(next_price * (1 + widening)),
                }
            )
            current_features = self._next_features(current_features, next_price, current_price)
            current_price = next_price
        self.forecast_df = pd.DataFrame(rows)
        return self.forecast_df

    @staticmethod
    def _next_business_day(current: pd.Timestamp) -> pd.Timestamp:
        next_day = current + pd.Timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += pd.Timedelta(days=1)
        return next_day

    def _next_features(
        self,
        previous: pd.Series,
        next_price: float,
        current_price: float,
    ) -> pd.Series:
        """Update the recursive feature vector using the supplied implementation's rules."""
        daily_return = (next_price - current_price) / current_price if current_price else 0.0
        updated = previous.copy()
        ma20 = float(previous.get("MA20", next_price) * 0.95 + next_price * 0.05)
        ma50 = float(previous.get("MA50", next_price) * 0.98 + next_price * 0.02)
        values = {
            "Return": daily_return,
            "MA20": ma20,
            "MA50": ma50,
            "Volatility": float(previous.get("Volatility", 0.015) * 0.95 + abs(daily_return) * 0.05),
            "Price_MA20_Ratio": next_price / ma20 if ma20 else 1.0,
            "Price_MA50_Ratio": next_price / ma50 if ma50 else 1.0,
            "Volume_Ratio": float(previous.get("Volume_Ratio", 1.0) * 0.95 + 0.05),
            "High_Low_Ratio": float(previous.get("High_Low_Ratio", 1.01) * 0.95 + (1 + abs(daily_return) * 0.5) * 0.05),
            "Close_Open_Ratio": 1 + daily_return,
            "Momentum_5": float(previous.get("Momentum_5", 0.0) * 0.8 + daily_return * 0.2),
            "Momentum_10": float(previous.get("Momentum_10", 0.0) * 0.9 + daily_return * 0.1),
        }
        for feature, value in values.items():
            if feature in updated.index:
                updated[feature] = value
        return updated[self.features]