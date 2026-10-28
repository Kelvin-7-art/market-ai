"""Yahoo Finance and scikit-learn pipeline for the Random Forest advanced model.

The Streamlit page owns presentation. This module owns the model lifecycle
for the Advanced variant: download, extended feature engineering (25+
indicators), hyperparameter tuning with GridSearchCV, TimeSeriesSplit
cross-validation, evaluation, feature importance, and recursive forecasting.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


# ============================================================
# DEFAULT FEATURE LIST (25+ indicators)
# ============================================================

ADVANCED_FEATURES: list[str] = [
    # Returns
    "Return_1d",
    "Return_5d",
    "Return_10d",
    "Return_20d",
    # Moving averages
    "MA5",
    "MA10",
    "MA20",
    "MA50",
    "MA200",
    # Price ratios
    "Price_MA5_Ratio",
    "Price_MA10_Ratio",
    "Price_MA20_Ratio",
    "Price_MA50_Ratio",
    # Volatility
    "Volatility_10",
    "Volatility_20",
    "Volatility_50",
    "ATR",
    # Volume
    "Volume_Ratio_5",
    "Volume_Ratio_20",
    "Volume_Change",
    "OBV",
    "OBV_Ratio",
    "CMF",
    # Momentum
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    # Patterns
    "High_Low_Ratio",
    "Close_Open_Ratio",
    "High_Close_Ratio",
    "Low_Close_Ratio",
    # Bollinger Bands
    "BB_Upper",
    "BB_Middle",
    "BB_Lower",
    "BB_Position",
    # Trend & structure
    "Stoch_K",
    "Stoch_D",
    "Williams_R",
    "ADX",
    "Trend_Strength",
    "Dist_From_High",
    "Dist_From_Low",
    "Price_Position",
]


# ============================================================
# DEFAULT HYPERPARAMETER GRID (reduced for responsiveness)
# ============================================================

DEFAULT_PARAM_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"],
    "class_weight": ["balanced", None],
}


# ============================================================
# COLUMN NORMALIZATION HELPERS
# ============================================================

def _flatten_yahoo_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance's single- and multi-ticker column formats."""
    result = frame.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            next(
                (
                    str(value)
                    for value in column
                    if str(value).lower()
                    in {"open", "high", "low", "close", "adj close", "volume"}
                ),
                str(column[0]),
            )
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

    result["Date"] = (
        pd.to_datetime(result["Date"]).dt.tz_localize(None).dt.normalize()
    )

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in result.columns]
    if missing:
        raise ValueError(
            f"Yahoo Finance data is missing required columns: {', '.join(missing)}"
        )

    result = (
        result.dropna(subset=required)
        .sort_values("Date")
        .reset_index(drop=True)
    )

    for column in required + (
        ["Adj_Close"] if "Adj_Close" in result.columns else []
    ):
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


# ============================================================
# ADVANCED RANDOM FOREST MODEL
# ============================================================

class RandomForestAdvancedModel:
    """
    Advanced Random Forest classifier with extended feature engineering,
    GridSearchCV hyperparameter tuning, and TimeSeriesSplit cross-validation.
    """

    def __init__(
        self,
        ticker: str = "AAPL",
        random_state: int = 42,
        test_size: float = 0.2,
        cv_splits: int = 3,
        forecast_days: int = 300,
        features: list[str] | None = None,
        param_grid: dict[str, list[Any]] | None = None,
        scoring: str = "accuracy",
    ) -> None:
        self.ticker = ticker.upper().strip()
        self.random_state = int(random_state)
        self.test_size = float(test_size)
        self.cv_splits = int(cv_splits)
        self.forecast_days = int(forecast_days)
        self.scoring = str(scoring)

        # Data
        self.data: pd.DataFrame | None = None
        self.raw_data: pd.DataFrame | None = None
        self.price_col = "Close"

        # Model
        self.model: RandomForestClassifier | None = None
        self.scaler = StandardScaler()

        # Splits
        self.X_train: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

        # Predictions
        self.predictions: np.ndarray | None = None
        self.test_probabilities: np.ndarray | None = None

        # Features
        self.features = list(features) if features else ADVANCED_FEATURES.copy()

        # Tuning
        self.param_grid = dict(param_grid) if param_grid else dict(DEFAULT_PARAM_GRID)
        self.best_params: dict[str, Any] = {}
        self.best_score: float | None = None
        self.cv_results_summary: pd.DataFrame | None = None
        self.last_grid_total_fits: int = 0

        # Results
        self.metrics: dict[str, Any] = {}
        self.feature_importance: pd.DataFrame | None = None
        self.forecast_df: pd.DataFrame | None = None
        self.last_price: float | None = None
        self.last_date: pd.Timestamp | None = None

        # Target
        self.target_mode = "Next-day direction"

    # --------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------

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
                raise ValueError(
                    "A start date and end date are required for Yahoo Finance downloads."
                )
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
                raise ValueError(
                    f"No Yahoo Finance data returned for {self.ticker}."
                )
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
        self.price_col = (
            "Adj_Close" if "Adj_Close" in normalized.columns else "Close"
        )
        return normalized

    # --------------------------------------------------------
    # EXTENDED FEATURE ENGINEERING (25+ INDICATORS)
    # --------------------------------------------------------

    @staticmethod
    def engineer_frame(
        data: pd.DataFrame,
        target_mode: str = "Next-day direction",
    ) -> pd.DataFrame:
        """Create the 25+ technical indicators used by the advanced model."""
        frame = data.copy()
        price_col = "Adj_Close" if "Adj_Close" in frame.columns else "Close"
        prices = frame[price_col].astype(float)

        # ---------- RETURNS ----------
        frame["Return_1d"] = prices.pct_change()
        frame["Return_5d"] = prices.pct_change(periods=5)
        frame["Return_10d"] = prices.pct_change(periods=10)
        frame["Return_20d"] = prices.pct_change(periods=20)

        # ---------- MOVING AVERAGES ----------
        frame["MA5"] = prices.rolling(5).mean()
        frame["MA10"] = prices.rolling(10).mean()
        frame["MA20"] = prices.rolling(20).mean()
        frame["MA50"] = prices.rolling(50).mean()
        frame["MA200"] = prices.rolling(200).mean()

        # ---------- PRICE / MA RATIOS ----------
        frame["Price_MA5_Ratio"] = prices / frame["MA5"]
        frame["Price_MA10_Ratio"] = prices / frame["MA10"]
        frame["Price_MA20_Ratio"] = prices / frame["MA20"]
        frame["Price_MA50_Ratio"] = prices / frame["MA50"]

        # ---------- VOLATILITY ----------
        frame["Volatility_10"] = frame["Return_1d"].rolling(10).std()
        frame["Volatility_20"] = frame["Return_1d"].rolling(20).std()
        frame["Volatility_50"] = frame["Return_1d"].rolling(50).std()

        # Average True Range
        tr1 = frame["High"] - frame["Low"]
        tr2 = (frame["High"] - frame["Close"].shift()).abs()
        tr3 = (frame["Low"] - frame["Close"].shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        frame["ATR"] = true_range.rolling(14).mean()

        # ---------- VOLUME ----------
        frame["Volume_Ratio_5"] = (
            frame["Volume"] / frame["Volume"].rolling(5).mean()
        )
        frame["Volume_Ratio_20"] = (
            frame["Volume"] / frame["Volume"].rolling(20).mean()
        )
        frame["Volume_Change"] = frame["Volume"].pct_change()
        frame["OBV"] = (
            np.sign(prices.diff()).fillna(0) * frame["Volume"]
        ).cumsum()
        frame["OBV_Ratio"] = frame["OBV"] / frame["OBV"].rolling(20).mean()
        mf_multiplier = (
            (frame["Close"] - frame["Low"]) - (frame["High"] - frame["Close"])
        ) / (frame["High"] - frame["Low"] + 0.001)
        mf_volume = mf_multiplier * frame["Volume"]
        frame["CMF"] = mf_volume.rolling(20).sum() / (
            frame["Volume"].rolling(20).sum() + 1e-9
        )

        # ---------- MOMENTUM ----------
        frame["Momentum_5"] = prices / prices.shift(5) - 1
        frame["Momentum_10"] = prices / prices.shift(10) - 1
        frame["Momentum_20"] = prices / prices.shift(20) - 1

        gains = prices.diff().clip(lower=0).rolling(14).mean()
        losses = (
            prices.diff().clip(upper=0).abs().rolling(14).mean().replace(0, np.nan)
        )
        frame["RSI"] = 100 - (100 / (1 + gains / losses))

        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        frame["MACD"] = ema12 - ema26
        frame["MACD_Signal"] = frame["MACD"].ewm(span=9, adjust=False).mean()
        frame["MACD_Histogram"] = frame["MACD"] - frame["MACD_Signal"]

        # ---------- PATTERNS ----------
        frame["High_Low_Ratio"] = frame["High"] / frame["Low"]
        frame["Close_Open_Ratio"] = frame["Close"] / frame["Open"]
        frame["High_Close_Ratio"] = frame["High"] / frame["Close"]
        frame["Low_Close_Ratio"] = frame["Low"] / frame["Close"]

        # ---------- BOLLINGER BANDS ----------
        ma20 = prices.rolling(20).mean()
        std20 = prices.rolling(20).std()
        frame["BB_Upper"] = ma20 + 2 * std20
        frame["BB_Middle"] = ma20
        frame["BB_Lower"] = ma20 - 2 * std20
        frame["BB_Position"] = (prices - frame["BB_Lower"]) / (
            frame["BB_Upper"] - frame["BB_Lower"] + 0.001
        )

        # ---------- TREND & STRUCTURE ----------
        low14 = frame["Low"].rolling(14).min()
        high14 = frame["High"].rolling(14).max()
        frame["Stoch_K"] = (
            100 * (frame["Close"] - low14) / (high14 - low14 + 0.001)
        )
        frame["Stoch_D"] = frame["Stoch_K"].rolling(3).mean()
        frame["Williams_R"] = (
            -100 * (high14 - frame["Close"]) / (high14 - low14 + 0.001)
        )

        plus_dm = frame["High"].diff()
        minus_dm = -frame["Low"].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        atr = true_range.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / (atr + 1e-9))
        minus_di = 100 * (minus_dm.rolling(14).mean() / (atr + 1e-9))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 0.001)
        frame["ADX"] = dx.rolling(14).mean()

        frame["Trend_Strength"] = frame["MA20"] / frame["MA50"] - 1

        frame["Dist_From_High"] = (
            frame["Close"] - frame["High"].rolling(20).max()
        ) / (frame["High"].rolling(20).max() + 1e-9)
        frame["Dist_From_Low"] = (
            frame["Close"] - frame["Low"].rolling(20).min()
        ) / (frame["Low"].rolling(20).min() + 1e-9)

        frame["Price_Position"] = (
            prices - frame["Low"].rolling(20).min()
        ) / (
            frame["High"].rolling(20).max()
            - frame["Low"].rolling(20).min()
            + 0.001
        )

        # ---------- TARGET ----------
        if target_mode == "5-day direction":
            target_return = prices.shift(-5) / prices - 1
            threshold = 0.0
        elif target_mode == "Threshold-based > +0.5%":
            target_return = prices.shift(-1) / prices - 1
            threshold = 0.005
        else:  # Next-day direction
            target_return = prices.shift(-1) / prices - 1
            threshold = 0.0

        frame["Trend"] = np.where(
            target_return > threshold, "bullish", "bearish"
        )

        return frame.dropna().reset_index(drop=True)

    def engineer_features(
        self, target_mode: str = "Next-day direction"
    ) -> pd.DataFrame:
        """Engineer the 25+ features and build the classification target."""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.target_mode = target_mode
        engineered = self.engineer_frame(self.raw_data, target_mode)

        available = [
            feature for feature in self.features if feature in engineered.columns
        ]
        if len(available) < 2:
            raise ValueError("At least two supported features are required.")

        self.features = available
        self.data = engineered
        return engineered

    # --------------------------------------------------------
    # DATA PREPARATION
    # --------------------------------------------------------

    def prepare_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare a chronological split and fit the feature scaler."""
        if self.data is None:
            raise ValueError(
                "Features are not prepared. Call engineer_features() first."
            )

        X = self.data[self.features]
        y = self.data["Trend"]

        split_idx = int(len(self.data) * (1 - self.test_size))
        if split_idx < 20 or len(self.data) - split_idx < 10:
            raise ValueError(
                "Not enough usable observations for the selected test size."
            )

        self.X_train = self.scaler.fit_transform(X.iloc[:split_idx])
        self.X_test = self.scaler.transform(X.iloc[split_idx:])
        self.y_train = y.iloc[:split_idx].to_numpy()
        self.y_test = y.iloc[split_idx:].to_numpy()

        return self.X_train, self.X_test, self.y_train, self.y_test

    # --------------------------------------------------------
    # HYPERPARAMETER TUNING
    # --------------------------------------------------------

    def tune_hyperparameters(self) -> dict[str, Any]:
        """Run GridSearchCV with TimeSeriesSplit and refit the best estimator."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        grid = {key: list(values) for key, values in self.param_grid.items()}

        total_combos = int(np.prod([len(values) for values in grid.values()]))
        self.last_grid_total_fits = total_combos * self.cv_splits

        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        search = GridSearchCV(
            estimator=RandomForestClassifier(
                random_state=self.random_state, n_jobs=-1
            ),
            param_grid=grid,
            cv=tscv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )

        search.fit(self.X_train, self.y_train)

        self.model = search.best_estimator_
        self.best_params = dict(search.best_params_)
        self.best_score = float(search.best_score_)

        try:
            results = pd.DataFrame(search.cv_results_)
            cols = [
                c
                for c in results.columns
                if c.startswith("param_")
                or c in {"mean_test_score", "std_test_score", "rank_test_score"}
            ]
            self.cv_results_summary = (
                results[cols]
                .sort_values("rank_test_score")
                .head(20)
                .reset_index(drop=True)
            )
        except Exception:
            self.cv_results_summary = None

        return self.best_params

    # --------------------------------------------------------
    # TRAIN / PREDICT / EVALUATE
    # --------------------------------------------------------

    def train(self) -> RandomForestClassifier:
        """Fit a Random Forest using either the tuned params or defaults."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        if self.model is None:
            self.model = RandomForestClassifier(
                random_state=self.random_state, n_jobs=-1
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

        classes = list(self.model.classes_)
        bull_index = classes.index("bullish") if "bullish" in classes else 0

        try:
            roc_auc = float(
                roc_auc_score(
                    self.y_test == "bullish",
                    probabilities[:, bull_index],
                )
            )
        except (ValueError, TypeError):
            roc_auc = 0.5

        self.metrics = {
            "accuracy": float(accuracy_score(self.y_test, self.predictions)),
            "precision": float(
                precision_score(
                    self.y_test,
                    self.predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    self.y_test,
                    self.predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    self.y_test,
                    self.predictions,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "roc_auc": roc_auc,
            "confusion_matrix": confusion_matrix(
                self.y_test, self.predictions, labels=labels
            ).tolist(),
            "classification_report": self._classification_rows(labels),
            "train_samples": int(
                len(self.y_train) if self.y_train is not None else 0
            ),
            "test_samples": int(len(self.y_test)),
        }
        return self.metrics

    def _classification_rows(
        self, labels: list[str]
    ) -> dict[str, dict[str, float]]:
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
                "f1-score": float(
                    2 * precision * recall / max(precision + recall, 1e-12)
                ),
                "support": float(actual.sum()),
            }
        return rows

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    def analyze_feature_importance(self) -> pd.DataFrame:
        """Return built-in Gini importance and permutation importance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if self.X_test is None or self.y_test is None:
            raise ValueError("Test data is required to compute importance.")

        builtin = self.model.feature_importances_

        perm = permutation_importance(
            self.model,
            self.X_test,
            self.y_test,
            n_repeats=5,
            random_state=self.random_state,
            n_jobs=-1,
            scoring=self.scoring,
        )

        self.feature_importance = (
            pd.DataFrame(
                {
                    "Feature": self.features,
                    "Builtin_Importance": builtin,
                    "Permutation_Mean": perm.importances_mean,
                    "Permutation_STD": perm.importances_std,
                }
            )
            .sort_values("Builtin_Importance", ascending=False)
            .reset_index(drop=True)
        )
        return self.feature_importance

    def get_feature_importance(self) -> pd.DataFrame:
        """Alias for analyze_feature_importance for API parity."""
        return self.analyze_feature_importance()

    # --------------------------------------------------------
    # RECURSIVE FORECAST (FIXED)
    # --------------------------------------------------------

    def generate_forecast(
        self, forecast_days: int | None = None
    ) -> pd.DataFrame:
        """Generate a recursive business-day forecast from fitted probabilities.

        Fixes over previous versions:
        - Drift blends the long-horizon mean with the recent 20-day momentum,
          so forecasts visibly trend rather than hug the start price.
        - Expected return has asymmetric bullish/bearish tilts.
        - Confidence band is anchored at t=0 and saturates (rather than
          exploding with sqrt(day)), and its cap scales with the horizon.
        """
        if self.model is None or self.data is None or self.raw_data is None:
            raise ValueError(
                "Model and engineered data are required before forecasting."
            )

        days = int(forecast_days or self.forecast_days)
        self.last_date = pd.Timestamp(self.raw_data["Date"].iloc[-1])
        self.last_price = float(self.raw_data[self.price_col].iloc[-1])
        current_features = (
            self.data[self.features].iloc[-1].astype(float).copy()
        )

        # ---- Historical statistics used to shape the forecast ----
        returns = self.raw_data[self.price_col].pct_change().dropna()
        std_return = float(returns.std()) if not returns.empty else 0.015

        mean_return = float(returns.mean()) if not returns.empty else 0.0005
        recent_return = (
            float(returns.tail(20).mean())
            if len(returns) >= 20
            else mean_return
        )
        blended_drift = 0.4 * mean_return + 0.6 * recent_return
        drift = float(np.clip(blended_drift, -0.0035, 0.0035))

        # Band saturation: 95% CI half-width approaches this cap.
        max_band = float(np.clip(0.10 + days / 365 * 0.20, 0.10, 0.30))

        current_price = self.last_price
        current_date = self.last_date
        rows: list[dict[str, Any]] = []

        for day in range(1, days + 1):
            scaled = self.scaler.transform(
                pd.DataFrame([current_features], columns=self.features)
            )
            prediction = str(self.model.predict(scaled)[0])
            probabilities = self.model.predict_proba(scaled)[0]
            classes = list(self.model.classes_)
            bull_index = classes.index("bullish") if "bullish" in classes else 0
            bull_probability = float(probabilities[bull_index])

            # Expected return: drift + confidence-scaled tilt
            tilt = (bull_probability - 0.5) * 0.008
            base = drift + tilt
            if prediction == "bearish":
                base -= 0.0008
            elif prediction == "bullish":
                base += 0.0004
            expected_return = float(np.clip(base, -0.015, 0.015))

            next_price = current_price * (1 + expected_return)
            current_date = self._next_business_day(current_date)

            # Saturating confidence band
            tau = max(4.0, days / 8)
            half_width = max_band * (1.0 - np.exp(-np.sqrt(day) / tau))
            half_width = max(
                half_width, 1.96 * std_return * np.sqrt(day) * 0.4
            )
            half_width = min(half_width, max_band)

            rows.append(
                {
                    "Day": day,
                    "Date": current_date,
                    "Forecast_Price": float(next_price),
                    "Predicted_Trend": prediction,
                    "Bullish_Probability": bull_probability,
                    "Confidence_Lower": float(next_price * (1 - half_width)),
                    "Confidence_Upper": float(next_price * (1 + half_width)),
                }
            )

            current_features = self._next_features(
                current_features, next_price, current_price
            )
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
        """Update the recursive feature vector with the same rules as the base model."""
        daily_return = (
            (next_price - current_price) / current_price
            if current_price
            else 0.0
        )
        updated = previous.copy()

        ma5 = float(previous.get("MA5", next_price) * 0.8 + next_price * 0.2)
        ma10 = float(previous.get("MA10", next_price) * 0.9 + next_price * 0.1)
        ma20 = float(previous.get("MA20", next_price) * 0.95 + next_price * 0.05)
        ma50 = float(previous.get("MA50", next_price) * 0.98 + next_price * 0.02)
        ma200 = float(
            previous.get("MA200", next_price) * 0.99 + next_price * 0.01
        )

        vol10 = float(
            previous.get("Volatility_10", 0.01) * 0.9
            + abs(daily_return) * 0.1
        )
        vol20 = float(
            previous.get("Volatility_20", 0.015) * 0.95
            + abs(daily_return) * 0.05
        )
        vol50 = float(
            previous.get("Volatility_50", 0.02) * 0.98
            + abs(daily_return) * 0.02
        )

        values: dict[str, float] = {
            "Return_1d": daily_return,
            "Return_5d": float(
                (1 + previous.get("Return_5d", 0.0)) * (1 + daily_return) - 1
            ),
            "Return_10d": float(
                (1 + previous.get("Return_10d", 0.0)) * (1 + daily_return) - 1
            ),
            "Return_20d": float(
                (1 + previous.get("Return_20d", 0.0)) * (1 + daily_return) - 1
            ),
            "MA5": ma5,
            "MA10": ma10,
            "MA20": ma20,
            "MA50": ma50,
            "MA200": ma200,
            "Price_MA5_Ratio": next_price / ma5 if ma5 else 1.0,
            "Price_MA10_Ratio": next_price / ma10 if ma10 else 1.0,
            "Price_MA20_Ratio": next_price / ma20 if ma20 else 1.0,
            "Price_MA50_Ratio": next_price / ma50 if ma50 else 1.0,
            "Volatility_10": vol10,
            "Volatility_20": vol20,
            "Volatility_50": vol50,
            "ATR": float(
                previous.get("ATR", 1.0) * 0.95
                + abs(daily_return) * next_price * 0.05
            ),
            "Volume_Ratio_5": float(
                previous.get("Volume_Ratio_5", 1.0) * 0.8 + 0.2
            ),
            "Volume_Ratio_20": float(
                previous.get("Volume_Ratio_20", 1.0) * 0.95 + 0.05
            ),
            "Volume_Change": float(
                previous.get("Volume_Change", 0.0) * 0.9 + daily_return * 0.1
            ),
            "OBV": float(
                previous.get("OBV", 0.0)
                + np.sign(daily_return) * 1_000_000.0
            ),
            "OBV_Ratio": float(
                previous.get("OBV_Ratio", 1.0) * 0.95 + 0.05
            ),
            "CMF": float(
                previous.get("CMF", 0.0) * 0.95
                + np.sign(daily_return) * 0.05 * 0.1
            ),
            "Momentum_5": float(
                previous.get("Momentum_5", 0.0) * 0.8 + daily_return * 0.2
            ),
            "Momentum_10": float(
                previous.get("Momentum_10", 0.0) * 0.9 + daily_return * 0.1
            ),
            "Momentum_20": float(
                previous.get("Momentum_20", 0.0) * 0.95 + daily_return * 0.05
            ),
            "RSI": float(
                np.clip(
                    100 - (100 / (1 + daily_return / (vol20 + 0.001))),
                    0,
                    100,
                )
            ),
            "MACD": float(
                previous.get("MACD", 0.0) * 0.95 + daily_return * 10 * 0.05
            ),
            "MACD_Signal": float(
                previous.get("MACD_Signal", 0.0) * 0.95
                + previous.get("MACD", 0.0) * 0.05
            ),
            "High_Low_Ratio": float(
                previous.get("High_Low_Ratio", 1.01) * 0.95
                + (1 + abs(daily_return) * 0.5) * 0.05
            ),
            "Close_Open_Ratio": 1 + daily_return,
            "High_Close_Ratio": float(
                previous.get("High_Close_Ratio", 1.01) * 0.95
                + (1 + abs(daily_return) * 0.3) * 0.05
            ),
            "Low_Close_Ratio": float(
                previous.get("Low_Close_Ratio", 0.99) * 0.95
                + (1 - abs(daily_return) * 0.3) * 0.05
            ),
            "BB_Upper": ma20 + 2 * vol20,
            "BB_Middle": ma20,
            "BB_Lower": ma20 - 2 * vol20,
            "BB_Position": (
                (next_price - (ma20 - 2 * vol20)) / (4 * vol20 + 0.001)
            ),
            "Stoch_K": float(np.clip(50 + daily_return * 100, 0, 100)),
            "Stoch_D": float(np.clip(50 + daily_return * 100, 0, 100)),
            "Williams_R": float(np.clip(-50 + daily_return * 100, -100, 0)),
            "ADX": float(np.clip(25 + abs(daily_return) * 100, 0, 100)),
            "Trend_Strength": (ma20 / ma50 - 1) if ma50 else 0.0,
            "Dist_From_High": float(
                previous.get("Dist_From_High", 0.0) * 0.95
                - abs(daily_return) * 0.05
            ),
            "Dist_From_Low": float(
                previous.get("Dist_From_Low", 0.0) * 0.95
                + abs(daily_return) * 0.05
            ),
            "Price_Position": float(np.clip(0.5 + daily_return * 10, 0, 1)),
        }

        values["MACD_Histogram"] = values["MACD"] - values["MACD_Signal"]

        for feature, value in values.items():
            if feature in updated.index:
                updated[feature] = value

        return updated[self.features]


# ============================================================
# MODULE EXPORTS
# ============================================================

__all__ = [
    "ADVANCED_FEATURES",
    "DEFAULT_PARAM_GRID",
    "RandomForestAdvancedModel",
    "to_chart_frame",
]