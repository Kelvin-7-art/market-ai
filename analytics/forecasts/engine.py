"""Time-series model fitting, evaluation, and Yahoo Finance data loading."""

from __future__ import annotations

import logging
import warnings
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import yfinance as yf

FORECAST_MODELS = ["AR", "MA", "ARMA", "ARIMA", "Prophet", "SARIMA / SARIMAX"]
INTERVALS = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}


class PriceDataError(ValueError):
    """A user-readable market-data failure."""


def normalize_ticker(ticker: str, market_group: str = "") -> str:
    """Normalize common custom forex/crypto forms without changing stock suffixes."""
    symbol = ticker.strip().upper().replace(" ", "")
    if not symbol:
        raise PriceDataError("Enter a ticker symbol.")
    if market_group.lower() == "forex":
        symbol = symbol.replace("/", "").removesuffix("=X")
        if len(symbol) != 6 or not symbol.isalpha():
            raise PriceDataError("Use a six-letter forex pair, for example EURUSD=X or GBP/USD.")
        symbol += "=X"
    elif market_group.lower() == "crypto":
        symbol = symbol.replace("/", "-")
        if "-" not in symbol:
            symbol += "-USD"
    return symbol


def _extract_prices(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name="price")
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            if ticker in frame.columns.get_level_values(level):
                frame = frame.xs(ticker, axis=1, level=level)
                break
        if isinstance(frame.columns, pd.MultiIndex):
            for level in range(frame.columns.nlevels):
                if "Close" in frame.columns.get_level_values(level):
                    frame.columns = frame.columns.get_level_values(level)
                    break
    column = next((c for c in ["Adj Close", "Close"] if c in frame and frame[c].notna().any()), None)
    if column is None:
        raise PriceDataError("Yahoo returned data without usable closing prices.")
    index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if index.tz is not None:
        index = index.tz_localize(None)
    series = pd.Series(pd.to_numeric(frame[column], errors="coerce").to_numpy(), index=index.normalize(), name="price")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    return series[~series.index.duplicated(keep="last")].sort_index()


def load_price_series(ticker: str, start_date: date, end_date: date, interval: str) -> pd.Series:
    """Return adjusted closes, or raise PriceDataError for the UI to display.

    Dates are inclusive to the user. Both sources fetch daily bars, then use
    identical local resampling. Failed calls are not cached.
    """
    ticker = normalize_ticker(ticker)
    if start_date >= end_date:
        raise PriceDataError("Start date must be earlier than end date.")
    if interval not in INTERVALS.values():
        raise PriceDataError("Select Daily, Weekly, or Monthly sampling.")
    end_date = min(end_date, date.today())
    if start_date >= end_date:
        raise PriceDataError("Choose a historical date range ending after its start date.")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    attempts = []
    series = pd.Series(dtype=float)
    source = "download"
    for source in ["download", "history(max)"]:
        try:
            if source == "download":
                raw = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1),
                    interval="1d", auto_adjust=False, progress=False, threads=False, timeout=20)
            else:
                raw = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False,
                    actions=False, timeout=20, raise_errors=True)
            candidate = _extract_prices(raw, ticker)
            candidate = candidate.loc[(candidate.index >= pd.Timestamp(start_date)) & (candidate.index <= pd.Timestamp(end_date))]
            if candidate.empty:
                attempts.append(source + ": no observations in selected dates")
                continue
            series = candidate
            break
        except Exception as error:
            attempts.append(source + ": " + type(error).__name__)
    if series.empty:
        raise PriceDataError(f"No usable prices for {ticker}. Ticker not found, no data in the selected dates, or Yahoo Finance is currently rate-limiting/unavailable. Please try again later or check the symbol and date range. Both download and history fallback were attempted.")
    # Period-end labels prevent assigning a weekly/monthly close to an earlier date.
    # Exclude incomplete periods, including the current daily bar.
    cutoff = min(pd.Timestamp(end_date), pd.Timestamp(date.today()) - pd.Timedelta(days=1))
    series = series.loc[series.index <= cutoff]
    if interval == "1wk":
        series = series.resample("W-SUN").last().dropna()
    elif interval == "1mo":
        series = series.resample(pd.offsets.MonthEnd()).last().dropna()
    series = series.loc[series.index <= cutoff]
    if len(series) < 30:
        raise PriceDataError(f"Only {len(series)} completed {interval} observations are available for {ticker}. Choose a longer date range or a finer sampling interval (at least 30 observations required).")
    series.attrs.update(source="Yahoo Finance", loader=source, attempts=attempts,
                        price_basis="Adjusted close where available; otherwise close",
                        sampling="Completed calendar periods; current daily bar excluded")
    return series


def make_future_index(last_date: pd.Timestamp, horizon: int, interval: str, market_group: str) -> pd.DatetimeIndex:
    """Create future labels strictly after the last historical observation."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")
    last = pd.Timestamp(last_date)
    if interval == "1d":
        offset = pd.offsets.Day() if market_group.lower() == "crypto" else pd.offsets.BusinessDay()
    elif interval == "1wk":
        offset = pd.offsets.Week()
    elif interval == "1mo":
        offset = pd.offsets.MonthEnd()
    else:
        raise ValueError(f"Unsupported time-series interval: {interval}")
    return pd.date_range(start=last + offset, periods=horizon, freq=offset)


def _fit_predict_detailed(
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    model_name: str,
    params: dict[str, int],
    interval: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None, float | None, str]:
    """Fit one estimator and return point/interval forecasts plus its AIC."""
    source_values = history.astype(float)
    horizon = len(future_index)
    if model_name == "Prophet":
        try:
            from prophet import Prophet
        except ImportError as error:
            raise ValueError("Prophet is not installed. Run: python -m pip install prophet") from error

        logger_names = ("prophet", "cmdstanpy")
        previous_levels = {
            name: logging.getLogger(name).level for name in logger_names
        }
        for name in logger_names:
            logging.getLogger(name).setLevel(logging.ERROR)
        try:
            training = pd.DataFrame(
                {"ds": source_values.index, "y": source_values.to_numpy()}
            )
            model = Prophet(
                interval_width=0.95,
                yearly_seasonality=len(source_values)
                >= (365 if interval == "1d" else 52 if interval == "1wk" else 18),
                weekly_seasonality=interval == "1d" and len(source_values) >= 30,
                daily_seasonality=False,
                uncertainty_samples=200,
            )
            model.fit(training)
            prediction = model.predict(pd.DataFrame({"ds": future_index}))
            aic = None
            return (
                prediction["yhat"].to_numpy(dtype=float),
                prediction["yhat_lower"].to_numpy(dtype=float),
                prediction["yhat_upper"].to_numpy(dtype=float),
                aic, None,
                f"Prophet: {len(training)} observations; 95% model uncertainty interval; "
                f"seasonalities={model.seasonalities}; changepoints={len(model.changepoints)}. "
                "AIC and BIC are not reported. Intervals use Prophet uncertainty simulation.",
            )
        finally:
            for name, level in previous_levels.items():
                logging.getLogger(name).setLevel(level)

    # Yahoo trading dates are irregular around weekends and market holidays.
    # Statsmodels cannot generate out-of-sample timestamps from that index, so
    # fit it on observation positions and keep calendar dates for output only.
    values = pd.Series(
        source_values.to_numpy(dtype=float),
        index=pd.RangeIndex(len(source_values), name="observation"),
        name=source_values.name or "price",
    )
    p = params.get("p", 2)
    d = params.get("d", 1)
    q = params.get("q", 2)
    if model_name == "AR":
        fitted = AutoReg(values, lags=p, trend="ct").fit()
        prediction = fitted.get_prediction(
            start=len(values),
            end=len(values) + horizon - 1,
            dynamic=False,
        ).summary_frame(alpha=0.05)
        mean = prediction["mean"].to_numpy(dtype=float)
        lower = prediction["mean_ci_lower"].to_numpy(dtype=float)
        upper = prediction["mean_ci_upper"].to_numpy(dtype=float)
    else:
        if model_name == "MA":
            order = (0, 0, q)
            fitted = ARIMA(values, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
        elif model_name == "ARMA":
            order = (p, 0, q)
            fitted = ARIMA(values, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
        elif model_name == "ARIMA":
            order = (p, d, q)
            fitted = ARIMA(values, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
        else:
            seasonal_order = (
                params.get("seasonal_p", 1),
                params.get("seasonal_d", 1),
                params.get("seasonal_q", 1),
                params.get("seasonal_period", 5 if interval == "1d" else 13 if interval == "1wk" else 12),
            )
            order = (p, d, q)
            fitted = SARIMAX(
                values,
                order=order,
                seasonal_order=seasonal_order,
                trend="n" if d + seasonal_order[1] else "c",
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=150)
        prediction = fitted.get_forecast(steps=horizon)
        mean = np.asarray(prediction.predicted_mean, dtype=float)
        intervals = np.asarray(prediction.conf_int(alpha=0.05), dtype=float)
        lower, upper = intervals[:, 0], intervals[:, 1]
    return mean, lower, upper, float(fitted.aic), float(fitted.bic), fitted.summary().as_text()


def _fit_predict(history, future_index, model_name, params, interval):
    """Compatibility wrapper retaining the previous four-value private result."""
    return _fit_predict_detailed(history, future_index, model_name, params, interval)[:4]


def _error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate chronological holdout errors."""
    errors = actual - predicted
    nonzero = np.abs(actual) > 1e-12
    mape = float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan")
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
        "mape": mape,
    }


def run_time_series_forecast(
    series: pd.Series,
    model_name: str,
    horizon: int,
    interval: str,
    market_group: str,
    params: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Backtest one model chronologically, then fit it to all data for a forecast."""
    if model_name not in FORECAST_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least one period.")
    values = series.astype(float).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    values = values[~values.index.duplicated(keep="last")]
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError("Prices must have a date index.")
    settings = params or {}
    if len(values) < 30:
        raise ValueError("At least 30 price observations are required.")

    test_size = min(max(5, int(round(len(values) * 0.2))), max(5, len(values) // 3))
    training, test = values.iloc[:-test_size], values.iloc[-test_size:]
    seasonal_period = settings.get(
        "seasonal_period",
        5 if interval == "1d" else 13 if interval == "1wk" else 12,
    )
    if model_name == "SARIMA / SARIMAX" and len(training) < max(30, 2 * seasonal_period):
        raise ValueError(
            f"SARIMA/SARIMAX needs at least {max(30, 2 * seasonal_period)} training observations "
            f"for a seasonal period of {seasonal_period}; choose a longer date range or smaller period."
        )
    if model_name != "Prophet" and len(training) <= settings.get("p", 2) + settings.get("q", 2) + 8:
        raise ValueError("The selected date range is too short for this model order.")

    backtest_index = pd.DatetimeIndex(test.index)
    future_index = make_future_index(values.index[-1], horizon, interval, market_group)
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always")
        backtest_mean, backtest_lower, backtest_upper, _, _, _ = _fit_predict_detailed(
            training, backtest_index, model_name, settings, interval
        )
        future_mean, future_lower, future_upper, aic, bic, summary = _fit_predict_detailed(
            values, future_index, model_name, settings, interval
        )

    for array in [backtest_mean, backtest_lower, backtest_upper, future_mean, future_lower, future_upper]:
        if not np.isfinite(array).all():
            raise ValueError("The model produced non-finite forecasts. Try simpler orders or a longer history.")
    metrics = _error_metrics(test.to_numpy(dtype=float), backtest_mean)
    backtest = pd.DataFrame(
        {
            "date": backtest_index,
            "actual": test.to_numpy(dtype=float),
            "forecast": backtest_mean,
            "lower": backtest_lower,
            "upper": backtest_upper,
        }
    )
    backtest["error"] = backtest["actual"] - backtest["forecast"]
    forecast = pd.DataFrame(
        {
            "date": future_index,
            "forecast": future_mean,
            "lower": future_lower,
            "upper": future_upper,
        }
    )
    last = float(values.iloc[-1])
    final = float(future_mean[-1])
    metrics.update(aic=aic, bic=bic, last_actual=last, final_forecast=final,
                   expected_return_pct=(final / last - 1) * 100 if last != 0 else None)
    return {
        "bic": bic, "summary": summary,
        "fit_warnings": sorted(set(str(w.message) for w in fit_warnings)),
        "holdout_train_start": values.index[0], "holdout_train_end": training.index[-1],
        "model": model_name,
        "forecast": forecast,
        "backtest": backtest,
        "metrics": metrics,
        "aic": aic,
        "observations": len(values),
        "train_observations": len(training),
        "test_observations": len(test),
        "last_observation": pd.Timestamp(values.index[-1]),
        "params": settings.copy(),
    }


__all__ = [
    "FORECAST_MODELS",
    "INTERVALS",
    "load_price_series",
    "normalize_ticker",
    "PriceDataError",
    "make_future_index",
    "run_time_series_forecast",
]
