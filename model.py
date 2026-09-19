"""
model.py
========
Core logic for the stock technical-analysis model.

For a given ticker, this module:
  1. Downloads historical daily price data.
  2. Computes a broad set of technical indicators (moving averages, RSI,
     MACD, Bollinger Bands, volume trends, volatility, rate of change).
  3. Trains a RandomForest classifier to predict whether the price ~21
     trading days (~1 month) ahead will be higher than today.
  4. Uses the trained model's feature importances to auto-select which
     indicators actually mattered most for THIS stock (different stocks
     can be driven by different indicators).
  5. Returns a prediction for the most recent data point: probability of
     price increase, a direction label, a confidence level, and the top
     indicators driving that call.

This file has no FastAPI/web dependency — it can be imported and used
directly, or wrapped by main.py as an API.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

PREDICTION_HORIZON_DAYS = 21  # ~1 trading month
MIN_ROWS_REQUIRED = 260       # need enough history to train meaningfully


@dataclass
class StockPrediction:
    ticker: str
    success: bool
    error: Optional[str] = None
    last_price: Optional[float] = None
    last_date: Optional[str] = None
    predicted_direction: Optional[str] = None       # "UP" or "DOWN"
    probability_up: Optional[float] = None           # 0-1
    confidence: Optional[str] = None                 # "Low"/"Medium"/"High"
    horizon_days: int = PREDICTION_HORIZON_DAYS
    top_indicators: List[Dict] = field(default_factory=list)
    backtest_accuracy: Optional[float] = None
    current_indicator_values: Dict = field(default_factory=dict)


def _fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch historical OHLCV data. Requires internet + yfinance at runtime."""
    import yfinance as yf
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a broad set of technical indicator columns to the dataframe."""
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]

    # --- Moving averages ---
    out["SMA10"] = close.rolling(10).mean()
    out["SMA20"] = close.rolling(20).mean()
    out["SMA50"] = close.rolling(50).mean()
    out["SMA200"] = close.rolling(200).mean()
    out["EMA12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA26"] = close.ewm(span=26, adjust=False).mean()

    # Price relative to moving averages (normalized, so comparable across stocks)
    out["price_to_sma20"] = close / out["SMA20"] - 1
    out["price_to_sma50"] = close / out["SMA50"] - 1
    out["price_to_sma200"] = close / out["SMA200"] - 1
    out["sma20_to_sma50"] = out["SMA20"] / out["SMA50"] - 1

    # --- MACD ---
    out["MACD"] = out["EMA12"] - out["EMA26"]
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # --- RSI (14) ---
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands (20, 2) ---
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["bb_upper"] = mid + 2 * std
    out["bb_lower"] = mid - 2 * std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / mid
    out["bb_pct_b"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])

    # --- Volatility & momentum ---
    out["volatility_20d"] = close.pct_change().rolling(20).std()
    out["roc_10"] = close.pct_change(10)
    out["roc_20"] = close.pct_change(20)

    # --- Average True Range (14) ---
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    out["atr_pct"] = out["ATR14"] / close

    # --- Volume trend ---
    out["vol_sma20"] = volume.rolling(20).mean()
    out["vol_ratio"] = volume / out["vol_sma20"]

    return out


FEATURE_COLUMNS = [
    "price_to_sma20", "price_to_sma50", "price_to_sma200", "sma20_to_sma50",
    "MACD", "MACD_signal", "MACD_hist",
    "RSI14",
    "bb_width", "bb_pct_b",
    "volatility_20d", "roc_10", "roc_20",
    "atr_pct",
    "vol_ratio",
]

FEATURE_LABELS = {
    "price_to_sma20": "Price vs 20-day average",
    "price_to_sma50": "Price vs 50-day average",
    "price_to_sma200": "Price vs 200-day average",
    "sma20_to_sma50": "20-day vs 50-day average trend",
    "MACD": "MACD",
    "MACD_signal": "MACD signal line",
    "MACD_hist": "MACD histogram (momentum shift)",
    "RSI14": "RSI (14-day momentum)",
    "bb_width": "Bollinger Band width (volatility)",
    "bb_pct_b": "Position within Bollinger Bands",
    "volatility_20d": "20-day volatility",
    "roc_10": "10-day rate of change",
    "roc_20": "20-day rate of change",
    "atr_pct": "Average true range (% of price)",
    "vol_ratio": "Volume vs 20-day average volume",
}


def analyze_stock(ticker: str, period: str = "5y") -> StockPrediction:
    """Run the full pipeline for one ticker and return a StockPrediction."""
    ticker = ticker.strip().upper()
    try:
        raw = _fetch_history(ticker, period=period)
    except Exception as e:
        return StockPrediction(ticker=ticker, success=False, error=f"Could not fetch data: {e}")

    if raw.empty or len(raw) < MIN_ROWS_REQUIRED:
        return StockPrediction(
            ticker=ticker, success=False,
            error=f"Not enough historical data for '{ticker}' (need ~{MIN_ROWS_REQUIRED} trading days).",
        )

    data = _compute_indicators(raw)

    # Target: is price PREDICTION_HORIZON_DAYS ahead higher than today?
    data["future_close"] = data["Close"].shift(-PREDICTION_HORIZON_DAYS)
    data["target"] = (data["future_close"] > data["Close"]).astype(int)

    model_df = data.dropna(subset=FEATURE_COLUMNS + ["target"])
    if len(model_df) < 100:
        return StockPrediction(
            ticker=ticker, success=False,
            error="Not enough clean data after computing indicators to train reliably.",
        )

    # Most recent row (features exist, but no future target yet) is what we predict on
    latest_row = data.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]

    X = model_df[FEATURE_COLUMNS]
    y = model_df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # preserve time order, no lookahead leakage
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        random_state=42, class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    backtest_acc = float(clf.score(X_test, y_test))

    # Auto-selected top indicators, by the model's own feature importance
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)
    top_indicators = [
        {"indicator": FEATURE_LABELS.get(name, name), "importance": round(float(val), 4)}
        for name, val in importances.head(5).items()
    ]

    proba_up = float(clf.predict_proba(latest_row[FEATURE_COLUMNS])[0][1])
    direction = "UP" if proba_up >= 0.5 else "DOWN"

    conf_gap = abs(proba_up - 0.5)
    if conf_gap >= 0.20:
        confidence = "High"
    elif conf_gap >= 0.08:
        confidence = "Medium"
    else:
        confidence = "Low"

    current_values = {
        FEATURE_LABELS.get(c, c): round(float(latest_row[c].iloc[0]), 4)
        for c in FEATURE_COLUMNS
    }

    return StockPrediction(
        ticker=ticker,
        success=True,
        last_price=round(float(latest_row["Close"].iloc[0]), 2),
        last_date=str(latest_row.index[-1].date()),
        predicted_direction=direction,
        probability_up=round(proba_up, 4),
        confidence=confidence,
        top_indicators=top_indicators,
        backtest_accuracy=round(backtest_acc, 4),
        current_indicator_values=current_values,
    )


def compare_stocks(tickers: List[str], period: str = "5y") -> List[StockPrediction]:
    """Run analyze_stock for each ticker, return results sorted by strongest UP signal first."""
    results = [analyze_stock(t, period=period) for t in tickers]
    results.sort(
        key=lambda r: (r.success, r.probability_up if r.probability_up is not None else -1),
        reverse=True,
    )
    return results
