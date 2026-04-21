"""Feature engineering: stationary, scale-invariant channels from OHLCV."""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "log_ret",
    "log_high_low",
    "log_close_open",
    "log_open_prev_close",
    "log_volume",
    "volume_ma_ratio",
    "rolling_mean_5",
    "rolling_std_5",
    "rolling_mean_20",
    "rolling_std_20",
    "rsi_14",
    "macd_signal",
]

TARGET_COL = "log_ret"


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # normalize from [0,100] to [-1,1]
    return (rsi / 50.0) - 1.0


def _macd_signal(close: pd.Series) -> pd.Series:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    diff = macd - signal
    # normalize by rolling std for scale-invariance
    denom = diff.rolling(60, min_periods=20).std().replace(0.0, np.nan)
    return (diff / denom).clip(-5.0, 5.0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a DataFrame with Open/High/Low/Close/Volume; returns engineered features."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)

    close_prev = df["close"].shift(1)

    out = pd.DataFrame(index=df.index)
    out["log_ret"] = np.log(df["close"] / close_prev)
    out["log_high_low"] = np.log(df["high"] / df["low"].replace(0.0, np.nan))
    out["log_close_open"] = np.log(df["close"] / df["open"].replace(0.0, np.nan))
    out["log_open_prev_close"] = np.log(df["open"] / close_prev)
    out["log_volume"] = np.log(df["volume"] + 1.0)
    vol_ma5 = df["volume"].rolling(5, min_periods=5).mean()
    out["volume_ma_ratio"] = df["volume"] / vol_ma5.replace(0.0, np.nan)

    out["rolling_mean_5"] = out["log_ret"].rolling(5, min_periods=5).mean()
    out["rolling_std_5"] = out["log_ret"].rolling(5, min_periods=5).std()
    out["rolling_mean_20"] = out["log_ret"].rolling(20, min_periods=20).mean()
    out["rolling_std_20"] = out["log_ret"].rolling(20, min_periods=20).std()
    out["rsi_14"] = _rsi(df["close"], 14)
    out["macd_signal"] = _macd_signal(df["close"])

    out = out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return out[FEATURE_COLS]
