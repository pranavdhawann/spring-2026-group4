"""Feature engineering from OHLCV.

All features are computed so that row t depends only on information available at
the close of day t (no lookahead). The target `log_return` at row t represents
the return realized on day t; windowing downstream pairs a history ending at t-1
with future returns t, t+1, ... t+h-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COL = "log_return"


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def _bb_width(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    return ((mid + n_std * std) - (mid - n_std * std)) / mid


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    out = pd.DataFrame(index=df.index)

    log_ret = np.log(c / c.shift(1))
    out[TARGET_COL] = log_ret
    out["hl_range"] = (h - l) / c
    out["co_gap"] = (c - o) / o

    hl = (h - l).replace(0.0, np.nan)
    out["upper_shadow"] = (h - c) / hl
    out["lower_shadow"] = (c - l) / hl
    out["vwap_proxy"] = (h + l + c) / 3.0

    out["log_volume"] = np.log1p(v)
    out["volume_ratio"] = v / v.rolling(20, min_periods=20).mean()

    for w in (5, 10, 20):
        m = log_ret.rolling(w, min_periods=w).mean()
        s = log_ret.rolling(w, min_periods=w).std()
        out[f"ret_mean_{w}"] = m
        out[f"ret_std_{w}"] = s
        out[f"ret_sharpe_{w}"] = m / s.replace(0.0, np.nan)

    out["rsi_14"] = _rsi(c, 14)
    out["atr_14"] = _atr(h, l, c, 14)
    macd, signal = _macd(c)
    out["macd"] = macd
    out["macd_signal"] = signal
    out["bb_width_20"] = _bb_width(c, 20)

    for lag in (1, 2, 3, 5):
        out[f"log_return_lag_{lag}"] = log_ret.shift(lag)

    return out.replace([np.inf, -np.inf], np.nan).dropna()
