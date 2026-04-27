"""Feature engineering for daily OHLCV. All features at time t use only data up to t."""
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


FEATURE_COLS = [
    "log_return",
    "log_range",
    "log_gap",
    "norm_volume",
    "rsi14",
    "atr14",
    "rv3",
    "rv5",
    "zret5",
    "zret20",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_month_end",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Input: OHLCV with a 'Date' column. Output: feature DataFrame indexed by date."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    df = df.sort_values("Date").set_index("Date")
    o = df["Open"].replace(0, np.nan)
    h = df["High"].replace(0, np.nan)
    l = df["Low"].replace(0, np.nan)
    c = df["Close"].replace(0, np.nan)
    v = df["Volume"]

    out = pd.DataFrame(index=df.index)
    out["log_return"] = np.log(c / c.shift(1))
    out["log_range"] = np.log(h / l)
    out["log_gap"] = np.log(o / c.shift(1))
    out["norm_volume"] = np.log((v + 1.0) / (v.rolling(20).mean() + 1.0))

    out["rsi14"] = _rsi(c, 14) / 100.0
    out["atr14"] = _atr(h, l, c, 14) / c

    out["rv3"] = out["log_return"].rolling(3).std()
    out["rv5"] = out["log_return"].rolling(5).std()
    m5 = out["log_return"].rolling(5).mean()
    s5 = out["log_return"].rolling(5).std()
    out["zret5"] = (out["log_return"] - m5) / (s5 + 1e-8)
    m20 = out["log_return"].rolling(20).mean()
    s20 = out["log_return"].rolling(20).std()
    out["zret20"] = (out["log_return"] - m20) / (s20 + 1e-8)

    dow = df.index.dayofweek.values
    month = df.index.month.values
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    out["is_month_end"] = df.index.is_month_end.astype(np.float32)

    return out[FEATURE_COLS].dropna()
