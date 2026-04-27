"""
technical_indicators_gnn.py — Compute technical indicators for time series features.

All indicators are computed using pandas/numpy (no external TA library required).
Each indicator is normalized to be roughly scale-invariant where appropriate.

Indicators (15 total):
    1.  RSI (14-period)
    2.  MACD line
    3.  MACD signal
    4.  MACD histogram
    5.  Bollinger Band upper (relative distance from close)
    6.  Bollinger Band lower (relative distance from close)
    7.  ATR (14-period, normalized by close)
    8.  OBV z-score
    9.  Stochastic %K
    10. Stochastic %D
    11. ADX (14-period)
    12. CCI (20-period)
    13. Williams %R (14-period)
    14. Rate of Change (10-period)
    15. Money Flow Index (14-period)
"""

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100, scaled to 0-1)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi / 100.0  # Scale to [0, 1]


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple:
    """MACD line, signal line, and histogram (normalized by close)."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)

    macd_line = (ema_fast - ema_slow) / (close + 1e-10)  # Normalize by price
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple:
    """
    Bollinger Bands — returns relative distance from close.
    bb_upper = (upper_band - close) / close
    bb_lower = (close - lower_band) / close
    """
    sma = _sma(close, window)
    std = close.rolling(window=window).std()

    upper = sma + num_std * std
    lower = sma - num_std * std

    bb_upper = (upper - close) / (close + 1e-10)
    bb_lower = (close - lower) / (close + 1e-10)

    return bb_upper, bb_lower


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range, normalized by close."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()
    return atr / (close + 1e-10)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume, returned as expanding z-score."""
    direction = np.sign(close.diff())
    obv = (volume * direction).cumsum()

    # Z-score with expanding window
    mean = obv.expanding(min_periods=20).mean()
    std = obv.expanding(min_periods=20).std().replace(0, 1e-10)
    return (obv - mean) / std


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple:
    """Stochastic Oscillator %K and %D (scaled to 0-1)."""
    lowest = low.rolling(window=k_period).min()
    highest = high.rolling(window=k_period).max()

    stoch_k = (close - lowest) / (highest - lowest + 1e-10)
    stoch_d = stoch_k.rolling(window=d_period).mean()

    return stoch_k, stoch_d


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average Directional Index (scaled to 0-1)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=high.index,
    )

    # Smoothed averages
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, min_periods=period).mean() / (atr + 1e-10)
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, min_periods=period).mean() / (atr + 1e-10)
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=1 / period, min_periods=period).mean()

    return adx / 100.0  # Scale to [0, 1]


def compute_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Commodity Channel Index (normalized by dividing by 200)."""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci = (typical_price - sma) / (0.015 * mad + 1e-10)
    return cci / 200.0  # Rough normalization to ~[-1, 1]


def compute_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R (scaled from [-100, 0] to [-1, 0])."""
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    wr = (highest - close) / (highest - lowest + 1e-10) * -1
    return wr  # Already in [-1, 0]


def compute_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (as fraction, not percentage)."""
    return close.pct_change(periods=period)


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index (scaled to 0-1)."""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    tp_diff = typical_price.diff()
    positive_flow = pd.Series(np.where(tp_diff > 0, money_flow, 0), index=close.index)
    negative_flow = pd.Series(np.where(tp_diff < 0, money_flow, 0), index=close.index)

    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()

    mfr = positive_sum / (negative_sum + 1e-10)
    mfi = 100 - (100 / (1 + mfr))
    return mfi / 100.0  # Scale to [0, 1]


def add_technical_indicators(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    volume: pd.Series,
) -> dict:
    """
    Compute all technical indicators and return as a dict of Series.

    Returns dict with 15 keys matching FeatureBuilder expectations.
    """
    # RSI
    rsi_14 = compute_rsi(close, period=14)

    # MACD
    macd, macd_signal, macd_hist = compute_macd(close)

    # Bollinger Bands
    bb_upper, bb_lower = compute_bollinger_bands(close)

    # ATR
    atr_14 = compute_atr(high, low, close, period=14)

    # OBV
    obv_zscore = compute_obv(close, volume)

    # Stochastic
    stoch_k, stoch_d = compute_stochastic(high, low, close)

    # ADX
    adx_14 = compute_adx(high, low, close, period=14)

    # CCI
    cci_20 = compute_cci(high, low, close, period=20)

    # Williams %R
    willr_14 = compute_williams_r(high, low, close, period=14)

    # Rate of Change
    roc_10 = compute_roc(close, period=10)

    # Money Flow Index
    mfi_14 = compute_mfi(high, low, close, volume, period=14)

    return {
        "rsi_14": rsi_14,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "atr_14": atr_14,
        "obv_zscore": obv_zscore,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "adx_14": adx_14,
        "cci_20": cci_20,
        "willr_14": willr_14,
        "roc_10": roc_10,
        "mfi_14": mfi_14,
    }
