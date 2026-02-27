"""TCN Multimodal Preprocessing"""

import time

import numpy as np
import pandas as pd


def calculate_bollinger_bands(prices, window=10, num_std=2.0):
    # BOLLINGER BANDS
    middle_band = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    return upper_band, middle_band, lower_band


def calculate_rsi(prices, window=7):
    # rsi
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    # MACD
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def preprocessTCNMMBaseline(time_series, dates, config, verbose=False):
    st_ = time.time()

    input_size = len(dates)
    base_features = config.get("features", ["open", "high", "low", "close", "volume"])
    calculate_indicators = config.get("calculate_indicators", True)

    rows = []
    for i in range(input_size):
        ts_point = time_series[i] if time_series[i] else {}
        row = {}
        for feat in base_features:
            value = ts_point.get(feat)
            row[feat] = value if value is not None else np.nan
        rows.append(row)

    features_df = pd.DataFrame(rows)

    close_prices = features_df["close"]

    if calculate_indicators:
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices)
        features_df["bb_upper"] = bb_upper.values
        features_df["bb_middle"] = bb_middle.values
        features_df["bb_lower"] = bb_lower.values

        # RSI
        features_df["rsi"] = calculate_rsi(close_prices).values

        # MACD
        macd_line, signal_line, histogram = calculate_macd(close_prices)
        features_df["macd"] = macd_line.values
        features_df["macd_signal"] = signal_line.values
        features_df["macd_histogram"] = histogram.values

    features_df = features_df.ffill().bfill().fillna(0)

    time_series_features = features_df.values

    if verbose:
        print(f" Time to preprocess TCN: {time.time() - st_:.4f}s")
        print(f" Features shape: {time_series_features.shape}")

    return [time_series_features]


if __name__ == "__main__":
    dummy_time_series = [
        {
            "open": 100 + i,
            "high": 102 + i,
            "low": 98 + i,
            "close": 100 + i,
            "volume": 1000000,
        }
        for i in range(14)
    ]

    dummy_dates = [f"2020-01-{i + 1:02d}" for i in range(14)]

    config = {
        "features": ["open", "high", "low", "close", "volume"],
        "calculate_indicators": True,
    }

    result = preprocessTCNMMBaseline(
        dummy_time_series, dummy_dates, config, verbose=True
    )

    print(f"\nResult format: List with {len(result)} element(s)")
    print(f"Features shape: {result[0].shape}")
    print(f"Sample features (first day):\n{result[0][0]}")
