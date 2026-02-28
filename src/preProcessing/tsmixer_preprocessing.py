"""TSMixer Preprocessing — Enhanced 23-Feature Engineering

Features (all computable within 14-day history window):
  Base OHLCV (5):     open, high, low, close, volume
  Returns (3):        ret_1d, ret_3d, ret_5d
  Volatility (3):     vol_5d, garman_klass, vol_ratio_5_7
  Volume (2):         abnormal_vol, volume_trend
  Microstructure (2): spread_proxy, amihud
  Technical (7):      bb_upper, bb_middle, bb_lower, rsi, macd, macd_signal, macd_histogram
  Momentum (1):       roc_5
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Technical Indicator Functions (adapted window sizes for 14-day sequences)
# ---------------------------------------------------------------------------

def calculate_bollinger_bands(prices, window=5, num_std=2.0):
    middle_band = prices.rolling(window=window, min_periods=1).mean()
    std = prices.rolling(window=window, min_periods=1).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    return upper_band, middle_band, lower_band


def calculate_rsi(prices, window=7):
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gains = gains.rolling(window=window, min_periods=1).mean()
    avg_losses = losses.rolling(window=window, min_periods=1).mean()
    rs = avg_gains / (avg_losses + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=5, slow=10, signal=4):
    """MACD with periods adapted for 14-day windows.

    V2: Changed from 12/26/9 to 5/10/4.  The original 26-period slow EMA
    is meaningless with only 14 data points (it collapses to the SMA of
    all values). These shorter periods give the EMA crossover enough room
    to produce meaningful signals within the available window.
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------------------------------------------------------------
# New Feature Functions
# ---------------------------------------------------------------------------

def _calculate_returns(close):
    """Log returns at 1d, 3d, 5d horizons (adapted for 14-day window)."""
    safe_close = close.replace(0, np.nan).ffill().fillna(1.0)

    ret_1d = np.log(safe_close / safe_close.shift(1)).fillna(0)
    ret_3d = np.log(safe_close / safe_close.shift(3)).fillna(0)
    ret_5d = np.log(safe_close / safe_close.shift(5)).fillna(0)

    return ret_1d, ret_3d, ret_5d


def _calculate_volatility(close, high, low, open_price, ret_1d):
    """Realized vol, Garman-Klass vol, vol ratio."""
    # 5-day realized volatility
    vol_5d = ret_1d.rolling(window=5, min_periods=2).std().fillna(0)

    # Garman-Klass volatility estimator (per-day, from OHLC)
    safe_high = high.replace(0, np.nan).ffill().fillna(1.0)
    safe_low = low.replace(0, np.nan).ffill().fillna(1.0)
    safe_open = open_price.replace(0, np.nan).ffill().fillna(1.0)
    safe_close = close.replace(0, np.nan).ffill().fillna(1.0)

    hl_ratio = np.log(safe_high / safe_low)
    co_ratio = np.log(safe_close / safe_open)
    garman_klass = 0.5 * hl_ratio ** 2 - (2 * np.log(2) - 1) * co_ratio ** 2

    # Volatility ratio: vol_5d / vol_7d (adapted for 14-day window)
    vol_7d = ret_1d.rolling(window=7, min_periods=2).std().fillna(0)
    vol_ratio = pd.Series(
        np.where(vol_7d == 0, 1.0, vol_5d / vol_7d),
        index=close.index
    )

    return vol_5d, garman_klass, vol_ratio


def _calculate_volume_features(volume):
    """Abnormal volume and volume trend."""
    avg_vol_10 = volume.rolling(window=10, min_periods=1).mean()
    avg_vol_5 = volume.rolling(window=5, min_periods=1).mean()

    abnormal_vol = pd.Series(
        np.where(avg_vol_10 == 0, 1.0, volume / avg_vol_10),
        index=volume.index
    )
    volume_trend = pd.Series(
        np.where(avg_vol_10 == 0, 1.0, avg_vol_5 / avg_vol_10),
        index=volume.index
    )

    return abnormal_vol, volume_trend


def _calculate_microstructure(close, high, low, volume, ret_1d):
    """Spread proxy and Amihud illiquidity."""
    safe_close = close.replace(0, np.nan).ffill().fillna(1.0)

    # Parkinson range proxy for bid-ask spread
    spread_proxy = (high - low) / safe_close

    # Amihud illiquidity ratio
    dollar_vol = (safe_close * volume).abs()
    safe_dollar_vol = dollar_vol.replace(0, np.nan).fillna(1.0)
    amihud = ret_1d.abs() / safe_dollar_vol

    return spread_proxy, amihud


def _calculate_roc(close, period=5):
    """Rate of change over given period (default 5 for 14-day window)."""
    safe_close = close.replace(0, np.nan).ffill().fillna(1.0)
    roc = ((close - close.shift(period)) / safe_close.shift(period) * 100).fillna(0)
    return roc


# ---------------------------------------------------------------------------
# Sample-Level Feature Extraction
# ---------------------------------------------------------------------------

def _extract_sample_features(sample, base_features):
    """Extract base features from a sample dict, same as TCN."""
    time_series = sample['time_series']
    target = sample['target']
    dates = sample['dates']

    rows = []
    for ts_point in time_series:
        row = {}
        for feat in base_features:
            value = ts_point.get(feat)
            row[feat] = value if value is not None else np.nan
        rows.append(row)

    features_df = pd.DataFrame(rows)
    return features_df, target, dates


def _calculate_all_features(features_df):
    """Add all 18 derived features to the base 5 OHLCV columns."""
    close = features_df['close']
    high = features_df['high']
    low = features_df['low']
    open_price = features_df['open']
    volume = features_df['volume']

    # Returns (3)
    ret_1d, ret_3d, ret_5d = _calculate_returns(close)
    features_df['ret_1d'] = ret_1d.values
    features_df['ret_3d'] = ret_3d.values
    features_df['ret_5d'] = ret_5d.values

    # Volatility (3)
    vol_5d, garman_klass, vol_ratio = _calculate_volatility(
        close, high, low, open_price, ret_1d
    )
    features_df['vol_5d'] = vol_5d.values
    features_df['garman_klass'] = garman_klass.values
    features_df['vol_ratio_5_7'] = vol_ratio.values

    # Volume features (2)
    abnormal_vol, volume_trend = _calculate_volume_features(volume)
    features_df['abnormal_vol'] = abnormal_vol.values
    features_df['volume_trend'] = volume_trend.values

    # Microstructure (2)
    spread_proxy, amihud = _calculate_microstructure(
        close, high, low, volume, ret_1d
    )
    features_df['spread_proxy'] = spread_proxy.values
    features_df['amihud'] = amihud.values

    # Bollinger Bands (3) — window=5 for 14-day sequences
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close)
    features_df['bb_upper'] = bb_upper.values
    features_df['bb_middle'] = bb_middle.values
    features_df['bb_lower'] = bb_lower.values

    # RSI (1) — window=7 for 14-day sequences
    features_df['rsi'] = calculate_rsi(close).values

    # MACD (3) — EWM adapts to available data
    macd_line, signal_line, histogram = calculate_macd(close)
    features_df['macd'] = macd_line.values
    features_df['macd_signal'] = signal_line.values
    features_df['macd_histogram'] = histogram.values

    # Rate of Change (1)
    features_df['roc_5'] = _calculate_roc(close).values

    return features_df


def _handle_nans(features_df):
    """Handle NaN/Inf values: ffill -> bfill -> fillna(0)."""
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.ffill()
    features_df = features_df.bfill()
    features_df = features_df.fillna(0)
    return features_df


# ---------------------------------------------------------------------------
# Main Preprocessing Function
# ---------------------------------------------------------------------------

def preprocess_for_tsmixer(train_dataset, test_dataset, config, verbose=False):
    """Preprocess data for TSMixer model.

    Args:
        train_dataset: BaselineDataLoader (iterable of sample dicts)
        test_dataset:  BaselineDataLoader
        config:        dict loaded from tsmixer_config.yaml
        verbose:       bool

    Returns:
        X_train, y_train, X_test, y_test: torch.Tensor (float32)
        scaler: fitted StandardScaler
        train_anchors, test_anchors: np.ndarray (last close prices)
        target_type: str ('pct_change' or 'ratio')
        target_mean, target_std: float (for z-score inverse transform)
    """
    preprocess_config = config.get('preprocessing', {})
    base_features = preprocess_config.get(
        'features', ['open', 'high', 'low', 'close', 'volume']
    )
    calculate_indicators = preprocess_config.get('calculate_indicators', True)
    standardization = preprocess_config.get('standardization', 'zscore')

    if verbose:
        print("=" * 70)
        print("TSMIXER PREPROCESSING (23 Enhanced Features)")
        print("=" * 70)
        print(f"  Base features: {base_features}")
        print(f"  Calculate indicators: {calculate_indicators}")
        print(f"  Standardization: {standardization}")

    def process_dataset(dataset, split_name):
        X_list = []
        y_list = []

        iterator = tqdm(
            range(len(dataset)),
            desc=f"Processing {split_name}"
        ) if verbose else range(len(dataset))

        for idx in iterator:
            sample = dataset[idx]

            features_df, target, dates = _extract_sample_features(
                sample, base_features
            )

            if calculate_indicators:
                features_df = _calculate_all_features(features_df)

            features_df = _handle_nans(features_df)

            target = [
                t if t is not None else np.nan
                for t in target
            ]

            if any(np.isnan(t) for t in target):
                continue

            X_list.append(features_df.values)
            y_list.append(target)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)

        if verbose:
            print(f"  {split_name} X shape: {X.shape}")
            print(f"  {split_name} y shape: {y.shape}")

        return X, y

    if verbose:
        print("\n[1/5] Processing training data...")
    X_train, y_train = process_dataset(train_dataset, "train")

    if verbose:
        print("\n[2/5] Processing test data...")
    X_test, y_test = process_dataset(test_dataset, "test")

    # ------------------------------------------------------------------
    # [3/5] Normalize targets by last close price (anchor)
    # pct_change: (price / anchor - 1) * 100  -> targets ~[-5, +5]
    # ratio:      price / anchor              -> targets ~[0.98, 1.02]
    # ------------------------------------------------------------------
    target_type = preprocess_config.get('target_type', 'pct_change')

    if verbose:
        print(f"\n[3/5] Normalizing targets (mode={target_type})...")

    # close is column index 3, last timestep is index -1
    train_anchors = X_train[:, -1, 3].copy()
    test_anchors = X_test[:, -1, 3].copy()

    # Safe division (avoid div-by-zero for penny stocks or bad data)
    train_anchors_safe = np.where(
        (train_anchors == 0) | np.isnan(train_anchors), 1.0, train_anchors
    )
    test_anchors_safe = np.where(
        (test_anchors == 0) | np.isnan(test_anchors), 1.0, test_anchors
    )

    if target_type == 'pct_change':
        # Percent change: (price / anchor - 1) * 100
        y_train = (y_train / train_anchors_safe[:, np.newaxis] - 1.0) * 100.0
        y_test = (y_test / test_anchors_safe[:, np.newaxis] - 1.0) * 100.0

        # Z-score targets: standardize to mean=0, std=1
        # This prevents the model from "hiding" in the small-magnitude regime
        target_mean = float(y_train.mean())
        target_std = float(y_train.std())
        y_train = (y_train - target_mean) / (target_std + 1e-8)
        y_test = (y_test - target_mean) / (target_std + 1e-8)
    else:
        # Legacy ratio mode (kept for backward compatibility)
        y_train = y_train / train_anchors_safe[:, np.newaxis]
        target_mean = 0.0
        target_std = 1.0

    if verbose:
        print(f"  Train anchors: min={train_anchors.min():.2f}, "
              f"max={train_anchors.max():.2f}, mean={train_anchors.mean():.2f}")
        print(f"  Target pct stats: mean={target_mean:.4f}, std={target_std:.4f}")
        print(f"  y_train after z-score: "
              f"min={y_train.min():.4f}, max={y_train.max():.4f}, "
              f"mean={y_train.mean():.4f}, std={y_train.std():.4f}")

    if verbose:
        print(f"\n[4/5] Standardizing features using {standardization}...")

    N_train, seq_len, num_features = X_train.shape
    N_test = X_test.shape[0]

    per_sample_ohlcv = preprocess_config.get('per_sample_ohlcv_norm', True)

    if per_sample_ohlcv:
        # V2: Hybrid normalization strategy
        # - OHLCV prices (cols 0-3): per-sample divide by anchor close
        # - Volume (col 4): per-sample divide by mean volume
        # - Derived features (cols 5-22): global z-score (already scale-invariant)
        #
        # This prevents mixing $5 and $500 stocks in the same global z-score,
        # while keeping global statistics for features that are naturally
        # cross-stock comparable (returns, RSI, volatility, etc.)

        # Per-sample OHLCV price normalization (cols 0-3)
        for X in [X_train, X_test]:
            anchor = X[:, -1:, 3:4].copy()  # last close, shape (N, 1, 1)
            anchor = np.where((anchor == 0) | np.isnan(anchor), 1.0, anchor)
            X[:, :, 0:4] = X[:, :, 0:4] / anchor

            # Per-sample volume normalization (col 4)
            vol_mean = np.mean(np.abs(X[:, :, 4:5]), axis=1, keepdims=True)
            vol_mean = np.where((vol_mean == 0) | np.isnan(vol_mean), 1.0, vol_mean)
            X[:, :, 4:5] = X[:, :, 4:5] / vol_mean

        # Global z-score ONLY for derived features (cols 5 onward)
        n_derived = num_features - 5
        if n_derived > 0:
            derived_train = X_train[:, :, 5:].reshape(-1, n_derived)
            derived_test = X_test[:, :, 5:].reshape(-1, n_derived)

            scaler = StandardScaler()
            derived_train_scaled = scaler.fit_transform(derived_train)
            derived_test_scaled = scaler.transform(derived_test)

            X_train[:, :, 5:] = derived_train_scaled.reshape(N_train, seq_len, n_derived)
            X_test[:, :, 5:] = derived_test_scaled.reshape(N_test, seq_len, n_derived)
        else:
            scaler = None

        if verbose:
            print("  V2: Per-sample OHLCV normalization + global z-score for derived features")
    else:
        # Legacy: global z-score for ALL features
        X_train_reshaped = X_train.reshape(-1, num_features)
        X_test_reshaped = X_test.reshape(-1, num_features)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

        X_train = X_train_scaled.reshape(N_train, seq_len, num_features)
        X_test = X_test_scaled.reshape(N_test, seq_len, num_features)

    # Replace any remaining NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    if verbose:
        print("\n[5/5] Converting to tensors...")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    if verbose:
        print("PREPROCESSING COMPLETE!")
        print(f"  X_train: {X_train.shape}  ->  (samples, seq_len, features)")
        print(f"  y_train: {y_train.shape}  ->  (samples, 7 days) [{target_type}]")
        print(f"  X_test:  {X_test.shape}   ->  (samples, seq_len, features)")
        print(f"  y_test:  {y_test.shape}   ->  (samples, 7 days) [{target_type}]")
        print(f"  Scaler:  {type(scaler).__name__} fitted on train")
        print(f"  Anchors: train={len(train_anchors)}, test={len(test_anchors)}")
        print(f"  Features ({num_features}):")
        feature_names = [
            'open', 'high', 'low', 'close', 'volume',
            'ret_1d', 'ret_3d', 'ret_5d',
            'vol_5d', 'garman_klass', 'vol_ratio_5_7',
            'abnormal_vol', 'volume_trend',
            'spread_proxy', 'amihud',
            'bb_upper', 'bb_middle', 'bb_lower', 'rsi',
            'macd', 'macd_signal', 'macd_histogram',
            'roc_5',
        ]
        for i, name in enumerate(feature_names):
            print(f"    [{i:2d}] {name}")

    return X_train, y_train, X_test, y_test, scaler, train_anchors, test_anchors, target_type, target_mean, target_std


if __name__ == "__main__":
    import os
    from src.dataLoader.dataLoaderBaseline import getTrainTestDataLoader
    from src.utils import read_json_file, read_yaml

    print("Loading configs...")
    config = read_yaml('config/config.yaml')
    tsmixer_config = read_yaml('config/tsmixer_config.yaml')

    ticker2idx = read_json_file(
        os.path.join(config['BASELINE_DATA_PATH'], config['TICKER2IDX'])
    )

    data_config = {
        'data_path': config['BASELINE_DATA_PATH'],
        'ticker2idx': ticker2idx,
        'test_train_split': 0.2,
        'random_seed': 42,
    }

    print("\nLoading dataloaders...")
    train_dataset, test_dataset = getTrainTestDataLoader(data_config)

    print("\nRunning preprocessing...")
    X_train, y_train, X_test, y_test, scaler, train_anchors, test_anchors, target_type, target_mean, target_std = preprocess_for_tsmixer(
        train_dataset,
        test_dataset,
        tsmixer_config,
        verbose=True
    )
    print(f"\n  Target type: {target_type}")
    print(f"  Target mean: {target_mean:.4f}, std: {target_std:.4f}")
    print(f"  Anchors: train={len(train_anchors)}, test={len(test_anchors)}")

    print("\nSample shapes:")
    print(f"  X_train[0]: {X_train[0].shape}")
    print(f"  y_train[0]: {y_train[0]}")

    # Sanity checks
    assert not torch.isnan(X_train).any(), "NaN in X_train!"
    assert not torch.isinf(X_train).any(), "Inf in X_train!"
    assert X_train.shape[2] == 23, f"Expected 23 features, got {X_train.shape[2]}"
    print("\nAll sanity checks passed!")
