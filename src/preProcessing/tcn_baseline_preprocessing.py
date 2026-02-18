"""TCN Baseline Preprocessing"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def calculate_bollinger_bands(prices, window=20, num_std=2.0):
    # Bollinger Bands
    middle_band = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)

    return upper_band, middle_band, lower_band

def calculate_rsi(prices, window=14):
    # RSI
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

def _extract_sample_features(sample, base_features):
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

def _calculate_indicators_for_window(features_df):
    close_prices = features_df['close']

    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices)
    features_df['bb_upper'] = bb_upper.values
    features_df['bb_middle'] = bb_middle.values
    features_df['bb_lower'] = bb_lower.values

    # RSI
    features_df['rsi'] = calculate_rsi(close_prices).values

    # MACD
    macd_line, signal_line, histogram = calculate_macd(close_prices)
    features_df['macd'] = macd_line.values
    features_df['macd_signal'] = signal_line.values
    features_df['macd_histogram'] = histogram.values

    return features_df

def _handle_nans(features_df):
    features_df = features_df.ffill()
    features_df = features_df.bfill()
    features_df = features_df.fillna(0)

    return features_df

def preprocess_for_tcn(train_dataset, test_dataset, config, verbose=False):
    preprocess_config = config.get('preprocessing', {})
    base_features = preprocess_config.get(
        'features', ['open', 'high', 'low', 'close', 'volume']
    )
    calculate_indicators = preprocess_config.get('calculate_indicators', True)
    standardization = preprocess_config.get('standardization', 'zscore')

    if verbose:
        print("=" * 70)
        print("TCN BASELINE PREPROCESSING")
        print("=" * 70)
        print(f"  Base features: {base_features}")
        print(f"  Calculate indicators: {calculate_indicators}")
        print(f"  Standardization: {standardization}")

    def process_dataset(dataset, split_name):
        """Process a single dataset split"""
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
                features_df = _calculate_indicators_for_window(features_df)

            features_df = _handle_nans(features_df)

            target = [
                t if t is not None else np.nan
                for t in target
            ]

            if any(np.isnan(t) for t in target):
                continue

            X_list.append(features_df.values)
            y_list.append(target)

        X = np.array(X_list)
        y = np.array(y_list)

        if verbose:
            print(f"  {split_name} X shape: {X.shape}")
            print(f"  {split_name} y shape: {y.shape}")

        return X, y

    if verbose:
        print("\n[1/4] Processing training data...")
    X_train, y_train = process_dataset(train_dataset, "train")

    if verbose:
        print("\n[2/4] Processing test data...")
    X_test, y_test = process_dataset(test_dataset, "test")

    if verbose:
        print(f"\n[3/4] Standardizing features using {standardization}...")

    N_train, seq_len, num_features = X_train.shape
    N_test = X_test.shape[0]

    X_train_reshaped = X_train.reshape(-1, num_features)
    X_test_reshaped = X_test.reshape(-1, num_features)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    X_train = X_train_scaled.reshape(N_train, seq_len, num_features)
    X_test = X_test_scaled.reshape(N_test, seq_len, num_features)

    if verbose:
        print("\n[4/4] Converting to tensors...")

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    if verbose:
        print("PREPROCESSING COMPLETE!")
        print(f"  X_train: {X_train.shape}  →  (samples, seq_len, features)")
        print(f"  y_train: {y_train.shape}  →  (samples, 7 days)")
        print(f"  X_test:  {X_test.shape}   →  (samples, seq_len, features)")
        print(f"  y_test:  {y_test.shape}   →  (samples, 7 days)")
        print(f"  Scaler:  {type(scaler).__name__} fitted on train")

    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    import os
    from src.dataLoader.dataLoaderBaseline import getTrainTestDataLoader
    from src.utils import read_json_file, read_yaml

    print("Loading configs...")
    config = read_yaml('config/config.yaml')
    tcn_config = read_yaml('config/tcn_config.yaml')

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
    X_train, y_train, X_test, y_test, scaler = preprocess_for_tcn(
        train_dataset,
        test_dataset,
        tcn_config,
        verbose=True
    )

    print("\nSample shapes:")
    print(f"  X_train[0]: {X_train[0].shape}")
    print(f"  y_train[0]: {y_train[0]}")


