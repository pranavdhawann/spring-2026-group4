"""
Data preprocessing pipeline for LSTM time-series forecasting.

Loads data via the existing BaselineDataLoader, extracts time_series and target
fields, engineers technical indicators + temporal features, normalizes per-stock,
and produces PyTorch-ready datasets for training, validation, and testing.

Usage:
    from src.preProcessing.data_preprocessing_lstm import prepare_lstm_data

    config = {
        "data_path": "data/baseline_data",
        "ticker2idx_path": "data/baseline_data/tickers2id.json",
        "val_ratio": 0.15,
        "feature_groups": ["ohlcv", "technical", "temporal"],
    }
    train_ds, val_ds, test_ds, feature_scalers, target_scalers = prepare_lstm_data(config)
"""

import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm

from src.dataLoader import getTrainTestDataLoader
from src.utils import read_json_file, read_yaml


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_ohlcv_features(time_series: List[dict]) -> np.ndarray:
    """
    Extract OHLCV features from a sample's time_series list.

    Args:
        time_series: List of dicts, each with keys:
            open, high, low, close, volume, dividends, stock_splits

    Returns:
        np.ndarray of shape (seq_len, 5) with columns [open, high, low, close, volume].
        None values are forward-filled then backward-filled.
    """
    cols = ["open", "high", "low", "close", "volume"]
    arr = np.array(
        [[ts.get(c) for c in cols] for ts in time_series],
        dtype=np.float64,
    )

    # Forward-fill then backward-fill NaN
    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]
        mask = np.isnan(col)
        if mask.all():
            arr[:, col_idx] = 0.0
            continue
        # Forward fill
        for i in range(1, len(col)):
            if mask[i]:
                col[i] = col[i - 1]
        # Backward fill
        for i in range(len(col) - 2, -1, -1):
            if np.isnan(col[i]):
                col[i] = col[i + 1]

    return arr


def compute_technical_indicators(
    ohlcv: np.ndarray,
    precomputed_rsi: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute technical indicators from OHLCV data.

    Args:
        ohlcv: shape (seq_len, 5) -- columns [open, high, low, close, volume]
        precomputed_rsi: optional precomputed RSI array of shape (seq_len,).
            When provided, skips the per-sample RSI calculation.

    Returns:
        np.ndarray of shape (seq_len, 7) with columns:
            returns, log_returns, volatility_5, sma_5, rsi_14,
            macd, high_low_range
    """
    close = ohlcv[:, 3]
    high = ohlcv[:, 1]
    low = ohlcv[:, 2]
    open_ = ohlcv[:, 0]
    seq_len = len(close)

    # 1. Returns
    returns = np.zeros(seq_len)
    returns[1:] = (close[1:] - close[:-1]) / np.where(
        close[:-1] == 0, 1.0, close[:-1]
    )

    # 2. Log returns
    log_returns = np.zeros(seq_len)
    safe_close = np.where(close <= 0, 1.0, close)
    log_returns[1:] = np.log(safe_close[1:] / safe_close[:-1])

    # 3. Rolling 5-day volatility of returns
    volatility_5 = _rolling_std(returns, window=5)

    # 4. SMA 5
    sma_5 = _rolling_mean(close, window=5)

    # 5. RSI 14 -- use precomputed if available
    if precomputed_rsi is not None:
        rsi_14 = precomputed_rsi
    else:
        rsi_14 = _compute_rsi(close, period=min(14, seq_len))

    # 6. MACD (12-EMA minus 26-EMA, approximated for short sequences)
    ema_12 = _ema(close, span=min(12, seq_len))
    ema_26 = _ema(close, span=min(26, seq_len))
    macd = ema_12 - ema_26

    # 7. High-low range normalised by close
    safe_close_hl = np.where(close == 0, 1.0, close)
    high_low_range = (high - low) / safe_close_hl

    indicators = np.column_stack([
        returns,
        log_returns,
        volatility_5,
        sma_5,
        rsi_14,
        macd,
        high_low_range,
    ])

    # Replace any remaining NaN with 0
    indicators = np.nan_to_num(indicators, nan=0.0)
    return indicators


def compute_temporal_features(dates: List[str]) -> np.ndarray:
    """
    Compute cyclical temporal features from date strings.

    Args:
        dates: List of date strings like "2023-01-01"

    Returns:
        np.ndarray of shape (seq_len, 4) with columns:
            dow_sin, dow_cos, month_sin, month_cos
    """
    features = []
    for date_str in dates:
        try:
            parts = date_str.split("-")
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            # Day of week: 0=Monday, 6=Sunday
            import datetime
            dow = datetime.date(year, month, day).weekday()
        except (ValueError, IndexError):
            dow = 0
            month = 1

        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)
        month_sin = np.sin(2 * np.pi * (month - 1) / 12)
        month_cos = np.cos(2 * np.pi * (month - 1) / 12)
        features.append([dow_sin, dow_cos, month_sin, month_cos])

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper functions for technical indicators
# ---------------------------------------------------------------------------

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with min_periods=1."""
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        out[i] = np.mean(arr[start : i + 1])
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation with min_periods=1."""
    out = np.zeros_like(arr)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        segment = arr[start : i + 1]
        out[i] = np.std(segment) if len(segment) > 1 else 0.0
    return out


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Compute RSI with Wilder smoothing, min_periods=1."""
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = _ema(gains, span=period)
    avg_loss = _ema(losses, span=period)

    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    sample: dict,
    feature_groups: List[str] = None,
    precomputed_rsi: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Build a complete feature matrix from a single dataloader sample.

    Args:
        sample: dict with keys: dates, time_series, target, ticker_text, ticker_id
        feature_groups: Which feature groups to include.
            Default: ["ohlcv", "technical", "temporal"]
        precomputed_rsi: optional precomputed RSI array for this sample's window.

    Returns:
        features: np.ndarray of shape (seq_len, num_features) or None if invalid
        target: np.ndarray of shape (forecast_horizon,) or None if invalid
        dates: list of date strings or None if invalid
    """
    if feature_groups is None:
        feature_groups = ["ohlcv", "technical", "temporal"]

    time_series = sample.get("time_series")
    target = sample.get("target")
    dates = sample.get("dates")

    if not time_series or not target:
        return None, None, None

    # Check for None values in target
    target_arr = np.array(
        [t if t is not None else np.nan for t in target], dtype=np.float64
    )
    if np.isnan(target_arr).any():
        return None, None, None

    ohlcv = extract_ohlcv_features(time_series)
    parts = []

    if "ohlcv" in feature_groups:
        parts.append(ohlcv)

    if "technical" in feature_groups:
        tech = compute_technical_indicators(ohlcv, precomputed_rsi=precomputed_rsi)
        parts.append(tech)

    if "temporal" in feature_groups and dates:
        temp = compute_temporal_features(dates)
        parts.append(temp)

    if not parts:
        return None, None, None

    features = np.concatenate(parts, axis=1)

    # Final NaN cleanup
    features = np.nan_to_num(features, nan=0.0)

    return features, target_arr, dates


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class LSTMTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset holding preprocessed time-series windows for LSTM training.

    Each sample is a dict containing:
        - features: Tensor of shape (seq_len, num_features)
        - targets: Tensor of shape (forecast_horizon,)
        - ticker_id: integer ticker ID
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        ticker_ids: np.ndarray,
        metadata: Optional[List[dict]] = None,
    ):
        """
        Args:
            features: shape (N, seq_len, num_features) -- already normalised
            targets: shape (N, forecast_horizon) -- already normalised
            ticker_ids: shape (N,) -- integer ticker IDs
            metadata: optional list of dicts with ticker_text, dates, etc.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.ticker_ids = torch.tensor(ticker_ids, dtype=torch.long)
        self.metadata = metadata or []

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "targets": self.targets[idx],
            "ticker_id": self.ticker_ids[idx],
        }


# ---------------------------------------------------------------------------
# Data processing helpers
# ---------------------------------------------------------------------------

def _precompute_rsi_for_tickers(
    dataloader: Dataset,
    max_workers: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute RSI once per ticker over its full close-price history.

    Groups all samples by ticker, extracts the close prices, builds a
    single contiguous close series (sorted by first date in each sample),
    and computes RSI over the full series.  Returns a dict mapping
    ticker_text -> full RSI array.

    The per-sample RSI slices are later looked up by matching dates.
    """
    from collections import defaultdict

    ticker_closes: Dict[str, List[Tuple[str, np.ndarray, List[str]]]] = defaultdict(list)

    n = len(dataloader)

    # Parallelize sample loading for RSI precomputation
    def load_sample_for_rsi(idx):
        sample = dataloader[idx]
        ticker = sample.get("ticker_text", "")
        ts = sample.get("time_series")
        dates = sample.get("dates")
        if not ts or not dates:
            return None
        close = np.array(
            [t.get("close", 0.0) or 0.0 for t in ts], dtype=np.float64
        )
        return (ticker, dates[0], close, dates)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(load_sample_for_rsi, idx) for idx in range(n)]
        for future in tqdm(as_completed(futures), total=n, desc="  Loading for RSI"):
            result = future.result()
            if result:
                ticker, first_date, close, dates = result
                ticker_closes[ticker].append((first_date, close, dates))

    ticker_rsi: Dict[str, Dict[str, float]] = {}
    for ticker, entries in ticker_closes.items():
        # Sort windows by first date to get temporal order
        entries.sort(key=lambda x: x[0])

        # Build a date -> index mapping from all unique dates
        date_to_close: Dict[str, float] = {}
        for _, close_arr, dates_list in entries:
            for d, c in zip(dates_list, close_arr):
                if d not in date_to_close:
                    date_to_close[d] = c

        # Sort dates and build full close series
        sorted_dates = sorted(date_to_close.keys())
        full_close = np.array([date_to_close[d] for d in sorted_dates], dtype=np.float64)

        # Compute RSI once over full series
        full_rsi = _compute_rsi(full_close, period=min(14, len(full_close)))

        # Build a date -> RSI value lookup
        date_to_rsi = {d: full_rsi[i] for i, d in enumerate(sorted_dates)}
        ticker_rsi[ticker] = date_to_rsi

    return ticker_rsi


def _load_and_process_sample(args):
    """Worker function for parallel sample loading and processing."""
    dataloader, idx, feature_groups, ticker_rsi = args

    # Load sample from disk (I/O operation)
    sample = dataloader[idx]

    # Extract precomputed RSI for this sample
    precomputed_rsi = None
    ticker = sample.get("ticker_text", "")
    if ticker in ticker_rsi:
        dates = sample.get("dates")
        if dates:
            rsi_values = ticker_rsi[ticker]
            precomputed_rsi = np.array(
                [rsi_values.get(d, 50.0) for d in dates], dtype=np.float64
            )

    # Compute features
    feat, tgt, dates = build_feature_matrix(
        sample, feature_groups, precomputed_rsi=precomputed_rsi
    )
    if feat is None:
        return None

    return {
        "features": feat,
        "target": tgt,
        "ticker_id": sample.get("ticker_id", 0),
        "metadata": {
            "ticker_text": sample.get("ticker_text", ""),
            "dates": dates,
            "first_date": dates[0] if dates else "",
        },
    }


def process_dataloader(
    dataloader: Dataset,
    feature_groups: List[str],
    max_workers: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """
    Iterate through a BaselineDataLoader and build feature matrices.

    Uses ThreadPoolExecutor to parallelize I/O and feature computation,
    and precomputes RSI once per ticker to avoid redundant work.

    Args:
        dataloader: A BaselineDataLoader instance.
        feature_groups: Which feature groups to include.
        max_workers: Number of parallel workers (default: CPU count * 2 for I/O).

    Returns:
        features: np.ndarray of shape (N, seq_len, num_features)
        targets: np.ndarray of shape (N, forecast_horizon)
        ticker_ids: np.ndarray of shape (N,)
        metadata: list of dicts with ticker_text, dates, first_date
    """
    n = len(dataloader)

    # --- Step 1: Precompute RSI per ticker (parallelized) ---
    compute_rsi = "technical" in feature_groups
    ticker_rsi = {}
    if compute_rsi:
        print("  Precomputing RSI per ticker...")
        ticker_rsi = _precompute_rsi_for_tickers(dataloader, max_workers=max_workers)

    # --- Step 2: Parallel I/O + feature computation ---
    # Use ThreadPoolExecutor since I/O is the bottleneck (reading JSONL files)
    # For I/O-bound tasks, threads are more efficient than processes
    print(f"  Loading and processing samples in parallel (workers={max_workers or 'auto'})...")

    # Prepare arguments for each sample
    args_list = [
        (dataloader, idx, feature_groups, ticker_rsi)
        for idx in range(n)
    ]

    all_features = []
    all_targets = []
    all_ticker_ids = []
    all_metadata = []

    # Use ThreadPoolExecutor for I/O-bound operations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_load_and_process_sample, args): i
            for i, args in enumerate(args_list)
        }
        results = [None] * len(args_list)
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Processing"):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                results[idx] = None

    for result in results:
        if result is None:
            continue
        all_features.append(result["features"])
        all_targets.append(result["target"])
        all_ticker_ids.append(result["ticker_id"])
        all_metadata.append(result["metadata"])

    features = np.array(all_features, dtype=np.float64)
    targets = np.array(all_targets, dtype=np.float64)
    ticker_ids = np.array(all_ticker_ids, dtype=np.int64)

    return features, targets, ticker_ids, all_metadata


def split_train_val(
    features: np.ndarray,
    targets: np.ndarray,
    ticker_ids: np.ndarray,
    metadata: List[dict],
    val_ratio: float = 0.15,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]],
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]],
]:
    """
    Split training data into train + validation sets.

    Takes the last ``val_ratio`` fraction of the data as validation.  The
    dataloader is assumed to already provide temporal ordering after the
    user's modification.

    Args:
        features, targets, ticker_ids, metadata: arrays from process_dataloader
        val_ratio: fraction of data to use for validation (default 0.15)

    Returns:
        (train_features, train_targets, train_ticker_ids, train_meta),
        (val_features, val_targets, val_ticker_ids, val_meta)
    """
    n = len(features)
    split_idx = int(n * (1 - val_ratio))

    train_data = (
        features[:split_idx],
        targets[:split_idx],
        ticker_ids[:split_idx],
        metadata[:split_idx],
    )
    val_data = (
        features[split_idx:],
        targets[split_idx:],
        ticker_ids[split_idx:],
        metadata[split_idx:],
    )

    return train_data, val_data


class _PerSampleTargetScaler:
    """
    Stores per-sample anchor prices so predictions in ratio-space
    can be converted back to dollar values via inverse_transform().
    """

    def __init__(self, anchors: np.ndarray):
        self.anchors = anchors  # shape (N,)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: shape (N * horizon, 1)  -- as called by evaluate()
        Returns:
            Same shape, multiplied by per-sample anchor prices.
        """
        n_samples = len(self.anchors)
        total = data.shape[0]
        horizon = total // n_samples
        reshaped = data.reshape(n_samples, horizon)
        result = reshaped * self.anchors[:, np.newaxis]
        return result.reshape(-1, 1)


def _normalize_split(
    features: np.ndarray,
    targets: np.ndarray,
    n_ohlcv: int,
    n_technical: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-sample normalization for one data split.

    - OHLCV prices (cols 0 .. n_ohlcv-2): divide by first close in window.
    - Volume (col n_ohlcv-1):             divide by mean volume in window.
    - Technical indicators (next n_technical cols): per-sample z-score.
    - Temporal / remaining cols:           left unchanged (already [-1, 1]).
    - Targets:                             divide by first close price.

    Returns:
        (norm_features, norm_targets, anchors)
    """
    n_samples, seq_len, n_feat = features.shape
    norm_features = features.copy()
    norm_targets = targets.copy()
    anchors = np.ones(n_samples)

    price_end = n_ohlcv - 1          # cols 0..3  (open, high, low, close)
    vol_col = n_ohlcv - 1            # col 4      (volume)
    tech_start = n_ohlcv             # col 5
    tech_end = n_ohlcv + n_technical  # col 12

    for i in range(n_samples):
        window = features[i]  # (seq_len, n_feat)

        # Anchor = first close price in the window (close is col 3)
        close_col = min(3, n_feat - 1)
        first_close = window[0, close_col]
        if first_close == 0 or np.isnan(first_close):
            first_close = 1.0
        anchors[i] = first_close

        # Normalize OHLCV price columns by first close
        if n_ohlcv >= 4:
            norm_features[i, :, 0:4] = window[:, 0:4] / first_close

        # Normalize volume by mean volume in window
        if n_ohlcv >= 5:
            mean_vol = np.mean(np.abs(window[:, 4]))
            if mean_vol == 0 or np.isnan(mean_vol):
                mean_vol = 1.0
            norm_features[i, :, 4] = window[:, 4] / mean_vol

        # Per-sample z-score for technical indicator columns
        for col in range(tech_start, min(tech_end, n_feat)):
            col_data = window[:, col]
            col_std = np.std(col_data)
            col_mean = np.mean(col_data)
            if col_std == 0 or np.isnan(col_std):
                norm_features[i, :, col] = 0.0
            else:
                norm_features[i, :, col] = (col_data - col_mean) / col_std

        # Temporal features (remaining cols) stay unchanged

        # Normalize targets by first close
        norm_targets[i] = targets[i] / first_close

    return norm_features, norm_targets, anchors


def normalize_features(
    train_features: np.ndarray,
    val_features: np.ndarray,
    test_features: np.ndarray,
    train_targets: np.ndarray,
    val_targets: np.ndarray,
    test_targets: np.ndarray,
    feature_groups: list = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    object, object,
]:
    """
    Per-sample normalization: each window is normalized relative to itself.

    OHLCV prices  -> divide by first close price in window
    Volume        -> divide by mean volume in window
    Technical     -> per-sample z-score
    Temporal      -> unchanged (already [-1, 1])
    Targets       -> divide by first close price

    Returns:
        Normalised arrays + (feature_scaler=None, target_scaler) for
        API compatibility.  target_scaler supports inverse_transform().
    """
    if feature_groups is None:
        feature_groups = ["ohlcv", "technical", "temporal"]

    # Determine column counts based on which feature groups are present
    n_ohlcv = 5 if "ohlcv" in feature_groups else 0
    n_technical = 7 if "technical" in feature_groups else 0

    train_features, train_targets, _ = _normalize_split(
        train_features, train_targets, n_ohlcv, n_technical
    )

    val_anchors = np.array([])
    if len(val_features) > 0:
        val_features, val_targets, val_anchors = _normalize_split(
            val_features, val_targets, n_ohlcv, n_technical
        )

    test_anchors = np.array([])
    if len(test_features) > 0:
        test_features, test_targets, test_anchors = _normalize_split(
            test_features, test_targets, n_ohlcv, n_technical
        )

    feature_scaler = None
    target_scaler = _PerSampleTargetScaler(
        test_anchors if len(test_anchors) > 0 else np.array([1.0])
    )

    return (
        train_features, val_features, test_features,
        train_targets, val_targets, test_targets,
        feature_scaler, target_scaler,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_lstm_data(
    config: dict,
) -> Tuple[
    LSTMTimeSeriesDataset,
    LSTMTimeSeriesDataset,
    LSTMTimeSeriesDataset,
    StandardScaler,
    StandardScaler,
]:
    """
    Main preprocessing entry point.

    Pipeline:
        1. Load train/test data via getTrainTestDataLoader
        2. Process each split through feature engineering
        3. Split training data into train + validation
        4. Normalise (fit on train only)
        5. Wrap in LSTMTimeSeriesDataset

    Args:
        config: dict with keys:
            - data_path: str, path to baseline_data folder
            - ticker2idx: dict, ticker -> id mapping (or loaded internally)
            - val_ratio: float (default 0.15)
            - feature_groups: list of str (default ["ohlcv", "technical", "temporal"])
            - yaml_config_path: str, path to config.yaml (optional)

    Returns:
        train_dataset, val_dataset, test_dataset,
        feature_scaler, target_scaler
    """
    feature_groups = config.get(
        "feature_groups", ["ohlcv", "technical", "temporal"]
    )
    val_ratio = config.get("val_ratio", 0.15)

    # Load data via existing dataloader
    print("Loading data via BaselineDataLoader...")
    train_loader, test_loader = getTrainTestDataLoader(config)

    # Process train and test splits
    print("\nProcessing training data...")
    train_feat, train_tgt, train_ids, train_meta = process_dataloader(
        train_loader, feature_groups
    )
    print(f"  Training samples: {len(train_feat)}")
    print(f"  Feature shape: {train_feat.shape}")

    print("\nProcessing test data...")
    test_feat, test_tgt, test_ids, test_meta = process_dataloader(
        test_loader, feature_groups
    )
    print(f"  Test samples: {len(test_feat)}")

    # Split training data into train + validation
    print(f"\nSplitting training data (val_ratio={val_ratio})...")
    (train_feat, train_tgt, train_ids, train_meta), \
    (val_feat, val_tgt, val_ids, val_meta) = split_train_val(
        train_feat, train_tgt, train_ids, train_meta, val_ratio
    )
    print(f"  Train: {len(train_feat)}, Val: {len(val_feat)}, Test: {len(test_feat)}")

    # Normalise
    print("\nNormalising features and targets (per-sample)...")
    (
        train_feat, val_feat, test_feat,
        train_tgt, val_tgt, test_tgt,
        feature_scaler, target_scaler,
    ) = normalize_features(
        train_feat, val_feat, test_feat,
        train_tgt, val_tgt, test_tgt,
        feature_groups=feature_groups,
    )

    # Wrap in datasets
    train_dataset = LSTMTimeSeriesDataset(
        train_feat, train_tgt, train_ids, train_meta
    )
    val_dataset = LSTMTimeSeriesDataset(
        val_feat, val_tgt, val_ids, val_meta
    )
    test_dataset = LSTMTimeSeriesDataset(
        test_feat, test_tgt, test_ids, test_meta
    )

    print(f"\nDatasets ready:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    print(f"  Features per timestep: {train_feat.shape[2]}")

    return train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler


# ---------------------------------------------------------------------------
# Driver / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    # Ensure project root is on path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    os.chdir(str(project_root))

    from src.utils import read_json_file, read_yaml, set_seed

    set_seed(42)

    yaml_config = read_yaml("config/config.yaml")
    ticker2idx = read_json_file(
        os.path.join(yaml_config["BASELINE_DATA_PATH"], yaml_config["TICKER2IDX"])
    )

    config = {
        "data_path": yaml_config["BASELINE_DATA_PATH"],
        "ticker2idx": ticker2idx,
        "test_train_split": 0.2,
        "random_seed": 42,
        "val_ratio": 0.15,
        "feature_groups": ["ohlcv", "technical", "temporal"],
    }

    train_ds, val_ds, test_ds, f_scaler, t_scaler = prepare_lstm_data(config)

    # Verify no NaN
    sample = train_ds[0]
    assert not torch.isnan(sample["features"]).any(), "NaN found in features"
    assert not torch.isnan(sample["targets"]).any(), "NaN found in targets"
    print("\nSmoke test passed: no NaN in first sample.")
    print(f"  Feature tensor shape: {sample['features'].shape}")
    print(f"  Target tensor shape:  {sample['targets'].shape}")
