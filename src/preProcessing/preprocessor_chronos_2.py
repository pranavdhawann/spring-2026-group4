"""Preprocessing utilities for time series data.

Handles missing-value imputation, frequency detection, and seasonality
mapping before the data is passed to the forecasting model.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.vocab.constants_chronos_2 import FREQ_TO_SEASONALITY_MAP

logger = logging.getLogger(__name__)


def detect_frequency(timestamps: pd.Series) -> str:
    """Infer the frequency of a datetime series.

    Args:
        timestamps: A sorted ``datetime64`` Series.

    Returns:
        A pandas frequency alias string (e.g. ``"H"``, ``"D"``).

    Raises:
        ValueError: If the frequency cannot be inferred automatically.
    """
    freq = pd.infer_freq(timestamps)
    if freq is None:
        raise ValueError(
            "Could not auto-detect time series frequency. "
            "Please specify 'frequency' explicitly in config/dataset_chronos_2.yaml."
        )
    logger.info("Detected frequency: %s", freq)
    return freq


def map_frequency_to_seasonality(frequency: str) -> int:
    """Map a pandas frequency alias to its seasonal period.

    Args:
        frequency: Pandas frequency string (e.g. ``"H"``, ``"D"``).

    Returns:
        Integer seasonal period used for MASE computation.
    """
    # Strip any numeric prefix (e.g. "2H" -> "H")
    base_freq = frequency.lstrip("0123456789")

    if base_freq in FREQ_TO_SEASONALITY_MAP:
        seasonality = FREQ_TO_SEASONALITY_MAP[base_freq]
    else:
        warnings.warn(
            f"Unknown frequency '{frequency}' — defaulting seasonality to 1.",
            stacklevel=2,
        )
        seasonality = 1

    logger.info("Mapped frequency '%s' to seasonality %d", frequency, seasonality)
    return seasonality


def _count_max_consecutive_nans(series: pd.Series) -> int:
    """Return the length of the longest consecutive-NaN run in *series*."""
    is_nan = series.isna()
    if not is_nan.any():
        return 0
    groups = is_nan.ne(is_nan.shift()).cumsum()
    return int(is_nan.groupby(groups).sum().max())


def handle_missing_values(
    df: pd.DataFrame,
    timestamp_column: str,
    method: str = "ffill_bfill",
    max_gap: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """Impute or drop columns with missing values.

    Args:
        df: Input DataFrame (timestamp + target columns).
        timestamp_column: Name of the timestamp column.
        method: Imputation strategy. One of ``"ffill_bfill"``,
            ``"interpolate"``, or ``"drop"``.
        max_gap: Maximum consecutive NaN gap allowed. Columns exceeding
            this threshold are dropped with a warning.

    Returns:
        A tuple of (cleaned DataFrame, list of dropped column names).
    """
    target_cols = [c for c in df.columns if c != timestamp_column]
    dropped: List[str] = []

    for col in target_cols:
        gap = _count_max_consecutive_nans(df[col])
        if gap > max_gap:
            logger.warning(
                "Column '%s' has %d consecutive NaNs (max_gap=%d) — dropping.",
                col,
                gap,
                max_gap,
            )
            dropped.append(col)

    if dropped:
        df = df.drop(columns=dropped)

    # Impute remaining NaNs
    remaining_targets = [c for c in df.columns if c != timestamp_column]
    if method == "ffill_bfill":
        df[remaining_targets] = df[remaining_targets].ffill().bfill()
    elif method == "interpolate":
        df[remaining_targets] = (
            df[remaining_targets].interpolate(method="linear").bfill().ffill()
        )
    elif method == "drop":
        df = df.dropna(subset=remaining_targets).reset_index(drop=True)
    else:
        raise ValueError(f"Unknown NaN handling method: '{method}'")

    return df, dropped


def preprocess(
    df: pd.DataFrame,
    dataset_config: Dict,
) -> Tuple[pd.DataFrame, str, int]:
    """Run the full preprocessing pipeline.

    Args:
        df: Raw DataFrame from
            :func:`src.data.csv_reader_chronos_2.read_csv`.
        dataset_config: The ``dataset`` section of the YAML config, containing
            keys like ``timestamp_column``, ``frequency``, and ``nan_handling``.

    Returns:
        A tuple of:
            - Cleaned DataFrame.
            - Frequency string (e.g. ``"H"``).
            - Seasonality integer (e.g. ``24``).
    """
    ts_col: str = dataset_config.get("timestamp_column", "ds")
    freq_override: Optional[str] = dataset_config.get("frequency")
    nan_cfg: Dict = dataset_config.get("nan_handling", {})
    nan_method: str = nan_cfg.get("method", "ffill_bfill")
    max_gap: int = nan_cfg.get("max_gap", 5)

    # 1. Handle missing values
    df, dropped = handle_missing_values(df, ts_col, nan_method, max_gap)
    if dropped:
        logger.info("Dropped columns due to excessive NaNs: %s", dropped)

    # 2. Detect or use provided frequency
    if freq_override:
        frequency = freq_override
        logger.info("Using user-specified frequency: %s", frequency)
    else:
        frequency = detect_frequency(df[ts_col])

    # 3. Map to seasonality
    seasonality = map_frequency_to_seasonality(frequency)

    logger.info(
        "Preprocessing complete — %d rows, %d series, freq=%s, seasonality=%d",
        len(df),
        len([c for c in df.columns if c != ts_col]),
        frequency,
        seasonality,
    )
    return df, frequency, seasonality
