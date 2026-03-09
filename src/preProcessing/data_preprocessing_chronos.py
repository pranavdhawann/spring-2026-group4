"""
Data preprocessing pipeline for Chronos time-series forecasting.

Loads ticker CSVs (produced by the data collection step), cleans OHLCV
columns, resamples to business-day frequency, computes log returns, and
z-score normalises volume.  Exposes two public helpers:

    load_ticker(path, target_col)       -> (dates, values)
    load_all_tickers(manifest, dir, …)  -> list[dict]

Run this file directly for a quick sanity-check on a few tickers.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make sure project root is importable regardless of cwd
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.utils import load_stock_csv, read_json_file

logger = logging.getLogger(__name__)

# Columns that receive fill / resample treatment
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------


def preprocess_ticker_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a single ticker DataFrame and return the enriched version.

    Steps
    -----
    1. Sort by Date, set Date as the index.
    2. Forward-fill then back-fill NaN values in OHLCV columns.
    3. Resample to business-day frequency (``'B'``), forward-filling gaps.
    4. Compute ``Returns`` as log-returns of Close.
    5. Z-score normalise ``Volume`` (per-ticker, in-place).

    Parameters
    ----------
    df : pd.DataFrame
        Raw ticker data with at least columns Date, Open, High, Low, Close,
        Volume.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame indexed by business-day dates.
    """
    df = df.copy()

    # Ensure Date is datetime and set as index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df["Date"] = df["Date"].dt.tz_localize(None)
        df = df.sort_values("Date").set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a 'Date' column or a DatetimeIndex.")

    # 1. Forward-fill then back-fill OHLCV NaNs
    existing_ohlcv = [c for c in OHLCV_COLS if c in df.columns]
    df[existing_ohlcv] = df[existing_ohlcv].ffill().bfill()

    # 2. Resample to business-day frequency, forward-fill gaps
    df = df.resample("B").ffill().bfill()

    # 3. Log returns of Close
    if "Close" in df.columns:
        df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Returns"] = df["Returns"].fillna(0.0)

    # 4. Z-score normalise Volume
    if "Volume" in df.columns:
        vol_mean = df["Volume"].mean()
        vol_std = df["Volume"].std()
        if vol_std > 0:
            df["Volume"] = (df["Volume"] - vol_mean) / vol_std
        else:
            df["Volume"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_ticker(
    path: str,
    target_col: str = "Close",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess a single ticker CSV.

    Parameters
    ----------
    path : str
        Absolute or relative path to the CSV file.
    target_col : str, optional
        Column to extract as the value array (default ``'Close'``).

    Returns
    -------
    dates : np.ndarray
        Array of ``datetime64`` dates.
    values : np.ndarray
        Array of ``float64`` values for *target_col*.
    """
    csv_path = Path(path)
    ticker = csv_path.stem
    parent_dir = csv_path.parent

    df = load_stock_csv(ticker, parent_dir)
    df = preprocess_ticker_df(df)

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    dates = df.index.values.astype("datetime64[D]")
    values = df[target_col].values.astype(np.float64)
    return dates, values


def load_all_tickers(
    manifest_path: str,
    data_dir: str,
    target_col: str = "Close",
) -> List[dict]:
    """Load every ticker listed in the filter-step manifest.

    Parameters
    ----------
    manifest_path : str
        Path to the JSON manifest produced by ``data_filter_chronos.py``.
    data_dir : str
        Root directory containing ticker CSVs (used as fallback when
        ``file_path`` is missing from a manifest entry).
    target_col : str, optional
        Column to extract (default ``'Close'``).

    Returns
    -------
    list[dict]
        Each dict has keys ``'ticker'`` (str), ``'dates'`` (np.ndarray of
        datetime64), and ``'values'`` (np.ndarray of float64).  Series may
        have different lengths (variable-length).
    """
    manifest = read_json_file(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Could not read manifest at {manifest_path}")

    data_dir = Path(data_dir)
    results: List[dict] = []

    for entry in manifest:
        ticker = entry["ticker"]
        file_path = entry.get("file_path")

        # Resolve the CSV path
        if file_path and Path(file_path).exists():
            csv_path = str(file_path)
        else:
            csv_path = str(data_dir / f"{ticker.lower()}.csv")

        try:
            dates, values = load_ticker(csv_path, target_col=target_col)
            results.append(
                {
                    "ticker": ticker,
                    "dates": dates,
                    "values": values,
                }
            )
        except Exception as exc:
            logger.warning("Skipping ticker %s: %s", ticker, exc)

    return results


# ---------------------------------------------------------------------------
# Sanity-check entry point
# ---------------------------------------------------------------------------


def _print_stats(ticker: str, dates: np.ndarray, values: np.ndarray) -> None:
    """Pretty-print shape & summary statistics for one ticker."""
    print(f"\n{'=' * 50}")
    print(f"Ticker : {ticker}")
    print(f"Shape  : {values.shape}")
    print(f"Range  : {dates[0]}  ->  {dates[-1]}")
    print(f"Min    : {values.min():.6f}")
    print(f"Max    : {values.max():.6f}")
    print(f"Mean   : {values.mean():.6f}")
    print(f"Std    : {values.std():.6f}")
    print(f"First 5: {values[:5]}")
    print(f"NaN?   : {np.isnan(values).sum()}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanity-check Chronos data preprocessing on a few tickers."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to the JSON manifest from the filter step.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing ticker CSVs.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="Close",
        help="Target column to extract (default: Close).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of tickers to sanity-check (default: 3).",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------
    # Resolve data directory
    # -------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent.parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / "data" / "multi-modal-dataset" / "sp500_time_series"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # -------------------------------------------------------------------
    # If a manifest is provided, use load_all_tickers
    # -------------------------------------------------------------------
    if args.manifest and Path(args.manifest).exists():
        print(f"Loading tickers from manifest: {args.manifest}")
        all_data = load_all_tickers(
            args.manifest, str(data_dir), target_col=args.target_col
        )
        for item in all_data[: args.n]:
            _print_stats(item["ticker"], item["dates"], item["values"])

    else:
        # Fallback: pick 3 well-known tickers that are likely present
        fallback_tickers = ["aapl", "msft", "goog"]
        print("No manifest provided -- falling back to hard-coded tickers.")

        for ticker in fallback_tickers[: args.n]:
            csv_path = data_dir / f"{ticker}.csv"
            if not csv_path.exists():
                print(f"  WARNING: {csv_path} not found, skipping.")
                continue
            dates, values = load_ticker(str(csv_path), target_col=args.target_col)
            _print_stats(ticker, dates, values)

    print("\nSanity check complete.")
