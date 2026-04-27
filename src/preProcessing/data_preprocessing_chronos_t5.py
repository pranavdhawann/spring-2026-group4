"""
Data preprocessing pipeline for Chronos time-series forecasting.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.utils import load_stock_csv, read_json_file

logger = logging.getLogger(__name__)
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]


def preprocess_ticker_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
        df["Date"] = df["Date"].dt.tz_localize(None)
        df = df.sort_values("Date").set_index("Date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a 'Date' column or a DatetimeIndex.")

    existing_ohlcv = [c for c in OHLCV_COLS if c in df.columns]
    df[existing_ohlcv] = df[existing_ohlcv].ffill().bfill()

    df = df.resample("B").ffill().bfill()

    if "Close" in df.columns:
        df["Returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Returns"] = df["Returns"].fillna(0.0)

    if "Volume" in df.columns:
        vol_mean = df["Volume"].mean()
        vol_std = df["Volume"].std()
        if vol_std > 0:
            df["Volume"] = (df["Volume"] - vol_mean) / vol_std
        else:
            df["Volume"] = 0.0

    return df


def load_ticker(path: str, target_col: str = "Close") -> Tuple[np.ndarray, np.ndarray]:
    csv_path = Path(path)
    ticker = csv_path.stem
    parent_dir = csv_path.parent

    df = load_stock_csv(ticker, parent_dir)
    df = preprocess_ticker_df(df)

    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. " f"Available: {list(df.columns)}"
        )

    dates = df.index.values.astype("datetime64[D]")
    values = df[target_col].values.astype(np.float64)
    return dates, values


def load_all_tickers(
    manifest_path: str,
    data_dir: str,
    target_col: str = "Close",
) -> List[dict]:
    manifest = read_json_file(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Could not read manifest at {manifest_path}")

    data_dir = Path(data_dir)
    results: List[dict] = []

    for entry in manifest:
        ticker = entry["ticker"]
        file_path = entry.get("file_path")

        if file_path and Path(file_path).exists():
            csv_path = str(file_path)
        else:
            csv_path = str(data_dir / f"{ticker.lower()}.csv")

        try:
            dates, values = load_ticker(csv_path, target_col=target_col)
            results.append({"ticker": ticker, "dates": dates, "values": values})
        except Exception as exc:
            logger.warning("Skipping ticker %s: %s", ticker, exc)

    return results


def _print_stats(ticker: str, dates: np.ndarray, values: np.ndarray) -> None:
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
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--target_col", type=str, default="Close")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = project_root / "data" / "multi-modal-dataset" / "sp500_time_series"

    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    if args.manifest and Path(args.manifest).exists():
        print(f"Loading tickers from manifest: {args.manifest}")
        all_data = load_all_tickers(
            args.manifest, str(data_dir), target_col=args.target_col
        )
        for item in all_data[: args.n]:
            _print_stats(item["ticker"], item["dates"], item["values"])
    else:
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
