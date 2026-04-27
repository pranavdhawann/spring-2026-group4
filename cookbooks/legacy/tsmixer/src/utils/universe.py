"""Universe selection: rank tickers by liquidity over a date window, take top N."""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def _load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["Date", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    return df


def liquidity_score(df: pd.DataFrame, start: str, end: str) -> Tuple[float, int]:
    """Median daily dollar-volume in window, plus number of days present."""
    mask = (df["Date"] >= pd.Timestamp(start)) & (df["Date"] < pd.Timestamp(end))
    w = df.loc[mask]
    if w.empty:
        return 0.0, 0
    dv = (w["Close"].astype(float) * w["Volume"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
    if dv.empty:
        return 0.0, int(len(w))
    return float(dv.median()), int(len(w))


def select_universe(
    data_dir: Path,
    start: str,
    end: str,
    top_n: int,
    min_days: int = 252,
) -> List[str]:
    """Return top-N ticker stems (filename without .csv) by median dollar volume in [start, end)."""
    rows = []
    for p in sorted(Path(data_dir).glob("*.csv")):
        try:
            df = _load_ohlcv(p)
        except Exception:
            continue
        score, n_days = liquidity_score(df, start, end)
        if n_days >= min_days and score > 0.0:
            rows.append((p.stem, score, n_days))
    if not rows:
        return []
    ranked = pd.DataFrame(rows, columns=["ticker", "dollar_vol", "days"]).sort_values(
        "dollar_vol", ascending=False
    )
    return ranked["ticker"].head(top_n).tolist()
