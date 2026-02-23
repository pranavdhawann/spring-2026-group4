"""
TabNet multi-modal preprocessing: one function to build a feature vector
from articles, time_series, and table_data. Column names and text_dim
come from config (e.g. config/tabnet_config.yaml under preprocessing).
"""
import time
from typing import Dict, List

import numpy as np


def preprocessTabNetMMBaseline(
    articles: List,
    time_series: List[Dict],
    table_data: List[Dict],
    config: Dict,
    verbose: bool = False,
):
    st_ = time.time()
    pre = config.get("preprocessing") or config
    ts_cols = pre.get("ts_cols")
    table_cols = pre.get("table_cols")
    text_dim = pre.get("text_dim") or 64
    if ts_cols is None:
        ts_cols = ["open", "high", "low", "close", "volume", "dividends", "stock splits"]
    if table_cols is None:
        table_cols = [
            "us-gaap_Assets",
            "us-gaap_AssetsCurrent",
            "us-gaap_Liabilities",
            "us-gaap_LiabilitiesCurrent",
            "us-gaap_StockholdersEquity",
            "us-gaap_NetCashProvidedByUsedInOperatingActivities",
            "us-gaap_RetainedEarningsAccumulatedDeficit",
        ]

    # Text: flatten articles and simple stats (no external model)
    texts = []
    for day_articles in articles or []:
        if not isinstance(day_articles, list):
            continue
        for a in day_articles:
            if isinstance(a, str) and a.strip():
                texts.append(a.strip())
    text_feats = np.zeros(text_dim, dtype=np.float32)
    if texts:
        lengths = [len(t.split()) for t in texts[:100]]
        text_feats[0] = len(texts)
        text_feats[1] = float(np.mean(lengths)) if lengths else 0.0
        text_feats[2] = float(np.std(lengths)) if len(lengths) > 1 else 0.0

    # Time series: min/mean/max per column
    values = {col: [] for col in ts_cols}
    for day in time_series or []:
        for col in ts_cols:
            v = day.get(col) if isinstance(day, dict) else None
            if v is not None and isinstance(v, (int, float)):
                values[col].append(float(v))
    ts_feats = []
    for col in ts_cols:
        arr = np.array(values[col]) if values[col] else np.array([0.0])
        ts_feats.extend([float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))])
    ts_feats = np.array(ts_feats, dtype=np.float32)

    # Table: last row, one value per column
    table_feats = np.zeros(len(table_cols), dtype=np.float32)
    if table_data:
        last = table_data[-1]
        for i, col in enumerate(table_cols):
            v = last.get(col)
            table_feats[i] = float(v) if v is not None and isinstance(v, (int, float)) else 0.0

    out = np.concatenate([text_feats, ts_feats, table_feats]).astype(np.float32)
    if verbose:
        print(" TabNet MM preprocess : ", round(time.time() - st_, 3), "s")
    return out
