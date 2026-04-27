"""

run this file from project root directory not from setupScripts

"""
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import read_jsonl, read_yaml

W_MIN = 360 * 10  # minimum trainable window (days)
C_MIN = 0.25  # min coverage inside window
G_MAX = 14  # max allowed p90 gap (days)

C_TARGET = 0.40
G_TARGET = 5
G90_TARGET = 14
V_TARGET = 1000
W_TARGET = 365

config = read_yaml("config/config.yaml")
data_dictionary_df = pd.read_csv(config["DATA_DICTIONARY"])
data_dictionary_df["Sector"] = data_dictionary_df["Sector"].fillna("N/A")
ticker_to_sector = data_dictionary_df.set_index("stock_name")["Sector"].to_dict()


def build_daily_series(df):
    daily = df.groupby("date").size().rename("count").to_frame()

    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0)
    daily.index.name = "date"
    return daily


def compute_gap_metrics(active_dates):
    if len(active_dates) < 2:
        return np.inf, np.inf

    gaps = np.diff(active_dates).astype("timedelta64[D]").astype(int)
    return gaps.mean(), np.percentile(gaps, 90)


def compute_entropy(counts):
    probs = counts[counts > 0] / counts.sum()
    entropy = -np.sum(probs * np.log(probs))
    return entropy / np.log(len(probs)) if len(probs) > 1 else 0.0


def find_best_window(daily, window_min=W_MIN):
    best = None

    active_days = daily[daily["count"] > 0]

    for start in active_days.index:
        end = start + timedelta(days=window_min)

        if end > daily.index.max():
            break

        window = daily.loc[start:end]
        coverage = (window["count"] > 0).mean()

        active = window[window["count"] > 0]
        if len(active) < 2:
            continue

        avg_gap, p90_gap = compute_gap_metrics(active.index.values)

        if coverage >= C_MIN and p90_gap <= G_MAX:
            quality = (
                0.4 * coverage
                + 0.3 * (1 - avg_gap / G_MAX)
                + 0.2 * compute_entropy(window["count"])
                + 0.1 * math.log(window["count"].sum() + 1)
            )

            if best is None or quality > best["quality"]:
                best = {
                    "start": start,
                    "end": end,
                    "quality": quality,
                    "window_days": (end - start).days,
                }

    return best


def score_ticker(daily):
    total_days = len(daily)
    active_days = daily[daily["count"] > 0]

    if len(active_days) < 2:
        return None

    coverage_ratio = len(active_days) / total_days
    avg_gap, p90_gap = compute_gap_metrics(active_days.index.values)
    entropy = compute_entropy(daily["count"])
    total_articles = daily["count"].sum()

    window = find_best_window(daily)
    if window is None:
        return None

    # Normalized scores
    coverage_score = min(coverage_ratio / C_TARGET, 1)
    gap_score = np.exp(-avg_gap / G_TARGET)
    p90_gap_score = np.exp(-p90_gap / G90_TARGET)
    volume_score = min(np.log(total_articles + 1) / np.log(V_TARGET), 1)
    window_score = min(window["window_days"] / W_TARGET, 1)

    final_score = (
        0.25 * coverage_score
        + 0.20 * gap_score
        + 0.15 * p90_gap_score
        + 0.15 * entropy
        + 0.15 * volume_score
        + 0.10 * window_score
    )

    return {
        "score": final_score,
        "coverage": coverage_ratio,
        "avg_gap": avg_gap,
        "p90_gap": p90_gap,
        "entropy": entropy,
        "total_articles": total_articles,
        "window": window,
    }


def remove_jsonl_from_ticker(ticker):
    ticker = ticker.replace(".jsonl", "")
    return ticker


results = []
list_of_tickers = os.listdir(config["NEWS_FOLDER"])
list_of_tickers.sort()

print("__________ processing ___________")
for ticker in tqdm(list_of_tickers):
    data = read_jsonl(os.path.join(config["NEWS_FOLDER"], ticker))
    ticker = remove_jsonl_from_ticker(ticker)
    records = []
    for article in data:
        records.append(article["Date"])
    df = pd.DataFrame({"date": pd.to_datetime(records)})
    daily = build_daily_series(df)

    metrics = score_ticker(daily)
    if metrics:
        results.append({"ticker": ticker, **metrics})


def results_to_dataframe(results):
    rows = []

    for r in results:
        row = {
            "ticker": r["ticker"],
            "score": float(r["score"]),
            "coverage": float(r["coverage"]),
            "avg_gap": float(r["avg_gap"]),
            "p90_gap": float(r["p90_gap"]),
            "entropy": float(r["entropy"]),
            "total_articles": int(r["total_articles"]),
            "sector": ticker_to_sector[r["ticker"].lower()]
            if r["ticker"].lower() in ticker_to_sector
            else "N/A",
        }

        if r.get("window"):
            row.update(
                {
                    "window_start": r["window"]["start"],
                    "window_end": r["window"]["end"],
                    "window_days": int(r["window"]["window_days"]),
                    "window_quality": float(r["window"]["quality"]),
                }
            )
        else:
            row.update(
                {
                    "window_start": pd.NaT,
                    "window_end": pd.NaT,
                    "window_days": 0,
                    "window_quality": np.nan,
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


results_df = results_to_dataframe(results)
results_df = results_df.sort_values(["score", "coverage", "entropy"], ascending=False)

results_df.to_csv(config["STOCK_SCORE_NEWS"], index=False)
print(
    f"Sucess ! the list of stocks with scores is saved at {config['STOCK_SCORE_NEWS']}"
)
