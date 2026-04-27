"""
Evaluate Chronos batch forecasts against realized prices.

Outputs:
1) per_ticker_metrics.csv
2) aggregate_metrics.csv
3) top10_forecast_plot.png
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preProcessing.data_preprocessing_chronos_t5 import load_ticker
from src.utils.metrics_utils import calculate_regression_metrics, print_metrics


def _resolve_history_csv(
    data_dir: Path, ticker: str, file_path: Optional[str]
) -> Optional[Path]:
    if file_path:
        p = Path(file_path)
        if p.exists():
            return p

    candidates = [
        data_dir / f"{ticker}.csv",
        data_dir / f"{ticker.lower()}.csv",
        data_dir / f"{ticker.upper()}.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _resolve_forecast_csv(forecast_dir: Path, ticker: str) -> Optional[Path]:
    candidates = [
        forecast_dir / f"{ticker}_forecast.csv",
        forecast_dir / f"{ticker.lower()}_forecast.csv",
        forecast_dir / f"{ticker.upper()}_forecast.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    matches = list(forecast_dir.glob(f"*{ticker}*_forecast.csv"))
    return matches[0] if matches else None


def _pick_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def _to_close_from_log_returns(last_close: float, log_returns: pd.Series) -> pd.Series:
    vals = pd.to_numeric(log_returns, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return pd.Series(last_close * np.exp(np.cumsum(vals)))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chronos batch forecasts.")
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to manifest JSON used for forecasting",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing raw CSV files"
    )
    parser.add_argument(
        "--forecast_dir",
        type=str,
        required=True,
        help="Directory with per-ticker forecast CSV files",
    )
    parser.add_argument(
        "--forecast_target_col",
        type=str,
        default="Returns",
        choices=["Returns", "Close", "returns", "close"],
        help="Target used during forecasting run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write metrics CSVs (default: parent of forecast_dir)",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    data_dir = Path(args.data_dir)
    forecast_dir = Path(args.forecast_dir)
    output_dir = Path(args.output_dir) if args.output_dir else forecast_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not forecast_dir.exists():
        raise FileNotFoundError(f"Forecast directory not found: {forecast_dir}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    forecast_is_returns = str(args.forecast_target_col).lower() == "returns"

    records = []
    skipped = 0
    plot_payload = {}

    for entry in manifest:
        ticker = str(entry.get("ticker", "UNKNOWN"))
        history_csv = _resolve_history_csv(data_dir, ticker, entry.get("file_path"))
        fcast_csv = _resolve_forecast_csv(forecast_dir, ticker)

        if history_csv is None or fcast_csv is None:
            skipped += 1
            continue

        try:
            close_dates, close_vals = load_ticker(str(history_csv), target_col="Close")
        except Exception:
            skipped += 1
            continue

        hist_df = pd.DataFrame(
            {"Date": pd.to_datetime(close_dates), "Actual_Close": close_vals}
        )
        fcast_df = pd.read_csv(fcast_csv)
        if "Date" not in fcast_df.columns:
            skipped += 1
            continue
        fcast_df["Date"] = pd.to_datetime(fcast_df["Date"])
        fcast_df = fcast_df.sort_values("Date").reset_index(drop=True)

        pred_close_col = _pick_col(fcast_df, ["Point_Forecast_Close"])
        low_close_col = _pick_col(fcast_df, ["Quantile_10_Close", "Forecast_P10_Close"])
        high_close_col = _pick_col(
            fcast_df, ["Quantile_90_Close", "Forecast_P90_Close"]
        )
        point_col = _pick_col(fcast_df, ["Point_Forecast", "Forecast_Median"])
        low_col = _pick_col(fcast_df, ["Quantile_10", "Forecast_P10"])
        high_col = _pick_col(fcast_df, ["Quantile_90", "Forecast_P90"])

        if point_col is None and pred_close_col is None:
            skipped += 1
            continue

        prior_hist = hist_df[hist_df["Date"] < fcast_df["Date"].min()]
        if prior_hist.empty:
            skipped += 1
            continue
        last_close = float(prior_hist["Actual_Close"].iloc[-1])

        if pred_close_col is not None:
            pred_close = pd.to_numeric(fcast_df[pred_close_col], errors="coerce")
        elif forecast_is_returns:
            pred_close = _to_close_from_log_returns(last_close, fcast_df[point_col])
        else:
            pred_close = pd.to_numeric(fcast_df[point_col], errors="coerce")

        low_close = None
        high_close = None
        if low_close_col and high_close_col:
            low_close = pd.to_numeric(fcast_df[low_close_col], errors="coerce")
            high_close = pd.to_numeric(fcast_df[high_close_col], errors="coerce")
        elif low_col and high_col:
            if forecast_is_returns:
                low_close = _to_close_from_log_returns(last_close, fcast_df[low_col])
                high_close = _to_close_from_log_returns(last_close, fcast_df[high_col])
            else:
                low_close = pd.to_numeric(fcast_df[low_col], errors="coerce")
                high_close = pd.to_numeric(fcast_df[high_col], errors="coerce")

        pred_df = pd.DataFrame({"Date": fcast_df["Date"], "Pred_Close": pred_close})
        if low_close is not None and high_close is not None:
            pred_df["Low_Close"] = low_close.to_numpy()
            pred_df["High_Close"] = high_close.to_numpy()

        merged = pred_df.merge(hist_df, on="Date", how="left").dropna(
            subset=["Pred_Close", "Actual_Close"]
        )
        if merged.empty:
            skipped += 1
            continue

        try:
            m = calculate_regression_metrics(
                merged["Actual_Close"].values, merged["Pred_Close"].values
            )
        except Exception:
            skipped += 1
            continue

        records.append(
            {
                "ticker": ticker,
                "n_points": int(len(merged)),
                "mse": m["mse"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "mape": m["mape"],
                "smape": m["smape"],
            }
        )

        history_window = (
            hist_df[hist_df["Date"] < fcast_df["Date"].min()].tail(60).copy()
        )
        actual_window = hist_df[
            (hist_df["Date"] >= fcast_df["Date"].min())
            & (hist_df["Date"] <= fcast_df["Date"].max())
        ].copy()
        plot_payload[ticker] = {
            "history_window": history_window,
            "pred_window": pred_df.copy(),
            "actual_window": actual_window,
            "forecast_start": fcast_df["Date"].min(),
        }

    if not records:
        raise RuntimeError(
            "No ticker metrics computed. Forecast dates likely do not overlap with available actuals. "
            "Re-run forecasting with --backtest_horizon equal to forecast horizon (e.g., 5)."
        )

    per_ticker = pd.DataFrame(records).sort_values("rmse", ascending=True)
    aggregate = {
        "num_tickers": int(len(per_ticker)),
        "mean_mse": float(per_ticker["mse"].mean()),
        "mean_rmse": float(per_ticker["rmse"].mean()),
        "mean_mae": float(per_ticker["mae"].mean()),
        "mean_mape": float(per_ticker["mape"].mean()),
        "mean_smape": float(per_ticker["smape"].mean()),
        "median_rmse": float(per_ticker["rmse"].median()),
        "median_mae": float(per_ticker["mae"].median()),
        "median_mape": float(per_ticker["mape"].median()),
        "median_smape": float(per_ticker["smape"].median()),
        "skipped_tickers": int(skipped),
    }
    aggregate_df = pd.DataFrame([aggregate])

    per_ticker_path = output_dir / "per_ticker_metrics.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    per_ticker.to_csv(per_ticker_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    print(f"Saved: {per_ticker_path}")
    print(f"Saved: {aggregate_path}")
    print(
        f"Tickers evaluated: {aggregate['num_tickers']} | skipped: {aggregate['skipped_tickers']}"
    )
    print("\nPer-ticker metrics:")
    for _, row in per_ticker.iterrows():
        print(
            f"{row['ticker']}: "
            f"MSE={row['mse']:.6f} RMSE={row['rmse']:.6f} MAE={row['mae']:.6f} "
            f"MAPE={row['mape']:.4f}% SMAPE={row['smape']:.4f}%"
        )

    print("\nAggregate (mean) metrics:")
    print_metrics(
        {
            "mse": aggregate["mean_mse"],
            "rmse": aggregate["mean_rmse"],
            "mae": aggregate["mean_mae"],
            "mape": aggregate["mean_mape"],
            "smape": aggregate["mean_smape"],
        },
        prefix="Mean",
    )

    top10 = per_ticker.head(10).copy()
    fig, axes = plt.subplots(5, 2, figsize=(18, 22), squeeze=False)
    axes_flat = axes.flatten()

    plotted = 0
    for _, row in top10.iterrows():
        ticker = row["ticker"]
        payload = plot_payload.get(ticker)
        if payload is None:
            continue

        ax = axes_flat[plotted]
        plotted += 1

        history_window = payload["history_window"]
        pred_window = payload["pred_window"]
        actual_window = payload["actual_window"]
        forecast_start = payload["forecast_start"]

        ax.plot(
            history_window["Date"],
            history_window["Actual_Close"],
            color="#1f4e79",
            linewidth=2.0,
            label="History (60d)",
        )
        ax.plot(
            pred_window["Date"],
            pred_window["Pred_Close"],
            color="#c33c1e",
            linewidth=2.0,
            linestyle="--",
            label="Forecast (P50)",
        )
        if "Low_Close" in pred_window.columns and "High_Close" in pred_window.columns:
            ax.fill_between(
                pred_window["Date"],
                pred_window["Low_Close"],
                pred_window["High_Close"],
                color="#e07a5f",
                alpha=0.25,
                label="Forecast CI",
            )

        ax.plot(
            actual_window["Date"],
            actual_window["Actual_Close"],
            color="#2a9d8f",
            linewidth=1.8,
            marker="o",
            markersize=3,
            label="Actual",
        )
        ax.axvline(forecast_start, color="#666666", linewidth=1.0, alpha=0.7)
        ax.set_title(f"{ticker.upper()} | RMSE={row['rmse']:.4f}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Close")
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(loc="best", fontsize=8)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )

    for i in range(plotted, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.tight_layout()
    plot_path = output_dir / "top10_forecast_plot.png"
    plt.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
