"""
forecast_plot_gnn.py — Generate forecast accuracy plots for stocks in the dataset.

For each stock:
    - Plot the last 60 trading days of actual close prices (history)
    - Plot the 5-day forecast from the model
    - Plot the 5 actual close prices for those forecast days
    - Save each plot to tgnn/results/accuracy/{TICKER}.png

Usage:
    python predict/forecast_plot_gnn.py --config config/config_gnn.yaml --checkpoint tgnn/checkpoints/best.pt
    python predict/forecast_plot_gnn.py --config config/config_gnn.yaml --checkpoint tgnn/checkpoints/best.pt --tickers AAPL,MSFT,NVDA
    python predict/forecast_plot_gnn.py --config config/config_gnn.yaml --checkpoint tgnn/checkpoints/best.pt --max-stocks 50
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_forecast_plots(config, checkpoint_path, tickers=None, max_stocks=50, output_dir=os.path.join("tgnn", "results", "accuracy")):
    """
    Generate forecast vs actual plots for stocks.

    Steps:
        1. Load data and model
        2. For each stock in the test set, get the LAST prediction date
        3. Extract 60-day history, 5-day forecast, 5-day actuals
        4. Plot and save
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    from src.dataset_gnn import FeatureBuilder, TemporalGraphDataset, load_fundamentals, load_news_embeddings, load_sectors, load_time_series
    from src.loss_gnn import reconstruct_prices
    from src.model_gnn import TemporalGNN
    from src.utils_gnn import build_master_calendar, get_device, load_checkpoint, set_seed, temporal_train_val_test_split

    set_seed(config.get("seed", 42))
    device = get_device()
    data_cfg = config["data"]
    data_dir = data_cfg["data_dir"]
    logger.info(
        "Forecast plot request | checkpoint=%s | output_dir=%s | max_stocks=%d | explicit_tickers=%s",
        os.path.abspath(checkpoint_path),
        os.path.abspath(output_dir),
        max_stocks,
        ",".join(tickers) if tickers else "ALL",
    )

    # ── Load data ──
    logger.info("Loading data...")
    ticker_dfs = load_time_series(data_dir, data_cfg.get("time_series_dir", "sp500_time_series"))

    # FIX P2: Apply the same max_tickers filtering as build_dataloaders so
    # forecast plots use the same ticker set the model was trained on.
    max_tickers = data_cfg.get("max_tickers")
    if max_tickers is not None and max_tickers < len(ticker_dfs):
        sorted_tickers = sorted(ticker_dfs, key=lambda t: len(ticker_dfs[t]), reverse=True)
        ticker_dfs = {t: ticker_dfs[t] for t in sorted_tickers[:max_tickers]}
        logger.info("Filtered to %d tickers (max_tickers=%d)", len(ticker_dfs), max_tickers)

    sectors_df = load_sectors(data_dir, data_cfg.get("description_file", "sp500stock_data_description.csv"))
    fundamentals = load_fundamentals(data_dir, data_cfg.get("table_dir", "sp500_table"), list(ticker_dfs.keys()))
    master_calendar = build_master_calendar(ticker_dfs)
    news_cache = load_news_embeddings(os.path.join(data_dir, "cache", "news_embeddings"), list(ticker_dfs.keys()))

    # Split to get test dates
    split_cfg = config["split"]
    _, _, test_dates = temporal_train_val_test_split(
        master_calendar, split_cfg.get("val_days", 45),
        split_cfg.get("test_days", 45), split_cfg.get("purge_days", 5))

    feature_builder = FeatureBuilder(data_cfg.get("log_return_clip_sigma", 5.0), data_cfg.get("zscore_min_days", 60))
    feature_builder.compute_global_stats(ticker_dfs, master_calendar[:len(master_calendar) - 100])

    # ── Load model ──
    logger.info("Loading model...")
    model = TemporalGNN(config, max_nodes=550).to(device)
    if fundamentals:
        model.reports_encoder.compute_normalization_stats(fundamentals)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    # ── Build test dataset ──
    test_ds = TemporalGraphDataset(
        config=config, dates=test_dates, ticker_dfs=ticker_dfs, sectors_df=sectors_df,
        feature_builder=feature_builder, news_cache=news_cache, fundamentals=fundamentals,
        master_calendar=master_calendar, max_nodes=550, mode="test")

    # ── Determine which tickers to plot ──
    if tickers:
        plot_tickers = set(t.upper() for t in tickers)
    else:
        # Use all tickers that appear in test set
        plot_tickers = set(test_ds.ticker_features.keys())

    # Collect forecasts per ticker from the LAST test date they appear in
    logger.info(f"Running inference on {len(test_ds)} test samples...")
    ticker_forecasts = {}  # ticker → {pred_date, pred_close, actual_close, last_close}

    with torch.no_grad():
        for idx in range(len(test_ds)):
            sample = test_ds[idx]
            if sample["num_active"] == 0:
                continue

            output = model(sample, device=device)
            pred_lr = output["log_returns"].cpu().numpy()  # (N_T, 5)
            last_close = sample["last_close"].numpy()       # (N_T,)
            target_close = sample["target_close"].numpy()   # (N_T, 5)
            target_tickers = sample["target_tickers"]
            pred_date = sample["pred_date"]

            # Reconstruct predicted close prices
            cum_lr = np.cumsum(pred_lr, axis=1)
            pred_close = last_close[:, None] * np.exp(cum_lr)  # (N_T, 5)

            for i, ticker in enumerate(target_tickers):
                if ticker in plot_tickers:
                    ticker_forecasts[ticker] = {
                        "pred_date": pred_date,
                        "pred_close": pred_close[i],      # (5,)
                        "actual_close": target_close[i],   # (5,)
                        "last_close": last_close[i],
                    }

    # ── Generate plots ──
    os.makedirs(output_dir, exist_ok=True)
    # FIX P1: master_calendar timestamps have a UTC time component (e.g. 04:00)
    # from pd.to_datetime(utc=True).dt.tz_localize(None) in load_time_series.
    # But pred_date_str is date-only ("2025-03-21") → pd.Timestamp gives midnight.
    # Normalize both sides to date-only to avoid the mismatch that caused all
    # plots to be silently skipped.
    date_to_idx = {d.normalize(): i for i, d in enumerate(master_calendar)}

    tickers_to_plot = sorted(ticker_forecasts.keys())[:max_stocks]
    logger.info(f"Generating {len(tickers_to_plot)} forecast plots...")

    for ticker in tickers_to_plot:
        fc = ticker_forecasts[ticker]
        pred_date_str = fc["pred_date"]

        # Get raw close prices from the ticker DataFrame
        df = ticker_dfs.get(ticker)
        if df is None:
            continue

        df_close = df.set_index("date")["close"].sort_index()
        pred_date = pd.Timestamp(pred_date_str).normalize()

        if pred_date not in date_to_idx:
            continue
        pred_idx = date_to_idx[pred_date]

        # 60-day history
        history_days = 60
        hist_start = max(0, pred_idx - history_days)
        hist_dates = master_calendar[hist_start:pred_idx + 1]
        hist_prices = []
        for d in hist_dates:
            if d in df_close.index:
                hist_prices.append(df_close.loc[d])
            else:
                hist_prices.append(np.nan)
        # Normalize index to date-only for consistent plotting with pred_date
        hist_dates_norm = hist_dates.normalize()
        hist_prices = pd.Series(hist_prices, index=hist_dates_norm).ffill().dropna()

        if len(hist_prices) < 10:
            continue

        # 5-day forecast dates and prices
        horizon = 5
        forecast_dates = []
        for k in range(1, horizon + 1):
            fi = pred_idx + k
            if fi < len(master_calendar):
                forecast_dates.append(master_calendar[fi].normalize())

        if len(forecast_dates) != horizon:
            continue

        pred_close = fc["pred_close"][:len(forecast_dates)]
        actual_close = fc["actual_close"][:len(forecast_dates)]

        # ── Plot ──
        fig, ax = plt.subplots(figsize=(14, 6))

        # History
        ax.plot(hist_prices.index, hist_prices.values,
                color="#2196F3", linewidth=1.5, label="Historical Close", zorder=2)

        # Connecting line from last history point to first forecast point
        connect_dates = [hist_prices.index[-1], forecast_dates[0]]
        connect_actual = [hist_prices.values[-1], actual_close[0]]
        connect_pred = [hist_prices.values[-1], pred_close[0]]

        # Forecast
        ax.plot(forecast_dates, pred_close,
                color="#FF5722", linewidth=2.5, marker="o", markersize=7,
                label="Forecast (5-day)", zorder=4)
        ax.plot(connect_dates, connect_pred, color="#FF5722", linewidth=1.5,
                linestyle="--", alpha=0.5, zorder=3)

        # Actual future
        ax.plot(forecast_dates, actual_close,
                color="#4CAF50", linewidth=2.5, marker="s", markersize=7,
                label="Actual (5-day)", zorder=4)
        ax.plot(connect_dates, connect_actual, color="#4CAF50", linewidth=1.5,
                linestyle="--", alpha=0.5, zorder=3)

        # Vertical line at prediction boundary
        ax.axvline(x=pred_date, color="gray", linestyle=":", alpha=0.7, label="Prediction Date")

        # Shade forecast region
        ax.axvspan(forecast_dates[0], forecast_dates[-1], alpha=0.06, color="orange")

        # Compute error for annotation
        mae = np.mean(np.abs(pred_close - actual_close))
        mape = np.mean(np.abs((pred_close - actual_close) / (actual_close + 1e-8))) * 100

        ax.set_title(f"{ticker} — 60-Day History + 5-Day Forecast\n"
                     f"MAE: ${mae:.2f}  |  MAPE: {mape:.2f}%  |  Pred Date: {pred_date_str}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Close Price ($)", fontsize=11)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{ticker}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved forecast plot for %s to %s", ticker, save_path)

    logger.info(f"Saved {len(tickers_to_plot)} plots to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Generate forecast accuracy plots")
    parser.add_argument("--config", default=os.path.join("config", "config_gnn.yaml"))
    parser.add_argument("--checkpoint", default=os.path.join("tgnn", "checkpoints", "best.pt"))
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers (e.g. AAPL,MSFT). Default: all in test set")
    parser.add_argument("--max-stocks", type=int, default=50, help="Max number of plots to generate")
    parser.add_argument("--output-dir", default=os.path.join("tgnn", "results", "accuracy"))
    args = parser.parse_args()

    from src.utils_gnn import load_config, log_runtime_context, setup_logging
    config = load_config(args.config)
    log_path = setup_logging(config, command_name="forecast_plot", config_path=args.config, args=args)
    logger.info("Loaded config from %s", os.path.abspath(args.config))
    log_runtime_context("forecast_plot", config, extra={"forecast_plot_log_path": log_path})

    tickers = args.tickers.split(",") if args.tickers else None
    generate_forecast_plots(config, args.checkpoint, tickers=tickers,
                            max_stocks=args.max_stocks, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
