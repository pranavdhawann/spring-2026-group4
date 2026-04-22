"""
predict_gnn.py — Inference pipeline: load checkpoint → process data → build graph → forward → MC Dropout → JSON output.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils_gnn import (
    build_trading_calendar,
    DEFAULT_BEST_CHECKPOINT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_RESULTS_DIR,
    get_device,
    load_checkpoint,
    load_config,
    log_runtime_context,
    resolve_data_path,
    set_seed,
    setup_logging,
)
from src.dataset_gnn import (
    FeatureBuilder,
    build_master_calendar,
    load_fundamentals,
    load_news_embeddings,
    load_sectors,
    load_time_series,
)
from src.graph_gnn import SectorEdgeBuilder
from src.model_gnn import TemporalGNN
logger = logging.getLogger(__name__)


def predict_single_date(
    config: dict,
    model: TemporalGNN,
    prediction_date: str,
    ticker_dfs: dict,
    sectors_df: pd.DataFrame,
    feature_builder: FeatureBuilder,
    news_cache: dict,
    fundamentals: dict,
    device: torch.device,
    mc_samples: int = 10,
) -> List[dict]:
    """
    Generate predictions for all active stocks on a given date.
    
    Args:
        config: Configuration dict
        model: Trained TemporalGNN
        prediction_date: Date string (YYYY-MM-DD)
        ticker_dfs: ticker → OHLCV DataFrame
        sectors_df: Sector mapping
        feature_builder: Pre-configured FeatureBuilder
        news_cache: Pre-loaded news embeddings
        device: Compute device
        mc_samples: Number of MC Dropout forward passes
    
    Returns:
        List of prediction dicts (one per stock)
    """
    pred_date = pd.Timestamp(prediction_date)
    data_cfg = config["data"]
    window_size = data_cfg.get("window_size", 60)
    min_history = data_cfg.get("min_history", 252)
    horizon = data_cfg.get("horizon", 5)
    max_nodes = 550

    master_calendar = build_master_calendar(ticker_dfs)
    logger.info(
        "Prediction request | requested_date=%s | resolved_calendar_days=%d | window=%d | horizon=%d | mc_samples=%d",
        prediction_date,
        len(master_calendar),
        window_size,
        horizon,
        mc_samples,
    )

    if pred_date not in master_calendar:
        from src.utils_gnn import get_prev_trading_day
        pred_date = get_prev_trading_day(pred_date, master_calendar)
        logger.info("Adjusted prediction date to trading day: %s", pred_date.date())

    pred_idx = master_calendar.get_loc(pred_date)
    if pred_idx < window_size:
        raise ValueError(
            f"Not enough history for prediction on {pred_date.date()}: need {window_size} days, have {pred_idx}"
        )

    history_dates = master_calendar[master_calendar <= pred_date]
    feature_builder.compute_global_stats(ticker_dfs, history_dates)

    active_tickers = []
    for ticker, df in ticker_dfs.items():
        dates = pd.to_datetime(df["date"])
        if (dates <= pred_date).sum() >= min_history:
            active_tickers.append(ticker)

    active_tickers = sorted(active_tickers)[:max_nodes]
    N = len(active_tickers)
    logger.info("Active tickers at %s: %d", pred_date.date(), N)
    if N == 0:
        return []

    ticker_to_local = {ticker: i for i, ticker in enumerate(active_tickers)}
    window_start = pred_idx - window_size
    window_dates = [str(master_calendar[t].date()) for t in range(window_start, pred_idx)]

    ts_features = np.zeros((N, window_size, FeatureBuilder.NUM_FEATURES), dtype=np.float32)
    last_close_list = []
    news_per_stock = []
    report_fundamentals = []

    for i, ticker in enumerate(active_tickers):
        features, _ = feature_builder.build_features(ticker_dfs[ticker], master_calendar)
        if len(features) > 0:
            ts_features[i] = features.iloc[window_start:pred_idx].values.astype(np.float32)

        close_series = ticker_dfs[ticker].set_index("date")["close"].reindex(master_calendar).ffill()
        last_close_list.append(close_series.iloc[pred_idx])

        ticker_news = news_cache.get(ticker, {})
        news_per_stock.append([ticker_news.get(date_str) for date_str in window_dates])
        report_fundamentals.append(fundamentals.get(ticker))

    sector_builder = SectorEdgeBuilder(sectors_df)
    sector_edge_index, sector_edge_attr = sector_builder.build(active_tickers, ticker_to_local)

    # FIX E3: build correlation edges for prediction too
    from src.graph_gnn import CorrelationEdgeBuilder
    graph_cfg = config.get("graph", {})
    corr_builder = CorrelationEdgeBuilder(
        top_k=graph_cfg.get("correlation_top_k", 10),
        window=graph_cfg.get("correlation_window", 60),
    )
    # Build returns_dict for correlation edges
    returns_dict = {}
    corr_window = graph_cfg.get("correlation_window", 60)
    for ticker in active_tickers:
        df = ticker_dfs[ticker]
        close_s = df.set_index("date")["close"].reindex(master_calendar).ffill()
        from src.utils_gnn import compute_log_returns
        lr = compute_log_returns(close_s).values
        r = lr[max(0, pred_idx - corr_window):pred_idx]
        returns_dict[ticker] = np.nan_to_num(r, nan=0.0)
    corr_edge_index, corr_edge_attr = corr_builder.build(
        pred_date, returns_dict, active_tickers, ticker_to_local,
    )

    sample = {
        "ts_features": torch.tensor(ts_features, dtype=torch.float),
        "news_per_stock": news_per_stock,
        "report_fundamentals": report_fundamentals,
        "sector_edge_index": sector_edge_index,
        "sector_edge_attr": sector_edge_attr,
        "corr_edge_index": corr_edge_index,
        "corr_edge_attr": corr_edge_attr,
        "returns_dict": returns_dict,
        "targets": torch.zeros(N, horizon, dtype=torch.float),
        "target_close": torch.zeros(N, horizon, dtype=torch.float),
        "last_close": torch.tensor(last_close_list, dtype=torch.float),
        "target_idx": torch.arange(N, dtype=torch.long),
        "target_tickers": active_tickers,
        "active_tickers": active_tickers,
        "pred_date": str(pred_date.date()),
        "pred_date_ts": pred_date,
        "num_active": N,
    }

    pred_bundle = model.predict_with_uncertainty(sample, device=device, n_mc=mc_samples)
    mean_lr = pred_bundle["log_returns_mean"].cpu().numpy()
    std_lr = pred_bundle["log_returns_std"].cpu().numpy()
    price_mean = pred_bundle["price_mean"].cpu().numpy()
    price_lower = pred_bundle["price_lower"].cpu().numpy()
    price_upper = pred_bundle["price_upper"].cpu().numpy()
    last_close_np = sample["last_close"].cpu().numpy()

    trading_cal = build_trading_calendar()
    future_dates = trading_cal[trading_cal > pred_date][:horizon]

    predictions = []
    sector_map = {}
    if not sectors_df.empty and "ticker" in sectors_df.columns:
        sector_map = sectors_df.set_index("ticker")["sector"].to_dict()

    for i, ticker in enumerate(active_tickers):
        forecast = []
        for k in range(horizon):
            forecast.append(
                {
                    "day": k + 1,
                    "date": str(future_dates[k].date()) if k < len(future_dates) else f"T+{k + 1}",
                    "log_return": round(float(mean_lr[i, k]), 6),
                    "predicted_close": round(float(price_mean[i, k]), 2),
                }
            )

        predictions.append(
            {
                "ticker": ticker,
                "prediction_date": str(pred_date.date()),
                "last_known_close": round(float(last_close_np[i]), 2),
                "forecast": forecast,
                "sector": sector_map.get(ticker, "Unknown"),
                "confidence_interval_95": {
                    "method": "mc_dropout",
                    "n_samples": mc_samples,
                    "close_lower": [round(float(v), 2) for v in price_lower[i]],
                    "close_upper": [round(float(v), 2) for v in price_upper[i]],
                    "log_return_std": [round(float(v), 6) for v in std_lr[i]],
                },
            }
        )

    return predictions


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Predict with Temporal GNN")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_BEST_CHECKPOINT)
    parser.add_argument("--date", type=str, required=True, help="Prediction date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--mc-samples", type=int, default=10, help="MC Dropout samples")
    parser.add_argument("--ticker", type=str, default=None, help="Single ticker to predict (optional)")
    args = parser.parse_args()
    
    config = load_config(args.config)
    log_path = setup_logging(config, command_name="predict", config_path=args.config, args=args)
    logger.info("Loaded config from %s", os.path.abspath(args.config))
    log_runtime_context("predict", config, extra={"prediction_log_path": log_path})
    set_seed(config.get("seed", 42))
    device = get_device()
    
    data_cfg = config["data"]
    data_dir = data_cfg["data_dir"]
    
    # Load data
    ts_subdir = data_cfg.get("time_series_dir", "sp500_time_series")
    ticker_dfs = load_time_series(data_dir, ts_subdir)
    sectors_df = load_sectors(data_dir, data_cfg.get("description_file", "sp500stock_data_description.csv"))
    fundamentals = load_fundamentals(data_dir, data_cfg.get("table_dir", "sp500_table"), list(ticker_dfs.keys()))
    logger.info("Prediction data loaded | tickers=%d | sectors=%d", len(ticker_dfs), len(sectors_df))
    
    # Feature builder
    feature_builder = FeatureBuilder(
        clip_sigma=data_cfg.get("log_return_clip_sigma", 5.0),
        zscore_min_days=data_cfg.get("zscore_min_days", 60),
    )
    logger.info("Initialized feature builder for prediction")
    
    # News cache
    _, news_cache_dir = resolve_data_path(
        data_dir,
        data_cfg.get("news_cache_dir"),
        os.path.join("cache", "news_embeddings"),
        kind="directory",
    )
    news_cache = load_news_embeddings(news_cache_dir, list(ticker_dfs.keys()))
    
    # Load model
    max_nodes = 550
    model = TemporalGNN(config, max_nodes=max_nodes).to(device)
    load_checkpoint(args.checkpoint, model)
    logger.info("Loaded prediction checkpoint from %s", os.path.abspath(args.checkpoint))
    
    # Predict
    predictions = predict_single_date(
        config, model, args.date, ticker_dfs, sectors_df,
        feature_builder, news_cache, fundamentals, device,
        mc_samples=args.mc_samples,
    )
    
    # Filter by ticker if specified
    if args.ticker:
        predictions = [p for p in predictions if p["ticker"] == args.ticker.upper()]
        logger.info("Filtered predictions to ticker=%s | remaining=%d", args.ticker.upper(), len(predictions))
    
    # Output
    output_path = args.output or os.path.join(DEFAULT_RESULTS_DIR, f"predictions_{args.date}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)
    
    logger.info(f"Saved {len(predictions)} predictions to {output_path}")
    
    # Print sample
    if predictions:
        sample = predictions[0]
        logger.info(f"\nSample prediction for {sample['ticker']}:")
        logger.info(json.dumps(sample, indent=2))
    else:
        logger.warning("No predictions were produced for the given request")


if __name__ == "__main__":
    main()
