"""
Batch zero-shot forecasting script using Chronos.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from chronos import ChronosPipeline

from src.preProcessing.data_preprocessing_chronos_t5 import load_ticker
from src.utils.utils import read_json_file, read_yaml, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Batch Zero-Shot Forecasting with Chronos"
    )
    parser.add_argument(
        "--manifest_path", type=str, required=True, help="Path to JSON manifest"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing ticker CSVs"
    )
    parser.add_argument(
        "--target_col", type=str, default="Close", help="Target column to extract"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Execution device. Use 'cuda' to force GPU.",
    )
    parser.add_argument(
        "--forecast_horizon", type=int, default=7, help="Number of steps to forecast"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of predictive samples to draw",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/baseline/chronos/forecasts/batch/",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of tickers to process before clearing GPU cache",
    )
    parser.add_argument(
        "--backtest_horizon",
        type=int,
        default=0,
        help="If > 0, hold out the last N points and forecast them for evaluable backtesting.",
    )
    parser.add_argument(
        "--config_path", type=str, default=None, help="Path to config YAML"
    )
    args = parser.parse_args()
    target_is_returns = args.target_col.lower() == "returns"

    set_seed(42)

    config = None
    if args.config_path:
        config = read_yaml(args.config_path)
        logger.info(f"Loaded config from {args.config_path}")

    output_dir = Path(args.output_dir)
    tickers_dir = output_dir / "tickers"
    tickers_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json_file(args.manifest_path)
    if not manifest:
        logger.error(
            f"Failed to read manifest or manifest is empty: {args.manifest_path}"
        )
        sys.exit(1)

    if args.device == "auto":
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            logger.error(
                "CUDA requested but not available. Check driver/PyTorch CUDA install."
            )
            sys.exit(1)
        device_map = "cuda"
    else:
        device_map = "cpu"

    logger.info(f"Using device: {device_map}")

    base_model_id = "amazon/chronos-t5-large"
    if config and "model_id" in config:
        base_model_id = config["model_id"]

    logger.info(f"Loading ChronosPipeline (zero-shot): {base_model_id}")
    try:
        pipeline = ChronosPipeline.from_pretrained(
            base_model_id,
            device_map=device_map,
            dtype=torch.bfloat16 if device_map == "cuda" else torch.float32,
        )
    except Exception as e:
        logger.error(f"Failed to load Chronos model: {e}")
        sys.exit(1)

    summary_records = []
    backtest_mode = args.backtest_horizon > 0
    if backtest_mode and args.backtest_horizon != args.forecast_horizon:
        logger.warning(
            f"backtest_horizon ({args.backtest_horizon}) != forecast_horizon ({args.forecast_horizon}). "
            "Using backtest_horizon as prediction length for date alignment."
        )

    for i, entry in enumerate(tqdm(manifest, desc="Forecasting")):
        ticker = entry.get("ticker", "UNKNOWN")
        file_path = entry.get("file_path")

        csv_path = None
        if file_path and Path(file_path).exists():
            csv_path = Path(file_path)
        else:
            fallback = Path(args.data_dir) / f"{ticker.lower()}.csv"
            if fallback.exists():
                csv_path = fallback

        if not csv_path:
            logger.warning(f"[{ticker}] CSV not found. Skipping.")
            continue

        try:
            dates, values = load_ticker(str(csv_path), target_col=args.target_col)
            if len(values) == 0:
                logger.warning(f"[{ticker}] Time series is empty. Skipping.")
                continue

            close_dates, close_values = load_ticker(str(csv_path), target_col="Close")
            if len(close_values) == 0:
                logger.warning(f"[{ticker}] Close series is empty. Skipping.")
                continue

            prediction_length = args.forecast_horizon
            actual_future_close = None

            if backtest_mode:
                h = int(args.backtest_horizon)
                if len(values) <= h or len(close_values) <= h:
                    logger.warning(
                        f"[{ticker}] Series too short for backtest_horizon={h}. Skipping."
                    )
                    continue
                prediction_length = h
                context_values = values[:-h]
                last_known_date = close_dates[-h - 1]
                last_known_close = float(close_values[-h - 1])
                future_dates = pd.to_datetime(close_dates[-h:])
                actual_future_close = close_values[-h:]
            else:
                context_values = values
                last_known_date = close_dates[-1]
                last_known_close = float(close_values[-1])
                start_fcast = pd.to_datetime(last_known_date) + pd.Timedelta(days=1)
                future_dates = pd.date_range(
                    start=start_fcast, periods=prediction_length, freq="B"
                ).values
                if len(future_dates) > prediction_length:
                    future_dates = future_dates[:prediction_length]
                elif len(future_dates) < prediction_length:
                    future_dates = pd.date_range(
                        start=start_fcast, periods=prediction_length, freq="D"
                    ).values

            context = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0)

            forecast = pipeline.predict(
                context,
                prediction_length,
                num_samples=args.num_samples,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
            )
            forecast_samples = forecast[0].float().cpu().numpy()

            point_forecast = np.median(forecast_samples, axis=0)
            q10 = np.percentile(forecast_samples, 10, axis=0)
            q90 = np.percentile(forecast_samples, 90, axis=0)

            ticker_df = pd.DataFrame(
                {
                    "Date": future_dates,
                    "Point_Forecast": point_forecast,
                    "Quantile_10": q10,
                    "Quantile_90": q90,
                }
            )

            point_for_summary = point_forecast
            q10_for_summary = q10
            q90_for_summary = q90

            if target_is_returns:
                # returns are log-returns, so convert back to close path via cumulative exp
                point_close = last_known_close * np.exp(np.cumsum(point_forecast))
                q10_close = last_known_close * np.exp(np.cumsum(q10))
                q90_close = last_known_close * np.exp(np.cumsum(q90))
                ticker_df["Point_Forecast_Close"] = point_close
                ticker_df["Quantile_10_Close"] = q10_close
                ticker_df["Quantile_90_Close"] = q90_close
                point_for_summary = point_close
                q10_for_summary = q10_close
                q90_for_summary = q90_close

            if actual_future_close is not None:
                ticker_df["Actual_Close"] = actual_future_close

            ticker_df.to_csv(tickers_dir / f"{ticker}_forecast.csv", index=False)

            pred_end_value = point_for_summary[-1]
            direction = "up" if pred_end_value > last_known_close else "down"
            pct_change = 0.0
            if last_known_close != 0:
                pct_change = (
                    (pred_end_value - last_known_close) / last_known_close
                ) * 100

            summary_records.append(
                {
                    "ticker": ticker,
                    "forecast_target_col": args.target_col,
                    "backtest_horizon": int(args.backtest_horizon),
                    "last_known_date": str(pd.to_datetime(last_known_date).date()),
                    "last_known_close": last_known_close,
                    "7_day_point_forecast_values": list(point_forecast),
                    "7_day_point_forecast_close_values": list(point_for_summary),
                    "7_day_q10_close_values": list(q10_for_summary),
                    "7_day_q90_close_values": list(q90_for_summary),
                    "forecast_direction": direction,
                    "predicted_percent_change": pct_change,
                }
            )
        except Exception as e:
            logger.warning(f"[{ticker}] Failed to process: {e}")
            continue

        if (i + 1) % args.batch_size == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(
            f"Saved generated summary for {len(summary_records)} tickers to {summary_path}"
        )
    else:
        logger.warning("No forecasts generated; summary CSV not created.")


if __name__ == "__main__":
    main()
