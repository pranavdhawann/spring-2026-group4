"""
Batch forecasting script using Chronos.

This script iterates through a manifest of tickers, loads their history, 
generates 7-day point and quantile forecasts using a Chronos pipeline 
(either base or fine-tuned), saves detailed per-ticker CSVs, and aggregates 
predictions into a single summary CSV. Includes ETA logging and GPU cache management.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Ensure project root is importable regardless of cwd
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from chronos import ChronosPipeline
from src.preProcessing.data_preprocessing_chronos import load_ticker
from src.utils.utils import read_json_file, read_yaml, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch Forecasting with Chronos")
    parser.add_argument("--model_mode", type=str, choices=["base", "finetuned"], required=True, help="Model mode (base or finetuned)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter if model_mode=finetuned")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to JSON manifest")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing ticker CSVs")
    parser.add_argument("--target_col", type=str, default="Close", help="Target column to extract")
    parser.add_argument("--forecast_horizon", type=int, default=7, help="Number of steps to forecast")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of predictive samples to draw")
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
    parser.add_argument("--config_path", type=str, default=None, help="Path to config YAML")

    args = parser.parse_args()

    # Set random seed
    set_seed(42)

    # Optionally load config
    config = None
    if args.config_path:
        config = read_yaml(args.config_path)
        logger.info(f"Loaded config from {args.config_path}")

    output_dir = Path(args.output_dir)
    tickers_dir = output_dir / "tickers"
    tickers_dir.mkdir(parents=True, exist_ok=True)

    # Read manifest
    manifest = read_json_file(args.manifest_path)
    if not manifest:
        logger.error(f"Failed to read manifest or manifest is empty: {args.manifest_path}")
        sys.exit(1)

    # Initialize device
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device_map}")

    # Initialize pipeline
    base_model_id = "amazon/chronos-t5-base"
    if config and "model_id" in config:
        base_model_id = config["model_id"]

    logger.info(f"Loading ChronosPipeline (mode: {args.model_mode})")
    try:
        if args.model_mode == "base":
            pipeline = ChronosPipeline.from_pretrained(
                base_model_id,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )
        elif args.model_mode == "finetuned":
            if not args.adapter_path:
                logger.error("--adapter_path is required when model_mode='finetuned'")
                sys.exit(1)
            
            logger.info(f"Applying adapter from {args.adapter_path}")
            pipeline = ChronosPipeline.from_pretrained(
                base_model_id,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
            )
            
            # Apply adapter using PEFT if pipeline exposes HF model or standard methods
            if hasattr(pipeline.model, "load_adapter"):
                pipeline.model.load_adapter(args.adapter_path)
            else:
                try:
                    from peft import PeftModel
                    pipeline.model = PeftModel.from_pretrained(pipeline.model, args.adapter_path)
                except ImportError:
                    logger.warning("peft not installed but model_mode=finetuned selected; prediction might fail if adapter needs peft.")

    except Exception as e:
        logger.error(f"Failed to load Chronos model: {e}")
        sys.exit(1)

    summary_records = []

    # Process tickers with ETA logging
    for i, entry in enumerate(tqdm(manifest, desc="Forecasting")):
        ticker = entry.get("ticker", "UNKNOWN")
        file_path = entry.get("file_path")

        # Resolve CSV path
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

            last_known_date = dates[-1]
            last_known_close = float(values[-1])

            # Prepare context for Chronos
            context = torch.tensor(values, dtype=torch.float32).unsqueeze(0)  # Shape: (1, seq_len)
            
            # Generate predictions
            forecast = pipeline.predict(
                context,
                args.forecast_horizon,
                num_samples=args.num_samples,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
            )
            # forecast shape: (batch_size=1, num_samples, prediction_length)
            forecast_samples = forecast[0].float().cpu().numpy()

            # Point forecast (median) and quantiles (10th, 90th array)
            point_forecast = np.median(forecast_samples, axis=0)
            q10 = np.percentile(forecast_samples, 10, axis=0)
            q90 = np.percentile(forecast_samples, 90, axis=0)

            # Build forecast future dates (business days)
            start_fcast = pd.to_datetime(last_known_date) + pd.Timedelta(days=1)
            future_dates = pd.date_range(start=start_fcast, periods=args.forecast_horizon, freq="B").values

            # Verify lengths match, truncate if pd.date_range somehow over-returns
            if len(future_dates) > args.forecast_horizon:
                future_dates = future_dates[:args.forecast_horizon]
            elif len(future_dates) < args.forecast_horizon:
                # Fallback to general days if B-day generation fails or under-returns
                future_dates = pd.date_range(start=start_fcast, periods=args.forecast_horizon, freq="D").values
            
            # Save individual ticker CSV
            ticker_df = pd.DataFrame({
                "Date": future_dates,
                "Point_Forecast": point_forecast,
                "Quantile_10": q10,
                "Quantile_90": q90,
            })
            ticker_df.to_csv(tickers_dir / f"{ticker}_forecast.csv", index=False)

            # Compute summary stats
            pred_end_value = point_forecast[-1]
            direction = "up" if pred_end_value > last_known_close else "down"

            # Check logic: percent change over forecast horizon relative to last close
            pct_change = 0.0
            if last_known_close != 0:
                pct_change = ((pred_end_value - last_known_close) / last_known_close) * 100

            summary_records.append({
                "ticker": ticker,
                "last_known_date": str(pd.to_datetime(last_known_date).date()),
                "last_known_close": last_known_close,
                "7_day_point_forecast_values": list(point_forecast),
                "forecast_direction": direction,
                "predicted_percent_change": pct_change,
            })

        except Exception as e:
            logger.warning(f"[{ticker}] Failed to process: {e}")
            continue

        # Clear CUDA cache every batch_size
        if (i + 1) % args.batch_size == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    # Save summary CSV
    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Saved generated summary for {len(summary_records)} tickers to {summary_path}")
    else:
        logger.warning("No forecasts generated; summary CSV not created.")

if __name__ == "__main__":
    main()
