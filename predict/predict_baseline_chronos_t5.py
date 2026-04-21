import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from chronos import ChronosPipeline
except ImportError:
    ChronosPipeline = None

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preProcessing.data_preprocessing_chronos_t5 import load_ticker
from src.utils.utils import read_yaml, set_seed


def get_args():
    parser = argparse.ArgumentParser(description="Zero-shot stock forecasting using Chronos")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the ticker CSV data")
    parser.add_argument("--target_column", type=str, default="Close", help="Target column to forecast")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples for probabilistic forecast")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml file (optional)")
    return parser.parse_args()


def main():
    args = get_args()
    target_is_returns = args.target_column.lower() == "returns"

    config = {
        "model_name": "amazon/chronos-t5-large",
        "target_column": args.target_column,
        "forecast_horizon": args.horizon,
        "num_samples": args.num_samples,
        "device": args.device,
        "context_length": 60,
        "seed": 42,
        "plot_dir": "experiments/baseline/chronos/plots",
    }

    if args.config and os.path.exists(args.config):
        yaml_config = read_yaml(args.config)
        config.update(yaml_config)
        if args.target_column != "Close":
            config["target_column"] = args.target_column
        if args.horizon != 7:
            config["forecast_horizon"] = args.horizon
        if args.num_samples != 20:
            config["num_samples"] = args.num_samples
        if args.device != "cuda":
            config["device"] = args.device

    set_seed(config["seed"])

    print(f"Loading data from {args.csv_path}...")
    try:
        dates, values = load_ticker(args.csv_path, target_col=config["target_column"])
        close_dates, close_values = load_ticker(args.csv_path, target_col="Close")
    except Exception as e:
        print(f"Error loading {args.csv_path}: {e}")
        return

    context_len = min(config["context_length"], len(values))
    context = values[-context_len:]
    print(f"Data loaded. Using last {context_len} points as context.")

    if ChronosPipeline is None:
        raise ImportError("chronos package is not installed. Install requirements first.")

    print(f"Loading Chronos model '{config['model_name']}' on {config['device']}...")
    pipeline = ChronosPipeline.from_pretrained(
        config["model_name"],
        device_map=config["device"],
        torch_dtype=torch.float32,
    )

    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        forecast_samples = pipeline.predict(
            context_tensor,
            prediction_length=config["forecast_horizon"],
            num_samples=config["num_samples"],
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    samples_2d = forecast_samples.squeeze(0)
    median_forecast = torch.quantile(samples_2d, 0.5, dim=0).cpu().numpy()
    low_forecast = torch.quantile(samples_2d, 0.1, dim=0).cpu().numpy()
    high_forecast = torch.quantile(samples_2d, 0.9, dim=0).cpu().numpy()

    last_date = pd.to_datetime(close_dates[-1])
    forecast_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=config["forecast_horizon"],
    )

    median_plot = median_forecast
    low_plot = low_forecast
    high_plot = high_forecast
    if target_is_returns:
        last_close = float(close_values[-1])
        median_plot = last_close * pd.Series(median_forecast).cumsum().apply(np.exp).to_numpy()
        low_plot = last_close * pd.Series(low_forecast).cumsum().apply(np.exp).to_numpy()
        high_plot = last_close * pd.Series(high_forecast).cumsum().apply(np.exp).to_numpy()

    results_df = pd.DataFrame(
        {
            "Date": forecast_dates,
            "Forecast_P10": low_forecast,
            "Forecast_Median": median_forecast,
            "Forecast_P90": high_forecast,
            "Forecast_P10_Close": low_plot if target_is_returns else low_forecast,
            "Forecast_Median_Close": median_plot if target_is_returns else median_forecast,
            "Forecast_P90_Close": high_plot if target_is_returns else high_forecast,
        }
    )
    print("\n=== Forecast Results ===")
    print(results_df.to_string(index=False))

    plot_dir = Path(config["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    ticker = Path(args.csv_path).stem
    out_plot_path = plot_dir / f"{ticker}_chronos_forecast_h{config['forecast_horizon']}.png"

    hist_dates = pd.to_datetime(close_dates[-context_len:])
    hist_vals = close_values[-context_len:]

    plt.figure(figsize=(10, 5))
    plt.plot(hist_dates, hist_vals, color="royalblue", label=f"Historical {ticker} Close", linewidth=2)
    plt.plot(
        forecast_dates,
        median_plot,
        color="tomato",
        label="Forecast Close (P50)" if target_is_returns else "Forecast Median (P50)",
        linestyle="--",
        linewidth=2,
    )
    plt.fill_between(
        forecast_dates,
        low_plot,
        high_plot,
        color="tomato",
        alpha=0.3,
        label="80% PI Close (P10-P90)" if target_is_returns else "80% PI (P10-P90)",
    )
    plt.title(f"Zero-Shot Forecast for {ticker} using {config['model_name']}")
    plt.xlabel("Date")
    plt.ylabel("Close" if target_is_returns else config["target_column"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=300)
    plt.close()
    print(f"Saved plot to {out_plot_path}")


if __name__ == "__main__":
    main()
