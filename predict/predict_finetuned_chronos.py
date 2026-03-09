"""
Chronos Post-Fine-Tune Inference Script

Given a saved LoRA adapter checkpoint, this script:
1. Loads the base model it was trained on.
2. Generates a baseline (zero-shot) forecast.
3. Merges the LoRA adapter.
4. Generates a fine-tuned forecast.
5. Outputs a CSV of the fine-tuned predictions (quantiles & point forecast).
6. Saves a side-by-side plot comparing both distributions.

Usage:
    python -m predict.predict_finetuned_chronos \
        --adapter_path train/chronos_lora_results/checkpoint-1000 \
        --ticker AAPL --data_dir data/sp500 \
        --horizon 7 --num_samples 20
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure project root is importable regardless of cwd
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preProcessing.data_preprocessing_chronos import load_ticker
from src.utils.utils import read_yaml, set_seed

try:
    from chronos import ChronosPipeline
    from peft import PeftModel
except ImportError:
    ChronosPipeline = None
    PeftModel = None

logger = logging.getLogger(__name__)

# The quantiles we output in the CSV
OUTPUT_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Chronos.")
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the saved LoRA checkpoint directory.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Direct path to a specific ticker CSV.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker name (used with --data_dir if --csv_path is not provided).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Root directory with ticker CSVs.",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="Close",
        help="Target column to forecast.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="Forecast horizon (number of steps to predict).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of samples for probabilistic forecast.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device (e.g., 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a optional config yaml file.",
    )
    return parser.parse_args()


def predict_pipeline(
    pipeline: "ChronosPipeline",
    context: np.ndarray,
    horizon: int,
    num_samples: int,
) -> dict:
    """Run inference for a single setup, returning quantiles and median."""
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        # returns [batch_size, num_samples, prediction_length]
        forecast_samples = pipeline.predict(
            context_tensor,
            prediction_length=horizon,
            num_samples=num_samples,
        )

    # Convert to shape [num_samples, prediction_length]
    samples_2d = forecast_samples.squeeze(0).cpu()

    # Calculate median & specified quantiles
    results = {}
    for q in OUTPUT_QUANTILES:
        results[q] = torch.quantile(samples_2d, q, dim=0).numpy()

    return results


def main():
    if ChronosPipeline is None or PeftModel is None:
        print("Error: chronos-forecasting and peft are required to run this script.")
        sys.exit(1)

    args = get_args()

    # 1. Configuration
    config = {
        "target_column": args.target_column,
        "forecast_horizon": args.horizon,
        "num_samples": args.num_samples,
        "device": args.device,
        "context_length": 60,
        "seed": 42,
        "out_dir_csv": "experiments/baseline/chronos/forecasts",
        "out_dir_plot": "experiments/baseline/chronos/plots",
    }

    if args.config and os.path.exists(args.config):
        yaml_config = read_yaml(args.config)
        config.update(yaml_config)

    set_seed(config["seed"])

    # 2. Resolve data file
    if args.csv_path and os.path.exists(args.csv_path):
        csv_file = args.csv_path
        ticker_name = Path(csv_file).stem.upper()
    elif args.ticker and args.data_dir:
        ticker_name = args.ticker.upper()
        csv_file = Path(args.data_dir) / f"{ticker_name.lower()}.csv"
    else:
        print("Error: Must provide either --csv_path or both --ticker and --data_dir.")
        sys.exit(1)

    print(f"\nLoading data for {ticker_name} from {csv_file}")
    try:
        dates, values = load_ticker(str(csv_file), target_col=config["target_column"])
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        sys.exit(1)

    context_len = min(config["context_length"], len(values))
    context = values[-context_len:]
    hist_dates = dates[-context_len:]
    print(f"Context loaded: last {context_len} periods.")

    # 3. Model Loading (Base -> LoRA)
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"Error: Adapter path {adapter_path} not found.")
        sys.exit(1)

    config_json = adapter_path / "adapter_config.json"
    if not config_json.exists():
        print(f"Error: {config_json} not found. Ensure this is a valid PEFT run.")
        sys.exit(1)

    with open(config_json, "r") as f:
        adapter_cfg = json.load(f)

    base_model_name = adapter_cfg.get("base_model_name_or_path", "amazon/chronos-2-large")

    print(f"\nLoading BASE model '{base_model_name}' on {config['device']}...")
    pipeline = ChronosPipeline.from_pretrained(
        base_model_name,
        device_map=config["device"],
        torch_dtype=torch.float32,
    )

    # 4. Infer Base Forecast
    print("Generating base zero-shot forecast...")
    base_preds = predict_pipeline(
        pipeline, context, config["forecast_horizon"], config["num_samples"]
    )

    # 5. Merge LoRA Adapter
    print(f"\nMerging LoRA adapter from '{adapter_path}'...")
    pipeline.model.model = PeftModel.from_pretrained(pipeline.model.model, str(adapter_path))
    pipeline.model.model = pipeline.model.model.merge_and_unload()
    print("Adapter merged.")

    # 6. Infer Fine-Tuned Forecast
    print("Generating fine-tuned forecast...")
    ft_preds = predict_pipeline(
        pipeline, context, config["forecast_horizon"], config["num_samples"]
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 7. Generate Output CSV
    last_date = dates[-1]
    # Business days (B) starting from next day
    forecast_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=config["forecast_horizon"]
    )

    results_df = pd.DataFrame({"Date": forecast_dates})
    results_df["point_forecast"] = ft_preds[0.5]  # median
    for q in OUTPUT_QUANTILES:
        key_name = f"q{int(q*100)}"
        results_df[key_name] = ft_preds[q]

    out_csv_dir = Path(config["out_dir_csv"])
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_csv_dir / f"{ticker_name}_finetuned_h{config['forecast_horizon']}.csv"
    results_df.to_csv(csv_out, index=False)
    print(f"\nSaved fine-tuned forecast CSV to: {csv_out}")

    # 8. Plot Side-by-Side Comparison
    out_plot_dir = Path(config["out_dir_plot"])
    out_plot_dir.mkdir(parents=True, exist_ok=True)
    plot_out = out_plot_dir / f"{ticker_name}_comparison_h{config['forecast_horizon']}.png"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    def plot_scenario(ax, title, preds):
        ax.plot(hist_dates, context, color='royalblue', label='Historical', linewidth=2)
        ax.plot(forecast_dates, preds[0.5], color='tomato', label='Forecast Median (P50)', linestyle='--', linewidth=2)
        ax.fill_between(
            forecast_dates,
            preds[0.1],
            preds[0.9],
            color='tomato', alpha=0.3, label='80% Prediction Interval (P10-P90)'
        )
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        ax.legend()
        # Rotate dates
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

    plot_scenario(ax1, f"Zero-Shot Base ({base_model_name})", base_preds)
    plot_scenario(ax2, "Fine-Tuned (with LoRA Adapter)", ft_preds)
    ax1.set_ylabel(config["target_column"])

    plt.suptitle(f"Chronos Forecast Comparison: {ticker_name}", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved side-by-side plot to: {plot_out}\nDone!")


if __name__ == "__main__":
    main()
