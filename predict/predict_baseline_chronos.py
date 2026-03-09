import argparse
import os
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Chronos specific
try:
    from chronos import ChronosPipeline
except ImportError:
    pass

# Fix imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.utils import read_yaml, set_seed
from src.preProcessing.data_preprocessing_chronos import load_ticker

def get_args():
    parser = argparse.ArgumentParser(description="Zero-shot stock forecasting using Chronos-2")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the ticker CSV data")
    parser.add_argument("--target_column", type=str, default="Close", help="Target column to forecast")
    parser.add_argument("--horizon", type=int, default=7, help="Forecast horizon (number of steps/days to predict)")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples for probabilistic forecast")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (e.g. 'cuda', 'cpu', 'mps')")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml file (optional)")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Configuration resolution
    config = {
        "model_name": "amazon/chronos-2-large",
        "target_column": args.target_column,
        "forecast_horizon": args.horizon,
        "num_samples": args.num_samples,
        "device": args.device,
        "context_length": 60,
        "seed": 42,
        "plot_dir": "experiments/baseline/chronos/plots"
    }

    if args.config and os.path.exists(args.config):
        yaml_config = read_yaml(args.config)
        config.update(yaml_config)
        # Override with explicit CLI args if they were provided (using defaults implicitly otherwise)
        if args.target_column != "Close": config["target_column"] = args.target_column
        if args.horizon != 7: config["forecast_horizon"] = args.horizon
        if args.num_samples != 20: config["num_samples"] = args.num_samples
        if args.device != "cuda": config["device"] = args.device

    print("=== Configuration ===")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Set random seed
    set_seed(config["seed"])

    # 2. Data Loading
    print(f"\nLoading data from {args.csv_path}...")
    try:
        series, dates = load_ticker(args.csv_path, target_column=config["target_column"])
    except Exception as e:
        print(f"Error loading {args.csv_path}: {e}")
        return

    # If the series is too young, we simply use the available length
    context_len = min(config["context_length"], len(series))
    context = series[-context_len:]

    print(f"Data loaded. Using last {context_len} days as context.")

    # 3. Model Loading
    print(f"\nLoading Chronos model '{config['model_name']}' on {config['device']}...")
    try:
        pipeline = ChronosPipeline.from_pretrained(
            config["model_name"],
            device_map=config["device"],
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 4. Inference
    print(f"\nForecasting {config['forecast_horizon']} steps...")
    
    # Move context to device, add batch dimension since pipeline.predict expects (batch_size, context_length)
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        # Predict: returns shape [batch_size, num_samples, prediction_length]
        forecast_samples = pipeline.predict(
            context_tensor,
            prediction_length=config["forecast_horizon"],
            num_samples=config["num_samples"]
        )
    
    # Explicit garbage collection to free GPU memory
    torch.cuda.empty_cache()

    # 5. Compute Forecast Statistics (Quantiles)
    # Shape of forecast_samples: [1, num_samples, horizon]
    # Squeeze batch dimension to get [num_samples, horizon]
    samples_2d = forecast_samples.squeeze(0)

    # Calculate median (point forecast) and quantiles (10th, 90th)
    median_forecast = torch.quantile(samples_2d, 0.5, dim=0).cpu().numpy()
    low_forecast = torch.quantile(samples_2d, 0.1, dim=0).cpu().numpy()
    high_forecast = torch.quantile(samples_2d, 0.9, dim=0).cpu().numpy()
    
    # 6. Forecast Output Generation & Printing
    last_date = dates.iloc[-1]
    # Generate business days (B) starting from the next day
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=config["forecast_horizon"])
    
    results_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast_P10": low_forecast,
        "Forecast_Median": median_forecast,
        "Forecast_P90": high_forecast
    })
    
    print("\n=== Forecast Results ===")
    # Print nice table
    pd.set_option('display.max_rows', None)
    print(results_df.to_string(index=False))

    # 7. Plotting & Saving
    plot_dir = Path(config["plot_dir"])
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    ticker = Path(args.csv_path).stem
    out_plot_path = plot_dir / f"{ticker}_chronos_forecast_h{config['forecast_horizon']}.png"
    
    print(f"\nGenerating and saving plot to {out_plot_path}...")
    
    # Historical values for plot
    hist_dates = dates.iloc[-context_len:].values
    hist_vals = context.numpy()
    
    plt.figure(figsize=(10, 5))
    plt.plot(hist_dates, hist_vals, color='royalblue', label=f'Historical {ticker} {config["target_column"]}', linewidth=2)
    plt.plot(forecast_dates, median_forecast, color='tomato', label='Forecast Median (P50)', linestyle='--', linewidth=2)
    plt.fill_between(forecast_dates, low_forecast, high_forecast, color='tomato', alpha=0.3, label='80% Prediction Interval (P10-P90)')
    
    plt.title(f"Zero-Shot Forecast for {ticker} using {config['model_name']}")
    plt.xlabel("Date")
    plt.ylabel(config["target_column"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_plot_path, dpi=300)
    plt.close()
    
    print("Done!")

if __name__ == "__main__":
    main()
