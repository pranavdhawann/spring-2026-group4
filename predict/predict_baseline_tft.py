"""
TFT Baseline Prediction Script

Loads the best saved TFT model and generates prediction plots
from the test set. Run this anytime to check model quality
without retraining.

Usage:
    PYTHONPATH=. python3 predict/predict_baseline_tft.py
"""

import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataLoader.dataLoaderBaselineAkshit import getTrainTestDataLoader
from src.models.tft_model import TFTModel
from src.preProcessing.tcn_baseline_preprocessing import preprocess_for_tcn
from src.utils import read_json_file, read_yaml
from src.utils.metrics_utils import calculate_regression_metrics


def save_prediction_plots(
    X_test, y_test, y_pred, test_dataset, scaler, save_dir, num_plots=10
):
    os.makedirs(save_dir, exist_ok=True)

    close_idx = 3
    num_plots = min(num_plots, len(y_test))

    # Sample evenly across the test set to show diverse stocks/time periods
    # (consecutive indices are overlapping windows from the same ticker)
    indices = np.linspace(0, len(y_test) - 1, num_plots, dtype=int)

    for plot_num, i in enumerate(indices):
        fig, ax = plt.subplots(figsize=(12, 6))
        input_seq = X_test[i].numpy()
        input_full = scaler.inverse_transform(input_seq)
        input_close = input_full[:, close_idx]
        seq_len = len(input_close)

        close_mean = scaler.mean_[close_idx]
        close_std = scaler.scale_[close_idx]

        actual = (y_test[i].numpy() * close_std) + close_mean
        predicted = (y_pred[i] * close_std) + close_mean
        forecast_horizon = actual.shape[0]

        try:
            pass
        except Exception:
            pass

        ax.plot(
            range(seq_len),
            input_close,
            label="Input (Historical Close)",
            color="blue",
            marker="o",
            markersize=3,
        )

        ax.plot(
            range(seq_len, seq_len + forecast_horizon),
            actual,
            label="Actual Price",
            color="green",
            marker="o",
            markersize=5,
            linestyle="--",
        )

        ax.plot(
            range(seq_len, seq_len + forecast_horizon),
            predicted,
            label="Predicted Price",
            color="red",
            marker="^",
            markersize=5,
            linestyle="--",
        )

        ax.axvline(
            x=seq_len - 1,
            color="gray",
            linestyle=":",
            label="Forecast Start",
        )

        ax.set_title(
            f"Stock Price Prediction (TFT) - Sample {plot_num + 1} (idx={i})",
            fontsize=13,
        )
        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel("Price ($)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"prediction_plot_{plot_num + 1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"  {num_plots} prediction plots saved to: {save_dir}")


def save_scatter_plot(y_true, y_pred, save_path, scaler=None, max_points=5000):
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if scaler is not None:
        close_idx = 3
        close_mean = scaler.mean_[close_idx]
        close_std = scaler.scale_[close_idx]
        y_true = (y_true * close_std) + close_mean
        y_pred = (y_pred * close_std) + close_mean

    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.4, s=10, color="blue")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        label="Perfect Prediction",
    )

    plt.xlabel("Actual Price ($)", fontsize=12)
    plt.ylabel("Predicted Price ($)", fontsize=12)
    plt.title("Predicted vs Actual Prices (TFT)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Scatter plot saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="TFT Baseline Prediction")
    parser.add_argument("--config", type=str, default="config/tft_config.yaml")
    parser.add_argument("--num_plots", type=int, default=10)
    args = parser.parse_args()

    print("=" * 60)
    print("  TFT Baseline — Prediction & Evaluation")
    print("=" * 60)

    # Load config
    config = read_yaml(args.config)
    yaml_config = read_yaml("config/config.yaml")
    config.update(yaml_config)
    config["rand_seed"] = 42
    config["verbose"] = False

    experiment_path = config.get(
        "experiment_path", "experiments/baseline/baseline_results_tft"
    )
    predict_dir = os.path.join(experiment_path, "testpred")
    os.makedirs(predict_dir, exist_ok=True)

    best_model_path = os.path.join(experiment_path, "checkpoints", "best_model.pth")
    if not os.path.exists(best_model_path):
        print(f"  ERROR: No best model found at {best_model_path}")
        print("  Train the model first before running predictions.")
        return

    # Load data
    print("\n[1/4] Loading data...")
    preprocess_cache = os.path.join(experiment_path, "preprocessed_data.pkl")
    dataloader_cache = os.path.join(experiment_path, "dataloaders.pkl")

    # Load test_dataset for date info in plots
    test_dataset = None
    if os.path.exists(dataloader_cache):
        with open(dataloader_cache, "rb") as f:
            dl = pickle.load(f)
        test_dataset = dl.get("test", None)

    if os.path.exists(preprocess_cache):
        print("  Loading cached preprocessed data...")
        with open(preprocess_cache, "rb") as f:
            pp = pickle.load(f)
        X_test = pp["X_test"]
        y_test = pp["y_test"]
        scaler = pp["scaler"]
    else:
        print("  No cache found, running preprocessing...")
        ticker2idx = read_json_file(
            os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
        )
        data_config = {
            "data_path": config["BASELINE_DATA_PATH"],
            "ticker2idx": ticker2idx,
            "test_train_split": 0.2,
            "random_seed": config["rand_seed"],
        }
        train_ds, test_dataset = getTrainTestDataLoader(data_config)
        _, _, X_test, y_test, scaler = preprocess_for_tcn(
            train_ds,
            test_dataset,
            config,
            verbose=False,
        )

    print(f"  X_test: {X_test.shape}")
    print(f"  y_test: {y_test.shape}")

    # Load model
    print("\n[2/4] Loading best TFT model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model_config = config.get("model", {})
    model = TFTModel(model_config).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded ({total_params:,} parameters)")

    # Run inference
    print("\n[3/4] Running inference on test set...")
    batch_size = config.get("training", {}).get("batch_size", 64)
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            actuals = y_batch.numpy()
            all_preds.append(preds)
            all_actuals.append(actuals)

    all_preds = np.concatenate(all_preds, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)

    # Metrics
    metrics = calculate_regression_metrics(all_actuals.flatten(), all_preds.flatten())

    print("\n  Test Metrics:")
    print(f"    MAE:   {metrics['mae']:.4f}")
    print(f"    MSE:   {metrics['mse']:.4f}")
    print(f"    RMSE:  {metrics['rmse']:.4f}")
    print(f"    MAPE:  {metrics['mape']:.2f}%")
    print(f"    SMAPE: {metrics['smape']:.2f}%")

    with open(os.path.join(predict_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Generate plots (exact same as train script)
    print(f"\n[4/4] Generating {args.num_plots} prediction plots...")

    save_prediction_plots(
        X_test,
        y_test,
        all_preds,
        test_dataset,
        scaler,
        save_dir=predict_dir,
        num_plots=args.num_plots,
    )

    save_scatter_plot(
        all_actuals,
        all_preds,
        save_path=os.path.join(predict_dir, "scatter_plot.png"),
        scaler=scaler,
    )

    print("\n" + "=" * 60)
    print("  Prediction complete!")
    print(f"  Results saved to: {predict_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
