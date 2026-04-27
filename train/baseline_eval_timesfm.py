import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

try:
    import timesfm
except ImportError:
    print("CRITICAL ERROR: 'timesfm' library not found!")
    print("Please run: pip install timesfm")
    print("Note: timesfm strictly requires Python 3.10 or 3.11.")
    sys.exit(1)


# ---------------------------------------------------------
# Plotting Helpers (Directly mapped from TFT architecture)
# ---------------------------------------------------------
def save_scatter_plot(
    y_true, y_pred, save_path, scaler=None, max_points=5000, prefix="Test"
):
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
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

    plt.xlabel("Actual Price ($)", fontsize=12)
    plt.ylabel("Predicted Price ($)", fontsize=12)
    plt.title(f"Predicted vs Actual Prices (TimesFM Zero-Shot {prefix})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_prediction_plots(
    X_test, y_test, y_pred, test_dataset, scaler, save_dir, num_plots=10
):
    os.makedirs(save_dir, exist_ok=True)
    close_idx = 3
    num_plots = min(num_plots, len(y_test))

    for i in range(num_plots):
        fig, ax = plt.subplots(figsize=(12, 6))
        # Handle tensors gracefully across Torch/Numpy modes
        input_seq = (
            X_test[i].numpy() if isinstance(X_test[i], torch.Tensor) else X_test[i]
        )

        input_full = scaler.inverse_transform(input_seq)
        input_close = input_full[:, close_idx]
        seq_len = len(input_close)

        close_mean = scaler.mean_[close_idx]
        close_std = scaler.scale_[close_idx]

        actual_y = (
            y_test[i].numpy() if isinstance(y_test[i], torch.Tensor) else y_test[i]
        )
        pred_y = y_pred[i].numpy() if isinstance(y_pred[i], torch.Tensor) else y_pred[i]

        actual = (actual_y * close_std) + close_mean
        predicted = (pred_y * close_std) + close_mean
        forecast_horizon = actual.shape[0]

        try:
            pass
        except Exception:
            pass

        ax.plot(
            range(seq_len),
            input_close,
            label="Historical Close",
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
            label="Predicted (TimesFM)",
            color="red",
            marker="^",
            markersize=5,
            linestyle="--",
        )
        ax.axvline(x=seq_len - 1, color="gray", linestyle=":", label="Forecast Start")
        ax.set_title(f"Stock Price Prediction (TimesFM) - Sample {i + 1}", fontsize=13)
        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel("Price ($)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"prediction_plot_{i + 1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------
def load_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def read_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "SMAPE": float(smape),
    }


def evaluate_dataset(tfm, dataloader, target_idx, desc="Evaluating"):
    all_y_true = []
    all_y_pred = []

    for X_batch, y_batch in tqdm(dataloader, desc=desc):
        # Extract Univariate Close
        close_history = X_batch[:, :, target_idx].numpy()
        timesfm_inputs = [seq for seq in close_history]

        # Forecast Zero Shot
        forecasts, _ = tfm.forecast(timesfm_inputs)

        all_y_true.append(y_batch.numpy())
        all_y_pred.append(np.array(forecasts))

    return np.concatenate(all_y_true, axis=0), np.concatenate(all_y_pred, axis=0)


def main():
    print("========================================")
    print("  TimesFM Zero-Shot Inference Pipeline  ")
    print("========================================")

    config_path = "config/config.yaml"
    timesfm_config_path = "config/timesfm_config.yaml"
    data_config = load_config(config_path)
    tfm_config = load_config(timesfm_config_path)

    # Set proper output directory mapping
    experiment_path = os.path.join(
        data_config.get("EXPERIMENTS_PATH", "experiments"),
        "baseline/baseline_results_timesfm",
    )
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(os.path.join(experiment_path, "predictions"), exist_ok=True)

    print("\n[1/4] Loading Cached Datasets (TFT Baseline Match)...")
    preprocess_cache = "experiments/baseline/baseline_results_tft/preprocessed_data.pkl"

    if os.path.exists(preprocess_cache):
        with open(preprocess_cache, "rb") as f:
            pp = pickle.load(f)
        X_train, y_train = pp["X_train"], pp["y_train"]
        X_test, y_test = pp["X_test"], pp["y_test"]
        scaler = pp["scaler"]
    else:
        print(
            "CRITICAL: preprocessed_data.pkl cache not found! Please run your TFT training first to generate the strict test indices."
        )
        sys.exit(1)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=tfm_config["evaluation"]["batch_size"],
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=tfm_config["evaluation"]["batch_size"],
        shuffle=False,
    )

    print(f"      Train Samples: {len(X_train)}")
    print(f"      Test Samples:  {len(X_test)}")

    print("\n[2/4] Initializing TimesFM Foundation Model...")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=tfm_config["architecture"]["backend"],
            per_core_batch_size=tfm_config["evaluation"]["batch_size"],
            horizon_len=tfm_config["timesfm"]["horizon_len"],
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id=tfm_config["architecture"]["repo_id"]
        ),
    )

    print("\n[3/4] Running Zero-Shot Inference...")
    target_idx = tfm_config["evaluation"]["target_idx"]

    print("\n==> Evaluating Train Core:")
    y_true_train, y_pred_train = evaluate_dataset(
        tfm, train_loader, target_idx, desc="Train Batches"
    )
    train_metrics = calculate_metrics(y_true_train, y_pred_train)

    print("\n==> Evaluating Test Core:")
    y_true_test, y_pred_test = evaluate_dataset(
        tfm, test_loader, target_idx, desc="Test Batches"
    )
    test_metrics = calculate_metrics(y_true_test, y_pred_test)

    print("\n[4/4] Generating Plots and Reports...")

    # Generate scatter plots
    save_scatter_plot(
        y_true_train,
        y_pred_train,
        os.path.join(experiment_path, "train_predictions_scatter.png"),
        scaler,
        prefix="Train",
    )
    save_scatter_plot(
        y_true_test,
        y_pred_test,
        os.path.join(experiment_path, "test_predictions_scatter.png"),
        scaler,
        prefix="Test",
    )

    # Generate line plots for first 10 test subjects
    save_prediction_plots(
        X_test,
        torch.tensor(y_true_test),
        torch.tensor(y_pred_test),
        test_dataset=None,
        scaler=scaler,
        save_dir=os.path.join(experiment_path, "predictions"),
        num_plots=10,
    )

    # Save standard JSON summary outputs
    final_output = {
        "architecture": tfm_config["architecture"]["name"],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    with open(os.path.join(experiment_path, "metrics.json"), "w") as f:
        json.dump(final_output, f, indent=4)

    print(
        f"\nPipeline Complete! All experiments seamlessly saved to: {experiment_path}"
    )


if __name__ == "__main__":
    main()
