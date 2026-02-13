"""
Metrics Utilities
"""

import numpy as np
import torch


def calculate_regression_metrics(y_true, y_pred):
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Shape mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}"
        )

    if len(y_true) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Calculate metrics

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))

    # Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error (MAPE)
    mask = y_true != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf

    # Symmetric Mean Absolute Percentage Error (SMAPE)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    mask = denominator != 0
    if np.sum(mask) > 0:
        smape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    else:
        smape = 0.0

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "smape": float(smape),
    }


def print_metrics(metrics, prefix=""):
    title = f"{prefix} Metrics:" if prefix else "Metrics:"
    print(title)
    print(f"  MSE:   {metrics['mse']:.4f}")
    print(f"  RMSE:  {metrics['rmse']:.4f}")
    print(f"  MAE:   {metrics['mae']:.4f}")
    print(f"  MAPE:  {metrics['mape']:.2f}%")
    print(f"  SMAPE: {metrics['smape']:.2f}%")


if __name__ == "__main__":
    print("Running example for regression metrics...\n")

    # -----------------------------------
    # Example 1: Using NumPy arrays
    # -----------------------------------
    y_true_np = np.array([100, 102, 101, 105, 110])
    y_pred_np = np.array([98, 101, 103, 107, 108])

    metrics_np = calculate_regression_metrics(y_true_np, y_pred_np)
    print_metrics(metrics_np, prefix="NumPy Example")

    print("\n" + "-" * 50 + "\n")

    # -----------------------------------
    # Example 2: Using PyTorch tensors
    # -----------------------------------
    y_true_torch = torch.tensor([50.0, 55.0, 53.0, 60.0, 65.0])
    y_pred_torch = torch.tensor([52.0, 54.0, 50.0, 59.0, 66.0])

    metrics_torch = calculate_regression_metrics(y_true_torch, y_pred_torch)
    print_metrics(metrics_torch, prefix="Torch Example")

    print("\nFinished running toy examples.")
