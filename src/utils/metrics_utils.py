import numpy as np


def calculate_regression_metrics(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    if y_true.size != y_pred.size:
        raise ValueError(f"Shape mismatch: y_true={y_true.size}, y_pred={y_pred.size}")

    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    denom_mape = np.where(np.abs(y_true) < eps, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs(err) / denom_mape) * 100.0)

    denom_smape = np.abs(y_true) + np.abs(y_pred)
    denom_smape = np.where(denom_smape < eps, np.nan, denom_smape)
    smape = float(np.nanmean(2.0 * np.abs(err) / denom_smape) * 100.0)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "smape": smape,
    }


def print_metrics(metrics, prefix=""):
    head = f"{prefix} " if prefix else ""
    print(f"{head}MSE   : {metrics['mse']:.6f}")
    print(f"{head}RMSE  : {metrics['rmse']:.6f}")
    print(f"{head}MAE   : {metrics['mae']:.6f}")
    print(f"{head}MAPE  : {metrics['mape']:.4f}%")
    print(f"{head}SMAPE : {metrics['smape']:.4f}%")
