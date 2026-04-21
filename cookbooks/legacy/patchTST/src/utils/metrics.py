"""Evaluation metrics for multi-step log-return forecasts."""
from __future__ import annotations

import numpy as np


def _summarize_errors(preds: np.ndarray, targets: np.ndarray, mape_floor: float = 1e-3) -> dict:
    errs = preds - targets
    mae_h = np.mean(np.abs(errs), axis=0)
    mse_h = np.mean(errs ** 2, axis=0)
    rmse_h = np.sqrt(mse_h)

    mape_h = []
    for h in range(targets.shape[1]):
        mask = np.abs(targets[:, h]) > mape_floor
        if mask.any():
            mape_h.append(float(np.mean(np.abs(errs[mask, h] / targets[mask, h])) * 100))
        else:
            mape_h.append(float("nan"))
    mape_h = np.array(mape_h)

    return {
        "per_step": {
            "mae": mae_h.tolist(),
            "mse": mse_h.tolist(),
            "rmse": rmse_h.tolist(),
            "mape_pct": mape_h.tolist(),
        },
        "aggregate": {
            "mae": float(mae_h.mean()),
            "mse": float(mse_h.mean()),
            "rmse": float(rmse_h.mean()),
            "mape_pct": float(np.nanmean(mape_h)),
        },
    }


def log_returns_to_prices(base_prices: np.ndarray, log_returns: np.ndarray) -> np.ndarray:
    """Convert log-return forecasts into close-price paths.

    base_prices: (N,) or scalar-like base close before each forecast window.
    log_returns: (N, H) or (H,) log returns.
    """
    base = np.asarray(base_prices, dtype=float)
    log_returns = np.asarray(log_returns, dtype=float)
    if log_returns.ndim == 1:
        return base * np.exp(np.cumsum(log_returns))
    if base.ndim == 0:
        base = np.full((log_returns.shape[0], 1), float(base))
    elif base.ndim == 1:
        base = base[:, None]
    return base * np.exp(np.cumsum(log_returns, axis=1))


def evaluate(preds: np.ndarray, targets: np.ndarray, mape_floor: float = 1e-3) -> dict:
    """preds, targets: (N, H). Returns per-step and aggregate metrics."""
    assert preds.shape == targets.shape
    out = _summarize_errors(preds, targets, mape_floor=mape_floor)

    dir_acc_h = np.mean(np.sign(preds) == np.sign(targets), axis=0) * 100.0

    pred_mean = preds.mean()
    pred_std = preds.std() + 1e-12
    sharpe = float(pred_mean / pred_std)

    out["per_step"]["directional_acc_pct"] = dir_acc_h.tolist()
    out["aggregate"]["directional_acc_pct"] = float(dir_acc_h.mean())
    out["aggregate"]["sharpe_pred"] = sharpe
    return out


def evaluate_close_prices(
    close_preds: np.ndarray,
    close_targets: np.ndarray,
    mape_floor: float = 1e-3,
) -> dict:
    """close_preds, close_targets: (N, H). Returns close-price-space metrics."""
    assert close_preds.shape == close_targets.shape
    return _summarize_errors(close_preds, close_targets, mape_floor=mape_floor)
