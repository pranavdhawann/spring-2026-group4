"""Evaluation metrics for log-return forecasts."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    err = pred - target
    mae = np.mean(np.abs(err))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.where(np.abs(target) < 1e-6, 1e-6, np.abs(target))
    mape = float(np.mean(np.abs(err) / denom))
    return {"MAE": float(mae), "MSE": mse, "RMSE": rmse, "MAPE": mape}


def price_path_from_log_returns(
    anchor_close: np.ndarray, log_returns: np.ndarray
) -> np.ndarray:
    anchor = np.asarray(anchor_close, dtype=np.float64).reshape(-1)
    returns = np.asarray(log_returns, dtype=np.float64)
    if np.any(anchor <= 0.0):
        raise ValueError("anchor_close must be positive to reconstruct prices.")
    growth = np.exp(np.cumsum(returns, axis=1))
    return anchor[:, None] * growth


def price_metrics(pred_price: np.ndarray, target_price: np.ndarray) -> dict:
    err = pred_price - target_price
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    return {"PriceMAE": mae, "PriceMSE": mse, "PriceRMSE": rmse}


def directional_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.sign(pred) == np.sign(target)))


def magnitude_ratio(pred: np.ndarray, target: np.ndarray) -> float:
    denom = float(np.mean(np.abs(target)))
    if denom < 1e-12:
        return 0.0
    return float(np.mean(np.abs(pred)) / denom)


def information_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred.flatten()
    t = target.flatten()
    if np.std(p) < 1e-12 or np.std(t) < 1e-12:
        return 0.0
    ic, _ = spearmanr(p, t)
    return float(ic) if np.isfinite(ic) else 0.0


def long_short_sharpe(pred: np.ndarray, target: np.ndarray) -> float:
    """Simple strategy: position = sign(pred) at each horizon step. Returns annualized Sharpe."""
    pos = np.sign(pred)
    pnl = (pos * target).flatten()
    if pnl.std() < 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * pnl.mean() / pnl.std())


def all_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    pred_price: np.ndarray | None = None,
    target_price: np.ndarray | None = None,
) -> dict:
    m = regression_metrics(pred, target)
    m["DirAcc"] = directional_accuracy(pred, target)
    m["MR"] = magnitude_ratio(pred, target)
    m["IC"] = information_coefficient(pred, target)
    m["Sharpe"] = long_short_sharpe(pred, target)
    if pred_price is not None and target_price is not None:
        m.update(price_metrics(pred_price, target_price))
    return m
