"""Per-horizon test metrics on both scaled and inverse-transformed targets."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def predict(
    model: nn.Module, X: np.ndarray, device: str, batch_size: int = 256
) -> np.ndarray:
    model.eval()
    device = torch.device(device)
    model.to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def _per_horizon_metrics(
    pred_scaled: np.ndarray,
    truth_scaled: np.ndarray,
    pred_original: np.ndarray,
    truth_original: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for h in range(pred_scaled.shape[1]):
        p_scaled, t_scaled = pred_scaled[:, h], truth_scaled[:, h]
        err_scaled = p_scaled - t_scaled
        denom_scaled = np.where(np.abs(t_scaled) < 1e-6, np.nan, t_scaled)

        p_original, t_original = pred_original[:, h], truth_original[:, h]
        err_original = p_original - t_original
        denom_original = np.where(np.abs(t_original) < 1e-6, np.nan, t_original)

        rows.append(
            {
                "horizon": f"day+{h + 1}",
                "MAE_scaled": float(np.mean(np.abs(err_scaled))),
                "MSE_scaled": float(np.mean(err_scaled**2)),
                "RMSE_scaled": math.sqrt(float(np.mean(err_scaled**2))),
                "MAPE_scaled_%": float(
                    np.nanmean(np.abs(err_scaled / denom_scaled)) * 100.0
                ),
                "MAE_original": float(np.mean(np.abs(err_original))),
                "MSE_original": float(np.mean(err_original**2)),
                "RMSE_original": math.sqrt(float(np.mean(err_original**2))),
                "MAPE_original_%": float(
                    np.nanmean(np.abs(err_original / denom_original)) * 100.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot(pred: np.ndarray, truth: np.ndarray, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    h = pred.shape[1]
    fig, axes = plt.subplots(h, 1, figsize=(10, 2.4 * h), sharex=True)
    if h == 1:
        axes = [axes]
    for i in range(h):
        ax = axes[i]
        ax.plot(truth[:, i], label="actual", linewidth=0.9)
        ax.plot(pred[:, i], label="predicted", linewidth=0.9, alpha=0.8)
        ax.set_title(f"day+{i + 1} log return")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


def evaluate(
    model: nn.Module,
    splits,
    device: str,
    plot_path: Path | None = None,
) -> pd.DataFrame:
    preds_scaled = predict(model, splits.X_test, device)
    truth_scaled = splits.y_test
    preds_original = splits.target_scaler.inverse_transform(preds_scaled)
    truth_original = splits.target_scaler.inverse_transform(truth_scaled)

    metrics = _per_horizon_metrics(
        preds_scaled, truth_scaled, preds_original, truth_original
    )
    if plot_path is not None:
        _plot(preds_original, truth_original, Path(plot_path))
    return metrics
