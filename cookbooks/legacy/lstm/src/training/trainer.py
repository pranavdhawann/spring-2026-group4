"""Training loop with Huber loss, grad clipping, LR plateau, and early stopping."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import BoundedAntiZeroHuber


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    patience: int = 15
    clip_norm: float = 1.0
    lr_factor: float = 0.5
    lr_patience: int = 5
    huber_delta: float = 1.0
    zero_alpha: float = 0.05
    zero_sigma: float = 0.25
    mag_weight_alpha: float = 0.0
    mag_weight_power: float = 1.0
    mag_weight_cap: float = 5.0
    direction_alpha: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _run_epoch(model, loader, loss_fn, device, opt=None, clip_norm=None) -> float:
    training = opt is not None
    model.train(training)
    total, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.set_grad_enabled(training):
            pred = model(xb)
            loss = loss_fn(pred, yb)
        if training:
            opt.zero_grad()
            loss.backward()
            if clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / max(count, 1)


def train(model: nn.Module, splits, cfg: TrainConfig) -> dict:
    device = torch.device(cfg.device)
    model.to(device)

    # Scale magnitude weights by the average train target magnitude in scaled space.
    mag_scale = max(float(np.mean(np.abs(splits.y_train))), 1e-6)
    loss_fn = BoundedAntiZeroHuber(
        delta=cfg.huber_delta,
        zero_alpha=cfg.zero_alpha,
        zero_sigma=cfg.zero_sigma,
        mag_weight_alpha=cfg.mag_weight_alpha,
        mag_weight_power=cfg.mag_weight_power,
        mag_weight_scale=mag_scale,
        mag_weight_cap=cfg.mag_weight_cap,
        direction_alpha=cfg.direction_alpha,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
    )

    train_loader = _loader(splits.X_train, splits.y_train, cfg.batch_size, shuffle=True)
    val_loader = _loader(splits.X_val, splits.y_val, cfg.batch_size, shuffle=False)

    best_val = math.inf
    best_state = None
    bad = 0
    history = {"train": [], "val": []}

    for epoch in range(1, cfg.epochs + 1):
        tr = _run_epoch(model, train_loader, loss_fn, device, opt, cfg.clip_norm)
        va = _run_epoch(model, val_loader, loss_fn, device)
        sched.step(va)
        history["train"].append(tr)
        history["val"].append(va)

        improved = va < best_val - 1e-6
        if improved:
            best_val = va
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

        print(
            f"epoch {epoch:3d}  train={tr:.6f}  val={va:.6f}"
            f"  lr={opt.param_groups[0]['lr']:.2e}"
            f"{'  *' if improved else ''}"
        )

        if bad >= cfg.patience:
            print(f"early stop at epoch {epoch} (best val={best_val:.6f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history
