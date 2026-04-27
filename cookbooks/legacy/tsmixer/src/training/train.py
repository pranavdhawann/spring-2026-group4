"""Training loop with early stopping and cosine annealing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.tsmixer import TSMixer
from ..preprocessing.dataset import TARGET_IDX, WindowDataset
from ..preprocessing.features import FEATURE_COLS
from .evaluate import all_metrics
from .losses import HuberPlusQuantile


@dataclass
class TrainCfg:
    lookback: int
    horizon: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    patience: int
    grad_clip: float
    huber_delta: float
    quantile_lambda: float
    quantile_q: float
    device: str
    target_scale: float
    n_blocks: int
    ff_dim: int
    dropout: float


def _loader(feats: np.ndarray, cfg: TrainCfg, shuffle: bool) -> DataLoader:
    ds = WindowDataset(feats, cfg.lookback, cfg.horizon, target_scale=cfg.target_scale)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False)


def _run_eval(model: TSMixer, loader: DataLoader, loss_fn, device: str):
    model.eval()
    preds, tgts, losses = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            losses.append(loss_fn(out, yb).item() * xb.size(0))
            preds.append(out.cpu().numpy())
            tgts.append(yb.cpu().numpy())
    n = sum(p.shape[0] for p in preds)
    return np.concatenate(preds), np.concatenate(tgts), sum(losses) / max(n, 1)


def train_one_asset(
    train_df, val_df, test_df, cfg: TrainCfg, ckpt_path: Path
) -> Tuple[Dict[str, float], Dict[str, float]]:
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    train_x = train_df[FEATURE_COLS].values
    val_x = val_df[FEATURE_COLS].values
    test_x = test_df[FEATURE_COLS].values

    train_loader = _loader(train_x, cfg, shuffle=True)
    val_loader = _loader(val_x, cfg, shuffle=False)
    test_loader = _loader(test_x, cfg, shuffle=False)

    model = TSMixer(
        lookback=cfg.lookback,
        n_features=len(FEATURE_COLS),
        horizon=cfg.horizon,
        target_idx=TARGET_IDX,
        n_blocks=cfg.n_blocks,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
    ).to(device)

    loss_fn = HuberPlusQuantile(
        loss_type="mixed",
        delta=cfg.huber_delta * cfg.target_scale,
        lam=cfg.quantile_lambda,
        q=cfg.quantile_q,
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    best_val = float("inf")
    best_epoch = -1
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()
        train_loss = running / max(n, 1)

        _, _, val_loss = _run_eval(model, val_loader, loss_fn, device)
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
        elif epoch - best_epoch >= cfg.patience:
            break

        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(
                f"  epoch {epoch:03d} train={train_loss:.5f} val={val_loss:.5f} best={best_val:.5f}"
            )

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    val_pred, val_tgt, _ = _run_eval(model, val_loader, loss_fn, device)
    test_pred, test_tgt, _ = _run_eval(model, test_loader, loss_fn, device)

    # Undo target scaling for reporting
    s = cfg.target_scale
    val_metrics = all_metrics(val_pred / s, val_tgt / s)
    test_metrics = all_metrics(test_pred / s, test_tgt / s)
    return val_metrics, test_metrics
