"""Training loop: AdamW + cosine LR + early stopping on Huber val loss."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.models.patchtst import PatchTST


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainState:
    step: int = 0
    best_val: float = math.inf
    patience_left: int = 0


def _loader(datasets: List[Dataset], batch_size: int, shuffle: bool) -> DataLoader:
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0
    )


@torch.no_grad()
def _evaluate_loss(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        total += loss_fn(pred, y).item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def train_model(
    model: PatchTST,
    train_sets: List[Dataset],
    val_sets: List[Dataset],
    *,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_steps: int,
    val_every: int,
    patience: int,
    grad_clip: float,
    huber_delta: float,
    ckpt_path: str | Path,
    device: torch.device,
) -> dict:
    train_loader = _loader(train_sets, batch_size, shuffle=True)
    val_loader = _loader(val_sets, batch_size, shuffle=False)

    loss_fn = nn.HuberLoss(delta=huber_delta)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_steps)

    state = TrainState(patience_left=patience)
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history = []

    model.to(device)
    data_iter = iter(train_loader)
    while state.step < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y = next(data_iter)

        model.train()
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        sched.step()
        state.step += 1

        if state.step % val_every == 0 or state.step == max_steps:
            val_loss = _evaluate_loss(model, val_loader, loss_fn, device)
            history.append(
                {
                    "step": state.step,
                    "train_loss": float(loss.item()),
                    "val_loss": val_loss,
                    "lr": sched.get_last_lr()[0],
                }
            )
            print(
                f"[step {state.step:5d}] train={loss.item():.6f}  val={val_loss:.6f}  "
                f"lr={sched.get_last_lr()[0]:.2e}"
            )
            # flush history so plot.py can read it mid-training
            (ckpt_path.parent / "history.json").write_text(json.dumps(history))

            if val_loss < state.best_val - 1e-8:
                state.best_val = val_loss
                state.patience_left = patience
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": state.step,
                        "val_loss": val_loss,
                    },
                    ckpt_path,
                )
            else:
                state.patience_left -= 1
                if state.patience_left <= 0:
                    print(
                        f"Early stopping at step {state.step} (best val={state.best_val:.6f})"
                    )
                    break

    return {"history": history, "best_val": state.best_val, "steps": state.step}
