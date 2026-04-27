"""Train one global TSMixer pooled across all tickers."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..models.tsmixer import TSMixer
from ..preprocessing.dataset import TARGET_IDX
from ..preprocessing.features import FEATURE_COLS
from ..preprocessing.pool import AssetWindows, PooledWindowDataset, TargetScalerStats
from .evaluate import all_metrics, price_metrics, price_path_from_log_returns
from .losses import ReturnLoss


@dataclass
class GlobalTrainCfg:
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
    loss_type: str = "mixed"
    ticker_embed_dim: int = 8
    per_ticker_eval: bool = False
    num_workers: int = 0


def _loader(
    assets: List[AssetWindows], cfg: GlobalTrainCfg, shuffle: bool
) -> DataLoader:
    ds = PooledWindowDataset(assets, cfg.lookback, cfg.horizon, target_scale=1.0)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


def _run_eval(model, loader, loss_fn, device):
    model.eval()
    preds, tgts, ids, anchors, total, n = [], [], [], [], 0.0, 0
    with torch.no_grad():
        for xb, yb, aid, anchor_close in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            aid = aid.to(device, non_blocking=True)
            out = model(xb, ticker_id=aid)
            total += loss_fn(out, yb).item() * xb.size(0)
            n += xb.size(0)
            preds.append(out.cpu().numpy())
            tgts.append(yb.cpu().numpy())
            ids.append(aid.cpu().numpy())
            anchors.append(anchor_close.cpu().numpy())
    return (
        np.concatenate(preds),
        np.concatenate(tgts),
        np.concatenate(ids),
        np.concatenate(anchors),
        total / max(n, 1),
    )


def _build_scaler_tables(
    ticker_to_id: Dict[str, int], target_scalers: Dict[str, TargetScalerStats]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    id_to_ticker = [""] * len(ticker_to_id)
    centers = np.zeros(len(ticker_to_id), dtype=np.float32)
    scales = np.ones(len(ticker_to_id), dtype=np.float32)
    for ticker, tid in ticker_to_id.items():
        id_to_ticker[tid] = ticker
        stats = target_scalers.get(ticker, TargetScalerStats(center=0.0, scale=1.0))
        centers[tid] = float(stats.center)
        s = float(stats.scale)
        scales[tid] = s if abs(s) > 1e-8 else 1.0
    return id_to_ticker, centers, scales


def _inverse_scale(
    arr: np.ndarray, ids: np.ndarray, centers: np.ndarray, scales: np.ndarray
) -> np.ndarray:
    c = centers[ids][:, None]
    s = scales[ids][:, None]
    return arr * s + c


def _price_metrics_from_outputs(
    pred: np.ndarray,
    tgt: np.ndarray,
    ids: np.ndarray,
    anchors: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
) -> dict:
    pred_raw = _inverse_scale(pred, ids, centers, scales)
    tgt_raw = _inverse_scale(tgt, ids, centers, scales)
    pred_price = price_path_from_log_returns(anchors, pred_raw)
    tgt_price = price_path_from_log_returns(anchors, tgt_raw)
    return price_metrics(pred_price, tgt_price)


def _all_metrics_from_outputs(
    pred: np.ndarray,
    tgt: np.ndarray,
    ids: np.ndarray,
    anchors: np.ndarray,
    centers: np.ndarray,
    scales: np.ndarray,
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_raw = _inverse_scale(pred, ids, centers, scales)
    tgt_raw = _inverse_scale(tgt, ids, centers, scales)
    pred_price = price_path_from_log_returns(anchors, pred_raw)
    tgt_price = price_path_from_log_returns(anchors, tgt_raw)
    return (
        all_metrics(pred_raw, tgt_raw, pred_price=pred_price, target_price=tgt_price),
        pred_raw,
        tgt_raw,
        pred_price,
        tgt_price,
    )


def _per_ticker_metrics(
    pred: np.ndarray,
    tgt: np.ndarray,
    ids: np.ndarray,
    id_to_ticker: List[str],
    pred_price: np.ndarray | None = None,
    tgt_price: np.ndarray | None = None,
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    buckets = defaultdict(list)
    for k in range(len(ids)):
        buckets[int(ids[k])].append(k)
    for aid, idxs in buckets.items():
        p = pred[idxs]
        t = tgt[idxs]
        pp = pred_price[idxs] if pred_price is not None else None
        tp = tgt_price[idxs] if tgt_price is not None else None
        out[id_to_ticker[aid]] = all_metrics(p, t, pred_price=pp, target_price=tp)
    return out


def train_global(
    train_assets: List[AssetWindows],
    val_assets: List[AssetWindows],
    test_assets: List[AssetWindows],
    ticker_to_id: Dict[str, int],
    target_scalers: Dict[str, TargetScalerStats],
    cfg: GlobalTrainCfg,
    ckpt_path: Path,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, dict] | None]:
    device = cfg.device if (torch.cuda.is_available() or cfg.device == "cpu") else "cpu"
    train_loader = _loader(train_assets, cfg, shuffle=True)
    val_loader = _loader(val_assets, cfg, shuffle=False)
    test_loader = _loader(test_assets, cfg, shuffle=False)

    model = TSMixer(
        lookback=cfg.lookback,
        n_features=len(FEATURE_COLS),
        horizon=cfg.horizon,
        target_idx=TARGET_IDX,
        n_blocks=cfg.n_blocks,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        num_tickers=len(ticker_to_id),
        ticker_embed_dim=cfg.ticker_embed_dim,
    ).to(device)

    loss_fn = ReturnLoss(
        cfg.loss_type, cfg.huber_delta, cfg.quantile_lambda, cfg.quantile_q
    )
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    id_to_ticker, centers, scales = _build_scaler_tables(ticker_to_id, target_scalers)

    best_val = float("inf")
    best_epoch = -1
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Global train: train_windows={len(train_loader.dataset)} "
        f"val_windows={len(val_loader.dataset)} test_windows={len(test_loader.dataset)} device={device}"
    )

    for epoch in range(cfg.epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb, aid, _ in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            aid = aid.to(device, non_blocking=True)
            opt.zero_grad()
            out = model(xb, ticker_id=aid)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        sched.step()
        train_loss = running / max(n, 1)

        vp, vt, vi, va, val_loss = _run_eval(model, val_loader, loss_fn, device)
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ticker_to_id": ticker_to_id,
                    "target_scalers": {
                        t: {"center": float(s.center), "scale": float(s.scale)}
                        for t, s in target_scalers.items()
                    },
                    "ticker_embed_dim": cfg.ticker_embed_dim,
                    "feature_cols": FEATURE_COLS,
                },
                ckpt_path,
            )
        val_metrics, vp_raw, vt_raw, vp_price, vt_price = _all_metrics_from_outputs(
            vp, vt, vi, va, centers, scales
        )
        print(
            f"  epoch {epoch:03d} train={train_loss:.5f} val={val_loss:.5f} best={best_val:.5f}"
            f" DA={val_metrics['DirAcc']:.4f} MR={val_metrics['MR']:.4f}"
            f"{'  *' if improved else ''}"
        )
        if cfg.per_ticker_eval:
            val_per_ticker = _per_ticker_metrics(
                vp_raw,
                vt_raw,
                vi,
                id_to_ticker,
                pred_price=vp_price,
                tgt_price=vt_price,
            )
            da_mr = {
                t: {"DA": round(m["DirAcc"], 4), "MR": round(m["MR"], 4)}
                for t, m in val_per_ticker.items()
            }
            print(f"    val per-ticker DA/MR: {da_mr}")
        if epoch - best_epoch >= cfg.patience:
            print(f"  early stop @ epoch {epoch} (best epoch {best_epoch})")
            break

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state_dict)

    vp, vt, vi, va, _ = _run_eval(model, val_loader, loss_fn, device)
    tp, tt, ti, ta, _ = _run_eval(model, test_loader, loss_fn, device)
    val_agg, _, _, _, _ = _all_metrics_from_outputs(vp, vt, vi, va, centers, scales)
    test_agg, tp_raw, tt_raw, tp_price, tt_price = _all_metrics_from_outputs(
        tp, tt, ti, ta, centers, scales
    )
    test_per_ticker = (
        _per_ticker_metrics(
            tp_raw, tt_raw, ti, id_to_ticker, pred_price=tp_price, tgt_price=tt_price
        )
        if cfg.per_ticker_eval
        else None
    )
    return val_agg, test_agg, test_per_ticker
