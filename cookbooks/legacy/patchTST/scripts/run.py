"""End-to-end: build features, train PatchTST, evaluate on test."""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import build_splits, load_universe_features
from src.data.universe import select_universe
from src.models.patchtst import PatchTST
from src.preprocessing.features import FEATURE_COLS
from src.training.train import seed_everything, train_model
from src.utils.metrics import evaluate, evaluate_close_prices, log_returns_to_prices


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override max_steps (useful for smoke tests)",
    )
    return p.parse_args()


@torch.no_grad()
def collect_preds(model, datasets, batch_size, device):
    model.eval()
    ds = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, tgts = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        tgts.append(y.numpy())
    return np.concatenate(preds, 0), np.concatenate(tgts, 0)


def collect_close_targets(datasets):
    base_prices = np.concatenate([ds.price_bases() for ds in datasets], axis=0)
    close_targets = np.concatenate([ds.price_targets() for ds in datasets], axis=0)
    return base_prices, close_targets


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
        cfg["train"]["val_every"] = min(cfg["train"]["val_every"], args.max_steps)

    seed_everything(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = ROOT / cfg["data"]["data_dir"]
    L = cfg["data"]["context_length"]
    H = cfg["data"]["horizon"]

    ucfg = cfg["universe"]
    print(
        f"Selecting universe by {ucfg['metric']} in [{ucfg['start']}, {ucfg['end']}]..."
    )
    files = select_universe(
        data_dir,
        start=ucfg["start"],
        end=ucfg["end"],
        top_n=ucfg["top_n"],
        metric=ucfg["metric"],
        min_days=ucfg["min_days"],
    )
    print(
        f"  selected {len(files)} tickers (target={ucfg['top_n']}): "
        f"{[f.stem for f in files[:10]]}{'...' if len(files) > 10 else ''}"
    )

    print("Building features...")
    feats = load_universe_features(
        files, ucfg["start"], ucfg["end"], min_rows=ucfg["min_days"]
    )
    print(f"  {len(feats)} symbols usable  |  channels={FEATURE_COLS}")

    train_sets, val_sets, test_sets, scaler = build_splits(
        feats, L, H, cfg["train"]["train_frac"], cfg["train"]["val_frac"]
    )
    print(
        f"  windows: train={sum(len(s) for s in train_sets)}  "
        f"val={sum(len(s) for s in val_sets)}  "
        f"test={sum(len(s) for s in test_sets)}"
    )

    mcfg = cfg["model"]
    model = PatchTST(
        num_channels=len(FEATURE_COLS),
        context_length=L,
        horizon=H,
        patch_len=mcfg["patch_len"],
        stride=mcfg["stride"],
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        encoder_layers=mcfg["encoder_layers"],
        ffn_dim=mcfg["ffn_dim"],
        dropout=mcfg["dropout"],
        fc_dropout=mcfg["fc_dropout"],
        revin_affine=mcfg["revin_affine"],
        learn_pos_embed=mcfg["learn_pos_embed"],
        target_index=FEATURE_COLS.index("log_ret"),
    )
    nparams = sum(p.numel() for p in model.parameters())
    print(f"  PatchTST params: {nparams/1e6:.2f}M  num_patches={model.num_patches}")

    tcfg = cfg["train"]
    ckpt = ROOT / tcfg["ckpt_path"]
    result = train_model(
        model,
        train_sets,
        val_sets,
        batch_size=tcfg["batch_size"],
        lr=tcfg["lr"],
        weight_decay=tcfg["weight_decay"],
        max_steps=tcfg["max_steps"],
        val_every=tcfg["val_every"],
        patience=tcfg["patience"],
        grad_clip=tcfg["grad_clip"],
        huber_delta=tcfg["huber_delta"],
        ckpt_path=ckpt,
        device=device,
    )

    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    model.to(device)

    print("Evaluating on test...")
    preds_s, tgts_s = collect_preds(model, test_sets, tcfg["batch_size"], device)
    # invert target-channel scaling to raw log-return space
    preds = preds_s * scaler.target_scale_ + scaler.target_mean_
    tgts = tgts_s * scaler.target_scale_ + scaler.target_mean_
    metrics = evaluate(preds, tgts)
    base_prices, close_targets = collect_close_targets(test_sets)
    close_preds = log_returns_to_prices(base_prices, preds)
    metrics["close_price_space"] = evaluate_close_prices(close_preds, close_targets)

    out = {
        "config": cfg,
        "train": result,
        "test_metrics": metrics,
        "n_test": int(len(tgts)),
    }
    predict_dir = ROOT / "predict"
    predict_dir.mkdir(parents=True, exist_ok=True)
    (predict_dir / "results.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(predict_dir / "preds.npz", preds=preds, targets=tgts)
    with open(predict_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved predictions -> predict/preds.npz")
    print("Generating plots...")
    subprocess.run([sys.executable, str(ROOT / "scripts" / "plot.py")], check=True)


if __name__ == "__main__":
    main()
