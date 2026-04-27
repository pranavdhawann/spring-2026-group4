"""Train the LSTM forecaster end-to-end from a YAML config.

Usage:
    python -m scripts.train --config configs/default.yaml
    python -m scripts.train --config configs/default.yaml --csv data/sp500_time_series/aa.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make `src` importable when run as a script.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_ohlcv
from src.models import LSTMForecaster
from src.preprocessing import build_features, prepare_splits
from src.training import TrainConfig, evaluate, train
from src.utils import load_config, set_seed


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument(
        "--csv", type=Path, default=None, help="override data.csv from config"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(int(cfg["training"]["seed"]))

    csv_path = args.csv or (ROOT / cfg["data"]["csv"])
    print(f"loading {csv_path}")
    raw = load_ohlcv(csv_path, date_col=cfg["data"].get("date_col", "Date"))
    print(f"  rows={len(raw)}  range={raw.index.min()} .. {raw.index.max()}")

    print("building features")
    feats = build_features(raw)
    print(f"  features={feats.shape[1]}  usable rows={len(feats)}")

    print("preparing splits")
    splits = prepare_splits(
        feats,
        lookback=int(cfg["window"]["lookback"]),
        horizon=int(cfg["window"]["horizon"]),
        train_frac=float(cfg["split"]["train"]),
        val_frac=float(cfg["split"]["val"]),
    )
    print(
        f"  train={splits.X_train.shape}  val={splits.X_val.shape}  test={splits.X_test.shape}"
    )

    bound = cfg["model"].get("output_bound")
    model = LSTMForecaster(
        n_features=splits.X_train.shape[-1],
        horizon=int(cfg["window"]["horizon"]),
        hidden1=int(cfg["model"]["hidden1"]),
        hidden2=int(cfg["model"]["hidden2"]),
        dropout=float(cfg["model"]["dropout"]),
        output_bound=None if bound is None else float(bound),
    )

    tcfg = TrainConfig(
        lr=float(cfg["training"]["lr"]),
        batch_size=int(cfg["training"]["batch_size"]),
        epochs=int(cfg["training"]["epochs"]),
        patience=int(cfg["training"]["patience"]),
        clip_norm=float(cfg["training"]["clip_norm"]),
        lr_factor=float(cfg["training"]["lr_factor"]),
        lr_patience=int(cfg["training"]["lr_patience"]),
        huber_delta=float(cfg["training"]["huber_delta"]),
        zero_alpha=float(cfg["training"].get("zero_alpha", 0.05)),
        zero_sigma=float(cfg["training"].get("zero_sigma", 0.25)),
        mag_weight_alpha=float(cfg["training"].get("mag_weight_alpha", 0.0)),
        mag_weight_power=float(cfg["training"].get("mag_weight_power", 1.0)),
        mag_weight_cap=float(cfg["training"].get("mag_weight_cap", 5.0)),
        direction_alpha=float(cfg["training"].get("direction_alpha", 0.0)),
    )
    print(f"training on {tcfg.device}")
    train(model, splits, tcfg)

    ckpt_path = ROOT / cfg["artifacts"]["checkpoint"]
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "feat_scaler": splits.feat_scaler,
            "target_scaler": splits.target_scaler,
            "feature_names": splits.feature_names,
            "n_features": splits.X_train.shape[-1],
            "horizon": int(cfg["window"]["horizon"]),
            "lookback": int(cfg["window"]["lookback"]),
            "model_cfg": cfg["model"],
        },
        ckpt_path,
    )
    print(f"saved checkpoint -> {ckpt_path}")

    print("evaluating on test set")
    metrics = evaluate(model, splits, tcfg.device, plot_path=None)
    print("\n=== test metrics (scaled + inverse-transformed log returns) ===")
    print(metrics.to_string(index=False))

    metrics_path = ROOT / cfg["artifacts"]["metrics"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    print(f"saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
