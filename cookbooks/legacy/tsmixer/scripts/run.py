"""End-to-end: load universe, build features, train ONE global TSMixer across all tickers."""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.dataset import load_all
from src.preprocessing.pool import build_splits
from src.training.train_global import GlobalTrainCfg, train_global
from src.utils.universe import select_universe


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_cfg(y: dict) -> GlobalTrainCfg:
    return GlobalTrainCfg(
        lookback=y["data"]["lookback"],
        horizon=y["data"]["horizon"],
        batch_size=y["train"]["batch_size"],
        epochs=y["train"]["epochs"],
        lr=float(y["train"]["lr"]),
        weight_decay=float(y["train"]["weight_decay"]),
        patience=y["train"]["patience"],
        grad_clip=float(y["train"]["grad_clip"]),
        huber_delta=float(y["train"]["huber_delta"]),
        quantile_lambda=float(y["train"]["quantile_lambda"]),
        quantile_q=float(y["train"]["quantile_q"]),
        device=y["train"]["device"],
        target_scale=float(y["data"]["scale_returns"]),
        n_blocks=y["model"]["n_blocks"],
        ff_dim=y["model"]["ff_dim"],
        dropout=float(y["model"]["dropout"]),
        loss_type=str(y["train"].get("loss_type", "mixed")),
        ticker_embed_dim=int(y["model"].get("ticker_embed_dim", 8)),
        per_ticker_eval=bool(y["train"].get("per_ticker_eval", False)),
        num_workers=int(y["train"].get("num_workers", 0)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = ap.parse_args()

    with open(args.config) as f:
        y = yaml.safe_load(f)
    set_seed(y["train"]["seed"])
    cfg = build_cfg(y)

    data_dir = ROOT / y["data"]["dir"]
    ckpt_dir = ROOT / y["output"]["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    u = y.get("universe", {})
    universe = select_universe(
        data_dir,
        start=u.get("start", "2020-01-01"),
        end=u.get("end", "2025-01-01"),
        top_n=int(u.get("top_n", 500)),
        min_days=int(u.get("min_days", 252)),
    )
    print(f"Universe: {len(universe)} tickers by liquidity in [{u.get('start')}, {u.get('end')})")
    (ckpt_dir / "universe.txt").write_text("\n".join(universe))

    universe_set = set(universe)
    assets = [asset for asset in load_all(data_dir) if asset[0] in universe_set]
    if not assets:
        raise SystemExit("No assets to train on.")
    print(f"Loaded features for {len(assets)} tickers")

    train_a, val_a, test_a, ticker_to_id, target_scalers = build_splits(
        assets, y["data"]["train_frac"], y["data"]["val_frac"]
    )
    val_agg, test_agg, test_per_ticker = train_global(
        train_a, val_a, test_a, ticker_to_id, target_scalers, cfg, ckpt_dir / "global.pt"
    )

    print("\nAggregate VAL :", val_agg)
    print("Aggregate TEST:", test_agg)
    out = {"val_agg": val_agg, "test_agg": test_agg}
    if test_per_ticker is not None:
        out["test_per_ticker"] = test_per_ticker
    (ckpt_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(f"Saved -> {ckpt_dir / 'results.json'}")


if __name__ == "__main__":
    main()
