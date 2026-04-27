"""Inference: load trained weights, forecast a ticker, plot history + actual vs predicted.

Usage:
    python scripts/forecast.py --ticker aal
    python scripts/forecast.py --ticker aal --history-days 30

The context window is shifted back H days so the final H rows of the CSV
serve as actual future values for comparison with the model forecast.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.universe import _load_ohlcv
from src.models.patchtst import PatchTST
from src.preprocessing.features import FEATURE_COLS, build_features

PREDICT = ROOT / "predict"
PLOTS   = PREDICT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True, help="Ticker symbol (CSV stem, e.g. 'aal')")
    p.add_argument("--history-days", type=int, default=30,
                   help="Days of history to show before the forecast window (default 30)")
    p.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    p.add_argument("--ckpt",   default=str(PREDICT / "best.pt"))
    return p.parse_args()


def load_artifacts(cfg_path: str, ckpt_path: str):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    with open(PREDICT / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    mcfg = cfg["model"]
    L = cfg["data"]["context_length"]
    H = cfg["data"]["horizon"]
    model = PatchTST(
        num_channels=len(FEATURE_COLS), context_length=L, horizon=H,
        patch_len=mcfg["patch_len"], stride=mcfg["stride"],
        d_model=mcfg["d_model"], n_heads=mcfg["n_heads"],
        encoder_layers=mcfg["encoder_layers"], ffn_dim=mcfg["ffn_dim"],
        dropout=mcfg["dropout"], fc_dropout=mcfg["fc_dropout"],
        revin_affine=mcfg["revin_affine"], learn_pos_embed=mcfg["learn_pos_embed"],
        target_index=FEATURE_COLS.index("log_ret"),
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, scaler, cfg


def load_ticker(ticker: str, data_dir: Path, context_length: int, horizon: int, history_days: int):
    csv_path = data_dir / f"{ticker.lower()}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No CSV found for '{ticker}' at {csv_path}")

    raw  = _load_ohlcv(csv_path)
    feats = build_features(raw)
    raw_aligned = raw.iloc[len(raw) - len(feats):].reset_index(drop=True)

    need = context_length + horizon
    if len(feats) < need:
        raise ValueError(f"Need >= {need} rows after feature engineering, got {len(feats)}")

    # Shift window back by H so actual future prices exist in the CSV
    ctx_end   = len(feats) - horizon          # exclusive end of context
    ctx_start = ctx_end - context_length      # inclusive start

    window_feats  = feats[FEATURE_COLS].to_numpy()[ctx_start:ctx_end]   # (L, C)
    actual_future = raw_aligned.iloc[ctx_end : ctx_end + horizon]       # H actual rows

    # History: `history_days` rows ending at ctx_end (last row before forecast)
    hist_start = max(ctx_end - history_days, 0)
    history    = raw_aligned.iloc[hist_start:ctx_end]

    return window_feats, history, actual_future


@torch.no_grad()
def run_forecast(model, window_feats: np.ndarray, scaler) -> np.ndarray:
    x = scaler.transform(window_feats).astype(np.float32)
    pred_scaled = model(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
    return pred_scaled * scaler.target_scale_ + scaler.target_mean_   # raw log-rets (H,)


def log_rets_to_prices(base: float, log_rets: np.ndarray) -> np.ndarray:
    return base * np.exp(np.cumsum(log_rets))


def plot(ticker, history, actual_future, pred_prices, pred_log_rets, rmse_per_step):
    hist_dates   = pd.to_datetime(history["date"]).dt.tz_localize(None)
    hist_prices  = history["close"].to_numpy(dtype=float)
    last_close   = hist_prices[-1]

    act_dates     = pd.to_datetime(actual_future["date"]).dt.tz_localize(None)
    act_close_raw = actual_future["close"].to_numpy(dtype=float)
    prev_closes   = np.concatenate([[last_close], act_close_raw[:-1]])
    act_prices    = last_close * np.exp(np.cumsum(np.log(act_close_raw / prev_closes)))

    # pivot date — last history point, shared by all three lines
    pivot_date  = hist_dates.iloc[-1]
    pivot_price = last_close

    # prepend pivot so lines are continuous from history
    all_act_dates  = [pivot_date] + list(act_dates)
    all_act_prices = np.concatenate([[pivot_price], act_prices])
    all_pred_dates  = [pivot_date] + list(act_dates)
    all_pred_prices = np.concatenate([[pivot_price], pred_prices])

    fig, ax = plt.subplots(figsize=(13, 5))

    # history
    ax.plot(hist_dates, hist_prices,
            color="#4C72B0", linewidth=2.0, label="History")

    # actual future — solid green
    ax.plot(all_act_dates, all_act_prices,
            color="#2ca02c", linewidth=2.0, marker="o", markersize=5,
            label="Actual")

    # predicted — solid orange
    ax.plot(all_pred_dates, all_pred_prices,
            color="#DD8452", linewidth=2.0, marker="s", markersize=5,
            label="Predicted")

    # confidence band
    if rmse_per_step:
        upper = [pivot_price] + [p * np.exp(r) for p, r in zip(pred_prices, rmse_per_step)]
        lower = [pivot_price] + [p * np.exp(-r) for p, r in zip(pred_prices, rmse_per_step)]
        ax.fill_between(all_pred_dates, lower, upper,
                        alpha=0.12, color="#DD8452", label="+-1 RMSE")

    # forecast start marker
    ax.axvline(pivot_date, color="gray", linestyle=":", linewidth=1.2)
    ax.text(pivot_date, 0.02, " forecast start",
            color="gray", fontsize=8, va="bottom", transform=ax.get_xaxis_transform())

    ax.set_title(f"{ticker.upper()} — {len(hist_prices)}-Day History + {len(act_prices)}-Day Forecast")
    ax.set_ylabel("Price ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    out = PLOTS / f"forecast_{ticker.lower()}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved -> {out.relative_to(ROOT)}")


def main():
    args  = parse_args()
    model, scaler, cfg = load_artifacts(args.config, args.ckpt)
    L = cfg["data"]["context_length"]
    H = cfg["data"]["horizon"]
    data_dir = ROOT / cfg["data"]["data_dir"]

    print(f"Loading ticker: {args.ticker.upper()}")
    window_feats, history, actual_future = load_ticker(
        args.ticker, data_dir, L, H, args.history_days)

    pred_log_rets = run_forecast(model, window_feats, scaler)
    base_price    = float(history["close"].iloc[-1])
    pred_prices   = log_rets_to_prices(base_price, pred_log_rets)

    rmse_per_step = None
    if (PREDICT / "results.json").exists():
        rmse_per_step = json.loads((PREDICT / "results.json").read_text())[
            "test_metrics"]["per_step"]["rmse"]

    plot(args.ticker, history, actual_future, pred_prices, pred_log_rets, rmse_per_step)

    act_prices = actual_future["close"].to_numpy(dtype=float)
    print(f"\n{'':8} {'Pred log-ret':>12} {'Pred $':>10} {'Actual $':>10} {'Error':>8}")
    print("-" * 52)
    for i, (lr, pp, ap) in enumerate(zip(pred_log_rets, pred_prices, act_prices)):
        print(f"h+{i+1:<5}  {lr*100:>+10.3f}%  ${pp:>8.2f}  ${ap:>8.2f}  {pp-ap:>+7.2f}")


if __name__ == "__main__":
    main()
