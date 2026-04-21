"""Plot 30 days of history + 5-day actual vs. predicted prices for a ticker.

Axis is reconstructed close price:
    price[t+i] = last_close * exp(cumsum(log_returns[1..i]))
History / actual / predicted lines are all anchored at (end_date, last_close) so
they are visually connected.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.tsmixer import TSMixer
from src.preprocessing.dataset import TARGET_IDX, load_asset
from src.preprocessing.features import FEATURE_COLS

HISTORY_DAYS = 30


def _load_close_series(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, usecols=["Date", "Close"])
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    return df.set_index("Date")["Close"].astype(float).sort_index()


def _reconstruct_prices(last_close: float, log_returns: np.ndarray) -> np.ndarray:
    """Return price path anchored at last_close (index 0) then last_close*exp(cumsum(r))."""
    cum = np.concatenate([[0.0], np.cumsum(log_returns)])
    return last_close * np.exp(cum)


def _extract_features_frame(load_result: object) -> pd.DataFrame:
    """Accept legacy/new loader outputs and return the features DataFrame."""
    if isinstance(load_result, pd.DataFrame):
        return load_result

    if isinstance(load_result, tuple):
        frames = [item for item in load_result if isinstance(item, pd.DataFrame)]
        if len(frames) == 1:
            return frames[0]

    raise TypeError(
        "load_asset(...) must return a pandas DataFrame or a tuple containing exactly "
        f"one pandas DataFrame, got {type(load_result).__name__}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--end_index", type=int, default=-1,
                    help="Row index (full feature frame) of the last lookback day; default = last valid")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        y = yaml.safe_load(f)
    L = y["data"]["lookback"]
    H = y["data"]["horizon"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    csv = ROOT / y["data"]["dir"] / f"{args.ticker}.csv"
    feats = _extract_features_frame(load_asset(csv))
    raw_x = feats[FEATURE_COLS].values.astype(np.float32)
    dates = feats.index

    ckpt = Path(args.ckpt) if args.ckpt else ROOT / y["output"]["ckpt_dir"] / "global.pt"
    payload = torch.load(ckpt, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        ticker_to_id = payload.get("ticker_to_id", {})
        target_scalers = payload.get("target_scalers", {})
        ticker_embed_dim = int(payload.get("ticker_embed_dim", y["model"].get("ticker_embed_dim", 8)))
    else:
        state_dict = payload
        ticker_to_id = {}
        target_scalers = {}
        ticker_embed_dim = int(y["model"].get("ticker_embed_dim", 8))

    scaler_stats = target_scalers.get(args.ticker, {"center": 0.0, "scale": 1.0})
    center = float(scaler_stats.get("center", 0.0))
    scale = float(scaler_stats.get("scale", 1.0))
    if abs(scale) < 1e-8:
        scale = 1.0
    X = raw_x.copy()
    X[:, TARGET_IDX] = ((X[:, TARGET_IDX] - center) / scale).astype(np.float32)

    end = args.end_index if args.end_index >= 0 else len(X) - H - 1
    start = end - L + 1
    if start < 0 or end + H >= len(X):
        raise SystemExit(f"Window out of bounds: L={L} H={H} len={len(X)} end={end}")

    window = X[start : end + 1]
    future_ret = raw_x[end + 1 : end + 1 + H, TARGET_IDX]
    future_dates = dates[end + 1 : end + 1 + H]

    hist_n = min(HISTORY_DAYS, L)
    hist_slice = slice(end - hist_n + 1, end + 1)
    hist_dates = dates[hist_slice]

    # --- model prediction (raw log returns) ---
    model = TSMixer(
        lookback=L,
        n_features=len(FEATURE_COLS),
        horizon=H,
        target_idx=TARGET_IDX,
        n_blocks=y["model"]["n_blocks"],
        ff_dim=y["model"]["ff_dim"],
        dropout=float(y["model"]["dropout"]),
        num_tickers=len(ticker_to_id),
        ticker_embed_dim=ticker_embed_dim,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(window).unsqueeze(0).to(device)
        tid = torch.tensor([int(ticker_to_id.get(args.ticker, 0))], device=device, dtype=torch.long)
        pred_scaled = model(xb, ticker_id=tid).cpu().numpy()[0]
    pred_ret = pred_scaled * scale + center

    # --- close prices ---
    close = _load_close_series(csv)
    # Align dates (feature index uses midnight-normalized dates)
    close.index = close.index.normalize()
    end_date = pd.Timestamp(dates[end]).normalize()
    if end_date not in close.index:
        raise SystemExit(f"No close price for {end_date.date()} in {csv}")
    last_close = float(close.loc[end_date])

    # History prices: use actual closes aligned to feature dates
    hist_close = close.reindex(pd.DatetimeIndex(hist_dates).normalize()).ffill().values

    # Anchor both forecast series at (end_date, last_close) so lines are continuous
    actual_dates_full = np.concatenate([[end_date.to_datetime64()], future_dates.values])
    pred_dates_full = actual_dates_full
    actual_prices = _reconstruct_prices(last_close, future_ret)
    pred_prices = _reconstruct_prices(last_close, pred_ret)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(hist_dates, hist_close, color="#4b4b4b", marker="o", ms=3, lw=1.4,
            label=f"History close ({hist_n}d)")
    ax.plot(actual_dates_full, actual_prices, color="#1f77b4", marker="o", lw=2,
            label="Actual (t+1..t+5)")
    ax.plot(pred_dates_full, pred_prices, color="#d62728", marker="s", lw=2, ls="--",
            label="Predicted (t+1..t+5)")
    ax.axvline(end_date, color="gray", ls=":", lw=1)
    ax.set_title(f"{args.ticker.upper()}  5-day price forecast @ {end_date.date()}  (last close = {last_close:.2f})")
    ax.set_ylabel("close price")
    ax.set_xlabel("date")
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    out = Path(args.out) if args.out else ROOT / y["output"]["ckpt_dir"] / f"forecast_{args.ticker}.png"
    fig.savefig(out, dpi=140)
    print(f"Saved plot -> {out}")
    print("Predicted log returns:", np.round(pred_ret, 5).tolist())
    print("Actual log returns   :", np.round(future_ret, 5).tolist())
    print("Predicted prices     :", np.round(pred_prices[1:], 3).tolist())
    print("Actual prices        :", np.round(actual_prices[1:], 3).tolist())


if __name__ == "__main__":
    main()
