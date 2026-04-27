"""Inference: load trained LSTM checkpoint and forecast from one entrypoint.

Usage:
    python -m scripts.forecast --ticker aal
    python -m scripts.forecast --ticker aal --history-days 30
    python -m scripts.forecast --csv data/sp500_time_series/aa.csv --checkpoint artifacts/model.pt --latest

The compare mode shifts the context window back H days so the final H rows of
the CSV serve as actual future values for comparison with the model forecast.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_ohlcv
from src.models import LSTMForecaster
from src.preprocessing import build_features
from src.utils import load_config

PREDICT = ROOT / "predict"
PLOTS = PREDICT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", help="Ticker symbol (CSV stem, e.g. 'aal')")
    ap.add_argument("--csv", type=Path, default=None, help="Direct CSV path override")
    ap.add_argument(
        "--history-days",
        type=int,
        default=30,
        help="Days of history to show before the forecast window (default 30)",
    )
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    ap.add_argument("--ckpt", "--checkpoint", dest="ckpt", type=Path, default=None, help="Override checkpoint path")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional override for config data directory",
    )
    ap.add_argument("--date-col", default=None, help="Override date column name")
    ap.add_argument(
        "--latest",
        action="store_true",
        help="Use the latest available lookback window and print predictions only",
    )
    return ap.parse_args(argv)


def load_artifacts(cfg_path: Path, ckpt_path: Path | None) -> tuple[dict, dict]:
    cfg = load_config(cfg_path)
    resolved_ckpt = ROOT / cfg.get("artifacts", {}).get("checkpoint", "artifacts/model.pt")
    if ckpt_path is not None:
        resolved_ckpt = ckpt_path
    if not resolved_ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {resolved_ckpt}. "
            "Train first (`python -m scripts.train`) or pass --ckpt."
        )
    ckpt = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
    return cfg, ckpt


def resolve_data_dir(cfg: dict, override: Path | None) -> Path:
    if override is not None:
        return override
    csv_cfg = cfg["data"]["csv"]
    return (ROOT / csv_cfg).parent


def resolve_csv_path(ticker: str | None, data_dir: Path, csv_path: Path | None) -> Path:
    if csv_path is not None:
        return csv_path
    if not ticker:
        raise ValueError("Provide either --ticker or --csv")
    return data_dir / f"{ticker.lower()}.csv"


def load_compare_window(
    csv_path: Path,
    feature_names: list[str],
    lookback: int,
    horizon: int,
    history_days: int,
    date_col: str | None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    raw = load_ohlcv(csv_path, date_col=date_col)
    feats = build_features(raw)

    missing = [name for name in feature_names if name not in feats.columns]
    if missing:
        raise ValueError(f"CSV features missing columns expected by checkpoint: {missing}")

    feats = feats[feature_names]

    raw_aligned = raw.loc[feats.index].copy()
    raw_aligned = raw_aligned.rename_axis("date").reset_index()

    need = lookback + horizon
    if len(feats) < need:
        raise ValueError(f"Need >= {need} rows after feature engineering, got {len(feats)}")

    ctx_end = len(feats) - horizon
    ctx_start = ctx_end - lookback

    window_feats = feats.to_numpy()[ctx_start:ctx_end]
    actual_future = raw_aligned.iloc[ctx_end : ctx_end + horizon]

    hist_start = max(ctx_end - history_days, 0)
    history = raw_aligned.iloc[hist_start:ctx_end]

    return window_feats, history, actual_future


def load_latest_window(
    csv_path: Path,
    feature_names: list[str],
    lookback: int,
    date_col: str | None,
) -> tuple[np.ndarray, pd.Timestamp]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    raw = load_ohlcv(csv_path, date_col=date_col)
    feats = build_features(raw)

    missing = [name for name in feature_names if name not in feats.columns]
    if missing:
        raise ValueError(f"CSV features missing columns expected by checkpoint: {missing}")

    feats = feats[feature_names]
    if len(feats) < lookback:
        raise ValueError(f"Need >= {lookback} rows after feature engineering, got {len(feats)}")

    window_feats = feats.to_numpy()[-lookback:]
    anchor_date = pd.to_datetime(feats.index[-1]).tz_localize(None)
    return window_feats, anchor_date


@torch.no_grad()
def run_forecast(model: LSTMForecaster, window_feats: np.ndarray, feat_scaler, target_scaler) -> np.ndarray:
    x = feat_scaler.transform(window_feats).astype(np.float32)
    pred_scaled = model(torch.from_numpy(x).unsqueeze(0)).squeeze(0).cpu().numpy()
    return target_scaler.inverse_transform(pred_scaled.reshape(1, -1)).ravel()


def log_rets_to_prices(base: float, log_rets: np.ndarray) -> np.ndarray:
    return base * np.exp(np.cumsum(log_rets))


def print_latest_table(anchor_date: pd.Timestamp, pred_log_rets: np.ndarray) -> None:
    out = pd.DataFrame({
        "step": [f"day+{i + 1}" for i in range(len(pred_log_rets))],
        "log_return": pred_log_rets,
        "implied_return_%": (np.exp(pred_log_rets) - 1.0) * 100.0,
    })
    print(f"anchor date: {anchor_date}")
    print(out.to_string(index=False))


def load_rmse_per_step(metrics_path: Path, horizon: int) -> list[float] | None:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    rmse_col = "RMSE_original" if "RMSE_original" in df.columns else "RMSE"
    if rmse_col not in df.columns:
        return None
    vals = [float(v) for v in df[rmse_col].tolist()]
    return vals[:horizon] if vals else None


def plot(
    ticker: str,
    history: pd.DataFrame,
    actual_future: pd.DataFrame,
    pred_prices: np.ndarray,
    rmse_per_step: list[float] | None,
) -> Path:
    hist_dates = pd.to_datetime(history["date"]).dt.tz_localize(None)
    hist_prices = history["Close"].to_numpy(dtype=float)
    last_close = hist_prices[-1]

    act_dates = pd.to_datetime(actual_future["date"]).dt.tz_localize(None)
    act_close_raw = actual_future["Close"].to_numpy(dtype=float)
    prev_closes = np.concatenate([[last_close], act_close_raw[:-1]])
    act_prices = last_close * np.exp(np.cumsum(np.log(act_close_raw / prev_closes)))

    pivot_date = hist_dates.iloc[-1]
    pivot_price = last_close

    all_act_dates = [pivot_date] + list(act_dates)
    all_act_prices = np.concatenate([[pivot_price], act_prices])
    all_pred_dates = [pivot_date] + list(act_dates)
    all_pred_prices = np.concatenate([[pivot_price], pred_prices])

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(hist_dates, hist_prices, color="#4C72B0", linewidth=2.0, label="History")

    ax.plot(
        all_act_dates,
        all_act_prices,
        color="#2ca02c",
        linewidth=2.0,
        marker="o",
        markersize=5,
        label="Actual",
    )

    ax.plot(
        all_pred_dates,
        all_pred_prices,
        color="#DD8452",
        linewidth=2.0,
        marker="s",
        markersize=5,
        label="Predicted",
    )

    if rmse_per_step:
        band = np.array(rmse_per_step[: len(pred_prices)], dtype=float)
        upper = [pivot_price] + [p * np.exp(r) for p, r in zip(pred_prices, band)]
        lower = [pivot_price] + [p * np.exp(-r) for p, r in zip(pred_prices, band)]
        ax.fill_between(all_pred_dates, lower, upper, alpha=0.12, color="#DD8452", label="+-1 RMSE")

    ax.axvline(pivot_date, color="gray", linestyle=":", linewidth=1.2)
    ax.text(
        pivot_date,
        0.02,
        " forecast start",
        color="gray",
        fontsize=8,
        va="bottom",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_title(f"{ticker.upper()} - {len(hist_prices)}-Day History + {len(act_prices)}-Day Forecast")
    ax.set_ylabel("Price ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()

    out = PLOTS / f"forecast_{ticker.lower()}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def build_model_from_ckpt(ckpt: dict) -> LSTMForecaster:
    bound = ckpt["model_cfg"].get("output_bound")
    model = LSTMForecaster(
        n_features=int(ckpt["n_features"]),
        horizon=int(ckpt["horizon"]),
        hidden1=int(ckpt["model_cfg"]["hidden1"]),
        hidden2=int(ckpt["model_cfg"]["hidden2"]),
        dropout=float(ckpt["model_cfg"]["dropout"]),
        output_bound=None if bound is None else float(bound),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg, ckpt = load_artifacts(args.config, args.ckpt)
    data_dir = resolve_data_dir(cfg, args.data_dir)

    lookback = int(ckpt["lookback"])
    horizon = int(ckpt["horizon"])
    date_col = args.date_col if args.date_col is not None else cfg["data"].get("date_col", "Date")
    csv_path = resolve_csv_path(args.ticker, data_dir, args.csv)

    model = build_model_from_ckpt(ckpt)
    if args.latest:
        window_feats, anchor_date = load_latest_window(
            csv_path=csv_path,
            feature_names=list(ckpt["feature_names"]),
            lookback=lookback,
            date_col=date_col,
        )
        pred_log_rets = run_forecast(
            model,
            window_feats,
            feat_scaler=ckpt["feat_scaler"],
            target_scaler=ckpt["target_scaler"],
        )
        print_latest_table(anchor_date, pred_log_rets)
        return

    label = args.ticker.upper() if args.ticker else csv_path.stem.upper()
    print(f"Loading series: {label}")
    window_feats, history, actual_future = load_compare_window(
        csv_path=csv_path,
        feature_names=list(ckpt["feature_names"]),
        lookback=lookback,
        horizon=horizon,
        history_days=args.history_days,
        date_col=date_col,
    )

    pred_log_rets = run_forecast(
        model,
        window_feats,
        feat_scaler=ckpt["feat_scaler"],
        target_scaler=ckpt["target_scaler"],
    )
    base_price = float(history["Close"].iloc[-1])
    pred_prices = log_rets_to_prices(base_price, pred_log_rets)

    metrics_cfg = cfg.get("artifacts", {}).get("metrics", "artifacts/test_metrics.csv")
    rmse_per_step = load_rmse_per_step(ROOT / metrics_cfg, horizon)

    out = plot(label, history, actual_future, pred_prices, rmse_per_step)
    print(f"Saved -> {out.relative_to(ROOT)}")

    act_prices = actual_future["Close"].to_numpy(dtype=float)
    print(f"\n{'':8} {'Pred log-ret':>12} {'Pred $':>10} {'Actual $':>10} {'Error':>8}")
    print("-" * 52)
    for i, (lr, pp, ap) in enumerate(zip(pred_log_rets, pred_prices, act_prices)):
        print(f"h+{i+1:<5}  {lr*100:>+10.3f}%  ${pp:>8.2f}  ${ap:>8.2f}  {pp-ap:>+7.2f}")


if __name__ == "__main__":
    main()
