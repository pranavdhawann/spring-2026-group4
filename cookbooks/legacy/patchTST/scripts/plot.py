"""Visualize PatchTST training and test results.

Reads predict/results.json  (training history + metrics)
      predict/preds.npz     (preds, targets arrays)

Produces 4 figures saved to predict/plots/:
  1. loss_curve.png       — train & val Huber loss over steps
  2. metrics_per_step.png — MAE / RMSE / Dir-Acc per horizon step
  3. scatter_h1.png       — predicted vs actual log-ret (h=1)
  4. equity_curve.png     — long-short strategy cumulative return
                            (go long if pred>0, short if pred<0, unit sizing)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREDICT = ROOT / "predict"
PLOTS = PREDICT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def load_data():
    results_path = PREDICT / "results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        npz = np.load(PREDICT / "preds.npz")
        return results, npz["preds"], npz["targets"]
    history_path = PREDICT / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        print("  (training in progress — showing loss curve only)")
        return {"train": {"history": history}}, None, None
    raise FileNotFoundError(f"No results found in {PREDICT}; run scripts/run.py first.")


# ── 1. Loss curve ──────────────────────────────────────────────────────────────
def plot_loss_curve(history: list) -> None:
    steps = [h["step"] for h in history]
    train_l = [h["train_loss"] for h in history]
    val_l = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, train_l, label="Train Huber", linewidth=1.6)
    ax.plot(steps, val_l, label="Val Huber", linewidth=1.6, linestyle="--")
    best_step = steps[int(np.argmin(val_l))]
    ax.axvline(best_step, color="gray", linestyle=":", linewidth=1, label=f"Best val (step {best_step})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Huber Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "loss_curve.png")
    plt.close(fig)
    print("  loss_curve.png")


# ── 2. Per-horizon metrics bar chart ───────────────────────────────────────────
def plot_metrics_per_step(metrics: dict) -> None:
    ps = metrics["per_step"]
    H = len(ps["mae"])
    horizons = [f"h+{i+1}" for i in range(H)]
    x = np.arange(H)
    width = 0.25

    mae = ps["mae"]
    rmse = ps["rmse"]
    dir_acc = [v / 100 for v in ps["directional_acc_pct"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(x - width/2, mae, width, label="MAE", color="#4C72B0")
    ax1.bar(x + width/2, rmse, width, label="RMSE", color="#DD8452")
    ax1.set_xticks(x); ax1.set_xticklabels(horizons)
    ax1.set_ylabel("Log-return error")
    ax1.set_title("MAE & RMSE per Horizon Step")
    ax1.legend()

    ax2.bar(x, dir_acc, color="#55A868")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (50%)")
    ax2.set_xticks(x); ax2.set_xticklabels(horizons)
    ax2.set_ylabel("Directional accuracy")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.set_title("Directional Accuracy per Horizon Step")
    ax2.set_ylim(0.35, 0.65)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(PLOTS / "metrics_per_step.png")
    plt.close(fig)
    print("  metrics_per_step.png")


# ── 3. Predicted vs actual scatter (h=1) ──────────────────────────────────────
def plot_scatter(preds: np.ndarray, targets: np.ndarray) -> None:
    p, t = preds[:, 0], targets[:, 0]
    # clip extreme outliers for readability
    lim = np.percentile(np.abs(t), 99) * 1.5

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, s=3, alpha=0.25, rasterized=True, color="#4C72B0")
    bound = lim
    ax.plot([-bound, bound], [-bound, bound], "r--", linewidth=1, label="Perfect forecast")
    ax.set_xlim(-bound, bound); ax.set_ylim(-bound, bound)
    ax.set_xlabel("Actual log-return (h+1)")
    ax.set_ylabel("Predicted log-return (h+1)")
    ax.set_title("Predicted vs Actual (h+1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "scatter_h1.png")
    plt.close(fig)
    print("  scatter_h1.png")


# ── 4. Long-short equity curve ─────────────────────────────────────────────────
def plot_equity_curve(preds: np.ndarray, targets: np.ndarray) -> None:
    """Unit-sized long-short: sign(pred_h1) × actual_h1."""
    signal = np.sign(preds[:, 0])          # +1 long / -1 short
    strategy_ret = signal * targets[:, 0]  # realised return
    buy_hold_ret = targets[:, 0]           # passive long

    cum_strat = np.cumprod(1 + np.clip(strategy_ret, -0.5, 0.5))
    cum_bh = np.cumprod(1 + np.clip(buy_hold_ret, -0.5, 0.5))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(cum_strat, label="Long-short (unit)", linewidth=1.6)
    ax.plot(cum_bh, label="Buy & hold", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Test window index")
    ax.set_ylabel("Cumulative return (×1)")
    ax.set_title("Long-Short Equity Curve (h+1, unit sizing) — test set")
    ax.legend()

    sharpe = float(strategy_ret.mean() / (strategy_ret.std() + 1e-12) * np.sqrt(252))
    ax.text(0.02, 0.05, f"Ann. Sharpe ~ {sharpe:.2f}",
            transform=ax.transAxes, fontsize=10, color="steelblue")

    fig.tight_layout()
    fig.savefig(PLOTS / "equity_curve.png")
    plt.close(fig)
    print(f"  equity_curve.png  (Ann. Sharpe ~ {sharpe:.2f})")


def main() -> None:
    print("Loading results...")
    results, preds, targets = load_data()

    history = results["train"]["history"]

    print("Plotting...")
    plot_loss_curve(history)

    if preds is not None:
        metrics = results["test_metrics"]
        agg = metrics["aggregate"]
        print(f"  Test | MAE={agg['mae']:.5f}  RMSE={agg['rmse']:.5f}  "
              f"DirAcc={agg['directional_acc_pct']:.1f}%  "
              f"PredSharpe={agg['sharpe_pred']:.3f}")
        print(f"  Test windows: {len(preds)}")
        plot_metrics_per_step(metrics)
        plot_scatter(preds, targets)
        plot_equity_curve(preds, targets)

    print(f"\nAll plots saved to predict/plots/")


if __name__ == "__main__":
    main()
