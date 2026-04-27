"""
evaluate_gnn.py — Full evaluation with all metrics and long-short backtest.

Always prints: MSE, MAE, RMSE, MAPE, sMAPE (on both log-returns and close prices)
Training/early-stopping still uses val_mae_log_return (unchanged).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_gnn import build_dataloaders
from src.model_gnn import TemporalGNN
from src.utils_gnn import (
    DEFAULT_BEST_CHECKPOINT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_RESULTS_DIR,
    get_device,
    load_checkpoint,
    load_config,
    log_runtime_context,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Metric Computation
# ===========================================================================


def _mape(pred, actual):
    """Mean Absolute Percentage Error. Skips where actual ≈ 0."""
    mask = np.abs(actual) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((pred[mask] - actual[mask]) / actual[mask])) * 100


def _smape(pred, actual):
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(pred) + np.abs(actual)) / 2 + 1e-8
    return np.mean(np.abs(pred - actual) / denom) * 100


def compute_metrics(
    pred_lr: np.ndarray,
    target_lr: np.ndarray,
    pred_close: np.ndarray,
    target_close: np.ndarray,
    horizon: int = 5,
) -> Dict[str, float]:
    """Compute every metric — statistical, directional, IC, and financial."""
    m = {}
    errors_lr = pred_lr - target_lr
    errors_close = pred_close - target_close

    # ── Log-Return metrics (aggregate) ──
    m["mse_log_return"] = float(np.mean(errors_lr**2))
    m["mae_log_return"] = float(np.mean(np.abs(errors_lr)))
    m["rmse_log_return"] = float(np.sqrt(m["mse_log_return"]))
    m["mape_log_return"] = float(_mape(pred_lr, target_lr))
    m["smape_log_return"] = float(_smape(pred_lr, target_lr))

    # ── Close-Price metrics (aggregate) ──
    m["mse_close"] = float(np.mean(errors_close**2))
    m["mae_close"] = float(np.mean(np.abs(errors_close)))
    m["rmse_close"] = float(np.sqrt(m["mse_close"]))
    m["mape_close"] = float(_mape(pred_close, target_close))
    m["smape_close"] = float(_smape(pred_close, target_close))

    # Relative MAE (% of price)
    rel_err = np.abs(errors_close) / (np.abs(target_close) + 1e-8)
    m["relative_mae_close_pct"] = float(np.mean(rel_err) * 100)

    # ── Per-horizon breakdown ──
    for h in range(horizon):
        e = errors_lr[:, h]
        ec = errors_close[:, h]
        m[f"mse_h{h+1}"] = float(np.mean(e**2))
        m[f"mae_h{h+1}"] = float(np.mean(np.abs(e)))
        m[f"rmse_h{h+1}"] = float(np.sqrt(m[f"mse_h{h+1}"]))
        m[f"mape_h{h+1}"] = float(_mape(pred_lr[:, h], target_lr[:, h]))
        m[f"smape_h{h+1}"] = float(_smape(pred_lr[:, h], target_lr[:, h]))
        m[f"mse_close_h{h+1}"] = float(np.mean(ec**2))
        m[f"mae_close_h{h+1}"] = float(np.mean(np.abs(ec)))
        m[f"rmse_close_h{h+1}"] = float(np.sqrt(m[f"mse_close_h{h+1}"]))
        m[f"mape_close_h{h+1}"] = float(_mape(pred_close[:, h], target_close[:, h]))
        m[f"smape_close_h{h+1}"] = float(_smape(pred_close[:, h], target_close[:, h]))

    # ── Directional Accuracy ──
    correct = (np.sign(pred_lr) == np.sign(target_lr)).astype(float)
    m["directional_accuracy"] = float(np.mean(correct))
    for h in range(horizon):
        m[f"directional_acc_h{h+1}"] = float(
            np.mean((np.sign(pred_lr[:, h]) == np.sign(target_lr[:, h])).astype(float))
        )

    # ── Information Coefficient (IC) ──
    for h in range(horizon):
        p, t = pred_lr[:, h], target_lr[:, h]
        if len(p) > 1 and np.std(p) > 1e-10 and np.std(t) > 1e-10:
            m[f"ic_h{h+1}"] = float(np.corrcoef(p, t)[0, 1])
        else:
            m[f"ic_h{h+1}"] = 0.0
    m["ic_mean"] = float(np.mean([m[f"ic_h{h+1}"] for h in range(horizon)]))

    return m


def compute_ic_ir(predictions_by_date, horizon_idx=0):
    ics = []
    for _, (pred, target) in predictions_by_date.items():
        p, t = pred[:, horizon_idx], target[:, horizon_idx]
        if len(p) > 5 and np.std(p) > 1e-10 and np.std(t) > 1e-10:
            ics.append(float(np.corrcoef(p, t)[0, 1]))
    if len(ics) < 2:
        return 0.0
    return float(np.mean(ics) / (np.std(ics) + 1e-8))


# ===========================================================================
# Long-Short Backtest
# ===========================================================================


def run_long_short_backtest(predictions_by_date, decile=0.1, **kwargs):
    daily_returns, prev_long, prev_short, turnover_list = [], set(), set(), []
    for date_str in sorted(predictions_by_date.keys()):
        d = predictions_by_date[date_str]
        tickers, pred, actual = d["tickers"], d["pred_lr_h1"], d["actual_lr_h1"]
        if len(tickers) < 10:
            continue
        n = max(1, int(len(tickers) * decile))
        order = np.argsort(pred)
        long_i, short_i = order[-n:], order[:n]
        daily_returns.append(np.mean(actual[long_i]) - np.mean(actual[short_i]))
        ls, ss = set(np.array(tickers)[long_i]), set(np.array(tickers)[short_i])
        if prev_long:
            turnover_list.append(
                (
                    len(ls.symmetric_difference(prev_long))
                    + len(ss.symmetric_difference(prev_short))
                )
                / (4 * n)
            )
        prev_long, prev_short = ls, ss

    if not daily_returns:
        return {
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "avg_turnover": 0,
            "total_return": 0,
            "avg_daily_return": 0,
            "return_std": 0,
            "num_days": 0,
        }
    r = np.array(daily_returns)
    cum = np.cumsum(r)
    return {
        "sharpe_ratio": float(np.mean(r) / (np.std(r) + 1e-8) * np.sqrt(252)),
        "max_drawdown": float(np.min(cum - np.maximum.accumulate(cum))),
        "avg_turnover": float(np.mean(turnover_list)) if turnover_list else 0.0,
        "total_return": float(np.sum(r)),
        "avg_daily_return": float(np.mean(r)),
        "return_std": float(np.std(r)),
        "num_days": len(r),
    }


# ===========================================================================
# Print Helpers
# ===========================================================================


def _print_metrics_table(metrics, horizon=5):
    """Print a clean table of all core metrics."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("  LOG-RETURN METRICS")
    logger.info("-" * 80)
    logger.info(
        f"  {'':12s} {'MSE':>12s} {'MAE':>12s} {'RMSE':>12s} {'MAPE(%)':>12s} {'sMAPE(%)':>12s}"
    )
    logger.info(
        f"  {'Overall':12s} {metrics['mse_log_return']:12.6f} {metrics['mae_log_return']:12.6f} "
        f"{metrics['rmse_log_return']:12.6f} {metrics['mape_log_return']:12.2f} {metrics['smape_log_return']:12.2f}"
    )
    for h in range(horizon):
        tag = f"Day {h+1}"
        logger.info(
            f"  {tag:12s} {metrics[f'mse_h{h+1}']:12.6f} {metrics[f'mae_h{h+1}']:12.6f} "
            f"{metrics[f'rmse_h{h+1}']:12.6f} {metrics[f'mape_h{h+1}']:12.2f} {metrics[f'smape_h{h+1}']:12.2f}"
        )

    logger.info("")
    logger.info("  CLOSE-PRICE METRICS")
    logger.info("-" * 80)
    logger.info(
        f"  {'':12s} {'MSE':>12s} {'MAE':>12s} {'RMSE':>12s} {'MAPE(%)':>12s} {'sMAPE(%)':>12s}"
    )
    logger.info(
        f"  {'Overall':12s} {metrics['mse_close']:12.4f} {metrics['mae_close']:12.4f} "
        f"{metrics['rmse_close']:12.4f} {metrics['mape_close']:12.2f} {metrics['smape_close']:12.2f}"
    )
    for h in range(horizon):
        tag = f"Day {h+1}"
        logger.info(
            f"  {tag:12s} {metrics[f'mse_close_h{h+1}']:12.4f} {metrics[f'mae_close_h{h+1}']:12.4f} "
            f"{metrics[f'rmse_close_h{h+1}']:12.4f} {metrics[f'mape_close_h{h+1}']:12.2f} {metrics[f'smape_close_h{h+1}']:12.2f}"
        )

    logger.info("")
    logger.info(f"  Relative MAE (close): {metrics['relative_mae_close_pct']:.2f}%")

    logger.info("")
    logger.info("  DIRECTIONAL ACCURACY")
    logger.info("-" * 80)
    logger.info(f"  Overall: {metrics['directional_accuracy']:.2%}")
    for h in range(horizon):
        logger.info(f"  Day {h+1}:   {metrics[f'directional_acc_h{h+1}']:.2%}")

    logger.info("")
    logger.info("  INFORMATION COEFFICIENT")
    logger.info("-" * 80)
    logger.info(f"  IC (mean): {metrics['ic_mean']:.4f}")
    for h in range(horizon):
        logger.info(f"  Day {h+1}:    {metrics[f'ic_h{h+1}']:.4f}")
    logger.info(f"  ICIR (H1): {metrics.get('icir_h1', 0):.4f}")
    logger.info("=" * 80)


def assess_tiers(metrics):
    """FIX E10: tier assessment now judges a model on skill metrics that
    a zero-predictor cannot game (directional accuracy, IC, Sharpe), not
    on absolute MAE which rewards predicting ≈0.  Overall tier is the
    MIN of the three individual tiers — a model has to be good at all
    three dimensions to count.
    """
    tiers = {}
    da = metrics.get("directional_accuracy", 0.0)
    ic = metrics.get("ic_mean", 0.0)
    sharpe = metrics.get("ls_sharpe_ratio", 0.0)

    def _tier(val, t3, t2, t1):
        if val > t3:
            return 3
        if val > t2:
            return 2
        if val > t1:
            return 1
        return 0

    da_t = _tier(da, 0.57, 0.54, 0.51)
    ic_t = _tier(ic, 0.08, 0.05, 0.02)
    sh_t = _tier(sharpe, 1.5, 0.75, 0.25)
    mae = metrics.get("mae_log_return", float("inf"))
    rmse = metrics.get("rmse_log_return", float("inf"))

    def _fmt(t):
        return {
            3: "Tier 3 (Excellent)",
            2: "Tier 2 (Good)",
            1: "Tier 1 (Minimum Viable)",
            0: "Below Tier 1",
        }[t]

    tiers["Directional Acc"] = f"{_fmt(da_t)} ({da:.2%})"
    tiers["IC (mean)"] = f"{_fmt(ic_t)} ({ic:.4f})"
    tiers["Long-Short Sharpe"] = f"{_fmt(sh_t)} ({sharpe:.3f})"

    overall = min(da_t, ic_t, sh_t)
    tiers["OVERALL"] = f"{_fmt(overall)}"

    # Reference-only: raw MAE/RMSE with a note that they do NOT decide tier.
    tiers["MAE (log return, ref)"] = f"{mae:.5f}"
    tiers["RMSE (log return, ref)"] = f"{rmse:.5f}"
    return tiers


# ===========================================================================
# FIX E11: Zero-baseline comparison — the single most important sanity check.
# If our model cannot beat ``pred = 0`` on IC/directional accuracy/Sharpe,
# we have not learned anything useful regardless of how small the MAE is.
# ===========================================================================
def compute_zero_baseline(
    target_lr, target_close, last_close, predictions_by_date, horizon=5, eval_cfg=None
):
    """Metrics for the constant-zero predictor (pred_lr = 0 everywhere)."""
    pred_lr_zero = np.zeros_like(target_lr)

    # Reconstruct "prices" from zero log returns == last_close for every horizon.
    # last_close arrives per-sample concatenated in the same order as target_lr.
    # We don't have direct access here, so reconstruct from target_close:
    # if pred_lr is all zero then pred_close[:, h] = last_close which equals
    # target_close[:, h] / exp(cumsum(target_lr[:, :h+1])).
    with np.errstate(over="ignore", invalid="ignore"):
        inferred_last = target_close / np.exp(np.cumsum(target_lr, axis=1))
    pred_close_zero = np.repeat(inferred_last[:, :1], horizon, axis=1)

    m = compute_metrics(
        pred_lr_zero, target_lr, pred_close_zero, target_close, horizon=horizon
    )

    # Zero predictor has zero IC (no variance) and ~50% directional accuracy.
    # But we also need to compute the long-short backtest as if we had made
    # zero predictions — which means random deciles → expected return 0.
    zero_by_date = {
        d: {
            "tickers": v["tickers"],
            "pred_lr": np.zeros_like(v["pred_lr"]),
            "actual_lr": v["actual_lr"],
            "pred_lr_h1": np.zeros_like(v["pred_lr_h1"]),
            "actual_lr_h1": v["actual_lr_h1"],
        }
        for d, v in predictions_by_date.items()
    }
    decile = (eval_cfg or {}).get("long_short_decile", 0.1)
    ls = run_long_short_backtest(zero_by_date, decile=decile)
    for k, v in ls.items():
        m[f"ls_{k}"] = v
    return m


# ===========================================================================
# Full Evaluation Pipeline
# ===========================================================================


@torch.no_grad()
def evaluate(config, checkpoint_path, split="test"):
    set_seed(config.get("seed", 42))
    device = get_device()
    logger.info(
        "Evaluation request | split=%s | checkpoint=%s",
        split,
        os.path.abspath(checkpoint_path),
    )

    train_loader, val_loader, test_loader, metadata = build_dataloaders(config)
    loader = test_loader if split == "test" else val_loader
    logger.info(
        "Evaluation dataloader | selected_split=%s | num_batches=%d | num_tickers=%d",
        split,
        len(loader.dataset),
        metadata.get("num_tickers", 0),
    )

    model = TemporalGNN(config, max_nodes=metadata.get("max_nodes", 550)).to(device)
    if metadata.get("fundamentals"):
        model.reports_encoder.compute_normalization_stats(metadata["fundamentals"])
    load_checkpoint(checkpoint_path, model)
    model.eval()

    all_pred_lr, all_target_lr = [], []
    all_pred_close, all_target_close = [], []
    predictions_by_date = {}

    logger.info(f"Evaluating on {split} set...")

    skipped_empty = 0
    for batch_idx, sample in enumerate(loader, 1):
        if isinstance(sample, list):
            sample = sample[0]
        if sample["num_active"] == 0:
            skipped_empty += 1
            continue

        output = model(sample, device=device)
        pred_lr = output["log_returns"].cpu().numpy()
        pred_close = output["pred_close"].cpu().numpy()
        targets = sample["targets"].numpy()
        target_close = sample["target_close"].numpy()

        all_pred_lr.append(pred_lr)
        all_target_lr.append(targets)
        all_pred_close.append(pred_close)
        all_target_close.append(target_close)

        date_str = sample["pred_date"]
        tickers = sample["target_tickers"]
        if date_str:
            predictions_by_date[date_str] = {
                "tickers": tickers,
                "pred_lr": pred_lr,
                "actual_lr": targets,
                "pred_lr_h1": pred_lr[:, 0],
                "actual_lr_h1": targets[:, 0],
            }
        if (
            batch_idx % max(1, config.get("logging", {}).get("log_every_n_steps", 10))
            == 0
        ):
            logger.debug(
                "Processed eval batch %d | pred_date=%s | active=%d | targets=%d",
                batch_idx,
                date_str,
                sample["num_active"],
                len(tickers),
            )

    if not all_pred_lr:
        logger.error("No valid predictions collected")
        return {}

    pred_lr = np.concatenate(all_pred_lr)
    target_lr = np.concatenate(all_target_lr)
    pred_close = np.concatenate(all_pred_close)
    target_close = np.concatenate(all_target_close)

    metrics = compute_metrics(pred_lr, target_lr, pred_close, target_close)
    metrics["icir_h1"] = compute_ic_ir(
        {d: (v["pred_lr"], v["actual_lr"]) for d, v in predictions_by_date.items()}
    )

    eval_cfg = config.get("evaluation", {})
    ls = run_long_short_backtest(
        predictions_by_date, decile=eval_cfg.get("long_short_decile", 0.1)
    )
    for k, v in ls.items():
        metrics[f"ls_{k}"] = v

    # ── FIX E11: Zero-baseline comparison ──
    zero_metrics = compute_zero_baseline(
        target_lr=target_lr,
        target_close=target_close,
        last_close=None,
        predictions_by_date=predictions_by_date,
        horizon=5,
        eval_cfg=eval_cfg,
    )

    # ── Print everything ──
    logger.info(f"\nSamples: {len(pred_lr)} | Dates: {len(predictions_by_date)}")
    logger.info("Evaluation bookkeeping | skipped_empty=%d", skipped_empty)
    _print_metrics_table(metrics)

    logger.info("\n  LONG-SHORT PORTFOLIO")
    logger.info("-" * 80)
    logger.info(
        f"  Sharpe: {ls['sharpe_ratio']:.4f} | MaxDD: {ls['max_drawdown']:.4f} | "
        f"Turnover: {ls['avg_turnover']:.2%} | Return: {ls['total_return']:.4f}"
    )

    logger.info("\n  ZERO-PREDICTOR BASELINE (FIX E11)")
    logger.info("-" * 80)
    logger.info(
        "  MAE=%.6f | RMSE=%.6f | DirAcc=%.2f%% | IC=%.4f | LS Sharpe=%.3f",
        zero_metrics["mae_log_return"],
        zero_metrics["rmse_log_return"],
        zero_metrics["directional_accuracy"] * 100.0,
        zero_metrics["ic_mean"],
        zero_metrics.get("ls_sharpe_ratio", 0.0),
    )
    # Verdict line: highlights whether the model beats the zero baseline.
    beats_dir = metrics["directional_accuracy"] > zero_metrics["directional_accuracy"]
    beats_ic = metrics["ic_mean"] > zero_metrics["ic_mean"]
    beats_sharpe = metrics.get("ls_sharpe_ratio", 0.0) > zero_metrics.get(
        "ls_sharpe_ratio", 0.0
    )
    logger.info(
        "  Beats zero? dir=%s | ic=%s | sharpe=%s",
        "YES" if beats_dir else "NO",
        "YES" if beats_ic else "NO",
        "YES" if beats_sharpe else "NO",
    )

    logger.info("\n  TIER ASSESSMENT")
    logger.info("-" * 80)
    for k, v in assess_tiers(metrics).items():
        logger.info(f"  {k}: {v}")

    # ── Save ──
    os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)
    output_path = os.path.join(DEFAULT_RESULTS_DIR, f"metrics_{split}.json")
    with open(output_path, "w") as f:
        json.dump(
            {"model": metrics, "zero_baseline": zero_metrics}, f, indent=2, default=str
        )
    logger.info("\nSaved to %s", output_path)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=DEFAULT_BEST_CHECKPOINT)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    config = load_config(args.config)
    log_path = setup_logging(
        config, command_name="evaluate", config_path=args.config, args=args
    )
    logger.info("Loaded config from %s", os.path.abspath(args.config))
    log_runtime_context("evaluate", config, extra={"evaluation_log_path": log_path})
    evaluate(config, args.checkpoint, args.split)


if __name__ == "__main__":
    main()
