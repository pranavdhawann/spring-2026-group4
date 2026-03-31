"""Forecast visualization for individual stocks.

Generates plots showing historical context, ground-truth horizon, median
forecast, and prediction intervals.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def plot_stock_forecast(
    ticker: str,
    context_values: np.ndarray,
    ground_truth: np.ndarray,
    median_forecast: np.ndarray,
    quantile_forecast: np.ndarray,
    quantile_levels: List[float],
    output_dir: str,
    context_days_shown: int = 90,
) -> Path:
    """Plot a single stock's forecast vs actuals with prediction intervals.

    Args:
        ticker: Stock ticker name (used in title and filename).
        context_values: 1-D array of historical context observations.
        ground_truth: 1-D array of actual future values (horizon).
        median_forecast: 1-D array of median (p50) forecast.
        quantile_forecast: Shape ``(Q, H)`` — quantile forecasts.
        quantile_levels: Quantile levels corresponding to rows of
            *quantile_forecast*.
        output_dir: Directory to save the plot.
        context_days_shown: Number of trailing context days to display.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    horizon = len(ground_truth)
    ctx_shown = min(context_days_shown, len(context_values))
    ctx_tail = context_values[-ctx_shown:]

    # X-axis: context days (negative) + horizon days (positive)
    ctx_x = np.arange(-ctx_shown, 0)
    horizon_x = np.arange(0, horizon)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical context
    ax.plot(ctx_x, ctx_tail, color="#2c3e50", linewidth=1.2, label="History")

    # Ground truth (horizon)
    ax.plot(horizon_x, ground_truth, color="#2c3e50", linewidth=1.2,
            linestyle="--", label="Actual")

    # Median forecast
    ax.plot(horizon_x, median_forecast, color="#e74c3c", linewidth=1.8,
            label="Median forecast")

    # Prediction intervals — shade from outer to inner
    # Pair quantiles symmetrically: (0.1, 0.9), (0.2, 0.8), (0.3, 0.7)
    q_levels = np.array(quantile_levels)
    alphas = [0.12, 0.18, 0.25, 0.35]
    paired = []
    for i in range(len(q_levels)):
        j = len(q_levels) - 1 - i
        if i >= j:
            break
        paired.append((i, j))

    for idx, (lo_idx, hi_idx) in enumerate(paired):
        lo_q = q_levels[lo_idx]
        hi_q = q_levels[hi_idx]
        alpha = alphas[min(idx, len(alphas) - 1)]
        ax.fill_between(
            horizon_x,
            quantile_forecast[lo_idx],
            quantile_forecast[hi_idx],
            alpha=alpha,
            color="#3498db",
            label=f"p{int(lo_q*100)}–p{int(hi_q*100)}" if idx < 3 else None,
        )

    # Vertical line at forecast start
    ax.axvline(x=0, color="#7f8c8d", linestyle=":", linewidth=0.8, alpha=0.7)

    ax.set_title(f"{ticker} — Chronos 2 Zero-Shot Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Trading days (relative to forecast start)")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(output_dir) / f"{ticker}_forecast_chronos_2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved plot for %s → %s", ticker, out_path)
    return out_path


def plot_metrics_summary(
    per_stock_metrics: dict,
    output_dir: str,
) -> Path:
    """Plot a bar chart summarizing aggregate metrics across all stocks.

    Args:
        per_stock_metrics: Dict mapping ``ticker -> {metric: value}``.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    # Aggregate per metric
    all_metrics: dict = {}
    for ticker, m in per_stock_metrics.items():
        for metric_name, val in m.items():
            if not np.isnan(val):
                all_metrics.setdefault(metric_name, []).append(val)

    metric_names = list(all_metrics.keys())
    means = [np.mean(all_metrics[m]) for m in metric_names]
    medians = [np.median(all_metrics[m]) for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, means, width, label="Mean", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, medians, width, label="Median", color="#e74c3c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_names])
    ax.set_title("Aggregate Metrics Across All Evaluated Stocks", fontsize=13, fontweight="bold")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = Path(output_dir) / "metrics_summary_chronos_2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved metrics summary → %s", out_path)
    return out_path


def plot_metric_distribution(
    per_stock_metrics: dict,
    output_dir: str,
) -> Path:
    """Plot histograms of each metric's distribution across stocks.

    Args:
        per_stock_metrics: Dict mapping ``ticker -> {metric: value}``.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib.pyplot as plt

    all_metrics: dict = {}
    for ticker, m in per_stock_metrics.items():
        for metric_name, val in m.items():
            if not np.isnan(val):
                all_metrics.setdefault(metric_name, []).append(val)

    n_metrics = len(all_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, all_metrics.items()):
        ax.hist(values, bins=30, color="#3498db", alpha=0.7, edgecolor="white")
        ax.axvline(np.mean(values), color="#e74c3c", linestyle="--", label=f"Mean: {np.mean(values):.3f}")
        ax.axvline(np.median(values), color="#2ecc71", linestyle="--", label=f"Median: {np.median(values):.3f}")
        ax.set_title(name.upper(), fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Metric Distributions Across Stocks", fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path = Path(output_dir) / "metric_distributions_chronos_2.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved metric distributions → %s", out_path)
    return out_path
