"""Forecast visualization for individual stocks.

Generates plots showing historical context, ground-truth horizon, median
forecast, and prediction intervals.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

QUANTILE_BAND_COLORS = [
    "#A0CBE8",
    "#FFBE7D",
    "#8CD17D",
    "#FF9D9A",
]


def _select_interval_pairs(quantile_levels: List[float]) -> List[Tuple[int, int]]:
    """Pick symmetric forecast intervals from outermost to innermost."""
    q_levels = np.asarray(quantile_levels, dtype=np.float64)
    paired: List[Tuple[int, int]] = []
    for low_idx in range(len(q_levels)):
        high_idx = len(q_levels) - 1 - low_idx
        if low_idx >= high_idx:
            break
        paired.append((low_idx, high_idx))
    return paired


def plot_stock_forecast(
    ticker: str,
    context_values: np.ndarray,
    ground_truth: np.ndarray,
    median_forecast: np.ndarray,
    quantile_forecast: np.ndarray,
    quantile_levels: List[float],
    output_dir: str,
    context_days_shown: int = 90,
    context_timestamps: Optional[np.ndarray] = None,
    forecast_timestamps: Optional[np.ndarray] = None,
) -> Path:
    """Plot a backtest-style forecast vs actual chart with a shared pivot.

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
        context_timestamps: Optional timestamps aligned to *context_values*.
        forecast_timestamps: Optional timestamps aligned to *ground_truth*.

    Returns:
        Path to the saved PNG file.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    horizon = len(ground_truth)
    ctx_shown = min(context_days_shown, len(context_values))
    ctx_tail = np.asarray(context_values[-ctx_shown:], dtype=np.float64)
    pivot_price = float(ctx_tail[-1])

    actual_values = np.concatenate(
        [[pivot_price], np.asarray(ground_truth, dtype=np.float64)]
    )
    predicted_values = np.concatenate(
        [[pivot_price], np.asarray(median_forecast, dtype=np.float64)]
    )

    interval_pairs = _select_interval_pairs(quantile_levels)

    use_dates = context_timestamps is not None and forecast_timestamps is not None
    if use_dates:
        history_x = pd.DatetimeIndex(
            pd.to_datetime(context_timestamps[-ctx_shown:])
        ).tz_localize(None)
        forecast_x = pd.DatetimeIndex(pd.to_datetime(forecast_timestamps)).tz_localize(
            None
        )
        pivot_x = history_x[-1]
        future_x = pd.DatetimeIndex([pivot_x, *forecast_x])
    else:
        history_x = np.arange(-ctx_shown + 1, 1)
        pivot_x = 0
        future_x = np.arange(0, horizon + 1)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(history_x, ctx_tail, color="#4C72B0", linewidth=2.0, label="History")
    ax.plot(
        future_x,
        actual_values,
        color="#2ca02c",
        linewidth=2.0,
        marker="o",
        markersize=5,
        label="Actual",
    )
    ax.plot(
        future_x,
        predicted_values,
        color="#DD8452",
        linewidth=2.0,
        marker="s",
        markersize=5,
        label="Predicted",
    )

    alphas = [0.12, 0.18, 0.24, 0.30]
    for band_idx, (lo_idx, hi_idx) in enumerate(interval_pairs):
        lower_band = np.concatenate(
            [[pivot_price], np.asarray(quantile_forecast[lo_idx], dtype=np.float64)]
        )
        upper_band = np.concatenate(
            [[pivot_price], np.asarray(quantile_forecast[hi_idx], dtype=np.float64)]
        )
        ax.fill_between(
            future_x,
            lower_band,
            upper_band,
            alpha=alphas[min(band_idx, len(alphas) - 1)],
            color=QUANTILE_BAND_COLORS[band_idx % len(QUANTILE_BAND_COLORS)],
            label=(
                f"Q{int(round(quantile_levels[lo_idx] * 100))}-"
                f"Q{int(round(quantile_levels[hi_idx] * 100))}"
            ),
        )

    ax.axvline(x=pivot_x, color="gray", linestyle=":", linewidth=1.2)
    ax.text(
        pivot_x,
        0.02,
        " forecast start",
        color="gray",
        fontsize=8,
        va="bottom",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_title(
        f"{ticker.upper()} - {len(ctx_tail)}-Step History + {horizon}-Step Forecast",
        fontsize=14,
        fontweight="bold",
    )
    if use_dates:
        ax.set_xlabel("Date")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
    else:
        ax.set_xlabel("Trading days (relative to forecast start)")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="best", fontsize=9)
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
    ax.set_title(
        "Aggregate Metrics Across All Evaluated Stocks", fontsize=13, fontweight="bold"
    )
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
        ax.axvline(
            np.mean(values),
            color="#e74c3c",
            linestyle="--",
            label=f"Mean: {np.mean(values):.3f}",
        )
        ax.axvline(
            np.median(values),
            color="#2ecc71",
            linestyle="--",
            label=f"Median: {np.median(values):.3f}",
        )
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
