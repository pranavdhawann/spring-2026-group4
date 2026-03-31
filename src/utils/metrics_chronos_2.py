"""Evaluation metrics for time series forecasting.

All public functions accept numpy arrays and return plain floats or dicts.
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def mse(
    ground_truth: np.ndarray,
    forecast: np.ndarray,
) -> np.ndarray:
    """Mean Squared Error per series.

    Args:
        ground_truth: Shape ``(N, H)`` — actual values.
        forecast: Shape ``(N, H)`` — point (median) forecasts.

    Returns:
        1-D array of length ``N`` with the MSE for each series.
    """
    return np.mean((forecast - ground_truth) ** 2, axis=1)


def mae(
    ground_truth: np.ndarray,
    forecast: np.ndarray,
) -> np.ndarray:
    """Mean Absolute Error per series.

    Args:
        ground_truth: Shape ``(N, H)``.
        forecast: Shape ``(N, H)``.

    Returns:
        1-D array of length ``N``.
    """
    return np.mean(np.abs(forecast - ground_truth), axis=1)


def mase(
    ground_truth: np.ndarray,
    forecast: np.ndarray,
    seasonal_errors: np.ndarray,
) -> np.ndarray:
    """Mean Absolute Scaled Error per series.

    Args:
        ground_truth: Shape ``(N, H)``.
        forecast: Shape ``(N, H)``.
        seasonal_errors: Shape ``(N,)`` — precomputed seasonal errors from
            the context window. NaN entries are propagated (excluded from
            the aggregate).

    Returns:
        1-D array of length ``N``. NaN where seasonal error was NaN.
    """
    mae_values = np.mean(np.abs(forecast - ground_truth), axis=1)
    return mae_values / seasonal_errors


def smape(
    ground_truth: np.ndarray,
    forecast: np.ndarray,
) -> np.ndarray:
    """Symmetric Mean Absolute Percentage Error per series.

    Uses the formula ``200 * mean(|f - a| / (|f| + |a|))``, returning
    values in the range ``[0, 200]``.

    Args:
        ground_truth: Shape ``(N, H)``.
        forecast: Shape ``(N, H)``.

    Returns:
        1-D array of length ``N`` (percentage scale, 0–200).
    """
    numerator = np.abs(forecast - ground_truth)
    denominator = np.abs(forecast) + np.abs(ground_truth)
    # Avoid 0/0 — treat as 0 error when both are zero
    ratio = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
    )
    return 200.0 * np.mean(ratio, axis=1)


def crps_ensemble(
    ground_truth: np.ndarray,
    samples: np.ndarray,
) -> np.ndarray:
    """Continuous Ranked Probability Score (energy form) per series.

    Uses the identity ``CRPS = E|X - y| - 0.5 * E|X - X'|`` where
    ``X, X'`` are independent draws from the forecast distribution ``F``.
    The spread term is computed efficiently via sorted samples.

    Args:
        ground_truth: Shape ``(N, H)`` — actual values.
        samples: Shape ``(N, S, H)`` — sampled forecast trajectories.

    Returns:
        1-D array of length ``N`` — CRPS averaged over the forecast horizon.
    """
    num_samples = samples.shape[1]

    # Term 1: E|X - y|  →  (1/S) * sum_i |x_i - y|
    abs_errors = np.abs(
        samples - ground_truth[:, np.newaxis, :]
    )  # (N, S, H)
    term1 = np.mean(abs_errors, axis=1)  # (N, H)

    # Term 2: spread via sorted samples
    # For sorted x_{(1)} <= ... <= x_{(S)}:
    #   E|X - X'| = (2 / S^2) * sum_i (2i - S - 1) * x_{(i)}
    sorted_samples = np.sort(samples, axis=1)  # (N, S, H)
    weights = (
        2.0 * np.arange(1, num_samples + 1) - num_samples - 1.0
    )  # shape (S,)
    weights = weights[np.newaxis, :, np.newaxis]  # (1, S, 1) for broadcasting
    spread = np.sum(weights * sorted_samples, axis=1) / (
        num_samples ** 2
    )  # (N, H)

    crps_values = term1 - spread  # (N, H)
    return np.mean(crps_values, axis=1)  # (N,)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def compute_all_metrics(
    ground_truth: np.ndarray,
    samples: np.ndarray,
    seasonal_errors: np.ndarray,
    metrics_to_compute: List[str],
) -> Dict[str, Any]:
    """Compute all requested metrics and return aggregated + per-series results.

    Args:
        ground_truth: Shape ``(N, H)``.
        samples: Shape ``(N, S, H)`` — sample forecast trajectories.
        seasonal_errors: Shape ``(N,)`` — from context windows.
        metrics_to_compute: List of metric names (e.g. ``["mse", "mae", ...]``).

    Returns:
        A dict with keys:
            - ``"aggregated"``: ``{metric_name: float}`` — mean across series.
            - ``"per_series"``: ``{metric_name: np.ndarray}`` — per-series values.
    """
    # Median forecast (0.5 quantile along samples axis)
    median_forecast = np.median(samples, axis=1)  # (N, H)

    per_series: Dict[str, np.ndarray] = {}
    aggregated: Dict[str, float] = {}

    metric_funcs = {
        "mse": lambda: mse(ground_truth, median_forecast),
        "mae": lambda: mae(ground_truth, median_forecast),
        "mase": lambda: mase(ground_truth, median_forecast, seasonal_errors),
        "smape": lambda: smape(ground_truth, median_forecast),
        "crps": lambda: crps_ensemble(ground_truth, samples),
    }

    for name in metrics_to_compute:
        name_lower = name.lower()
        if name_lower not in metric_funcs:
            logger.warning("Unknown metric '%s' — skipping.", name)
            continue

        values = metric_funcs[name_lower]()
        per_series[name_lower] = values

        # For MASE, ignore NaN entries (from zero seasonal error)
        aggregated[name_lower] = float(np.nanmean(values))

    logger.info("Metric summary: %s", aggregated)

    return {
        "aggregated": aggregated,
        "per_series": per_series,
    }
