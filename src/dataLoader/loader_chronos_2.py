"""Evaluation data splitting and batching.

Splits each :class:`TimeSeriesDataset` into a context window (model input)
and a ground-truth horizon, then computes the seasonal error needed for MASE.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from src.dataset.time_series_dataset_chronos_2 import TimeSeriesDataset
from src.vocab.constants_chronos_2 import MAX_CONTEXT_LENGTH

logger = logging.getLogger(__name__)


@dataclass
class EvaluationBatch:
    """Holds one series ready for evaluation.

    Attributes:
        series_id: Identifier of the originating series.
        context: 1-D float tensor fed to the model as historical context.
        ground_truth: 1-D numpy array of actual future values (horizon).
        seasonal_error: Mean absolute seasonal difference computed from the
            context, used as the MASE scaling denominator.
    """

    series_id: str
    context: torch.Tensor
    ground_truth: np.ndarray
    seasonal_error: float


def _compute_seasonal_error(values: np.ndarray, seasonality: int) -> float:
    """Compute the mean absolute seasonal difference from in-sample data.

    This mirrors the GluonTS ``seasonal_error`` computation:
    ``mean(|y_t - y_{t-m}|)`` for ``t = m, ..., T-1``.

    Args:
        values: 1-D array of in-sample (context) observations.
        seasonality: Seasonal period ``m``.

    Returns:
        The seasonal error. Returns ``np.nan`` if the value would be zero
        (to avoid division-by-zero in MASE).
    """
    if seasonality <= 0:
        seasonality = 1

    # Fall back to seasonality=1 when context is shorter than the period
    if len(values) <= seasonality:
        seasonality = 1

    diffs = np.abs(values[seasonality:] - values[:-seasonality])
    error = float(np.mean(diffs)) if len(diffs) > 0 else 0.0

    if error == 0.0:
        logger.warning(
            "Seasonal error is 0 (constant series?) — MASE will be excluded "
            "for this series."
        )
        return float("nan")

    return error


class EvaluationSplitter:
    """Splits time series into context / ground-truth pairs for evaluation.

    Args:
        prediction_length: Number of future time steps to forecast.
        max_context_length: Maximum number of context observations to keep.
            Defaults to :data:`MAX_CONTEXT_LENGTH` (8192).
    """

    def __init__(
        self,
        prediction_length: int,
        max_context_length: int = MAX_CONTEXT_LENGTH,
    ) -> None:
        self.prediction_length = prediction_length
        self.max_context_length = max_context_length

    def split(
        self,
        datasets: List[TimeSeriesDataset],
    ) -> List[EvaluationBatch]:
        """Split each dataset into context and horizon.

        The last ``prediction_length`` observations become the ground truth;
        the preceding observations (up to ``max_context_length``) become the
        model context.

        Args:
            datasets: List of :class:`TimeSeriesDataset` instances.

        Returns:
            A list of :class:`EvaluationBatch` instances ready for inference.

        Raises:
            ValueError: If any series is too short to form both context and
                horizon.
        """
        batches: List[EvaluationBatch] = []

        for ds in datasets:
            n = len(ds.values)
            h = self.prediction_length

            if n <= h:
                raise ValueError(
                    f"Series '{ds.series_id}' has {n} observations but "
                    f"prediction_length is {h}. Need at least {h + 1} points."
                )

            ground_truth = ds.values[-h:]
            context_values = ds.values[:-h]

            # Truncate context from the left if it exceeds max length
            if len(context_values) > self.max_context_length:
                context_values = context_values[-self.max_context_length :]

            seasonal_error = _compute_seasonal_error(
                context_values, ds.seasonality
            )

            context_tensor = torch.tensor(
                context_values, dtype=torch.float32
            )

            batches.append(
                EvaluationBatch(
                    series_id=ds.series_id,
                    context=context_tensor,
                    ground_truth=ground_truth,
                    seasonal_error=seasonal_error,
                )
            )

            logger.debug(
                "Series '%s': context=%d, horizon=%d, seasonal_error=%.4f",
                ds.series_id,
                len(context_values),
                h,
                seasonal_error,
            )

        logger.info(
            "Split %d series into context/horizon (prediction_length=%d)",
            len(batches),
            self.prediction_length,
        )
        return batches
