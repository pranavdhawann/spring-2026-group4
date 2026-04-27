"""Wrapper around Chronos forecasting pipelines.

Provides a stable interface for loading the model and running zero-shot
quantile-based inference.  Supports Chronos-2, Chronos-Bolt, and legacy
Chronos-T5 models through the unified ``BaseChronosPipeline`` entrypoint.

Chronos2Pipeline.predict_quantiles returns a **list** of tensors:
    quantiles — list of N tensors, each shaped ``(D, H, Q)``
    mean      — list of N tensors, each shaped ``(D, H)``
    where D=1 for univariate forecasting.

ChronosPipeline / ChronosBoltPipeline return plain tensors:
    quantiles — ``(N, H, Q)``
    mean      — ``(N, H)``

This wrapper normalises all outputs to:
    quantiles — ``(N, Q, H)``
    mean      — ``(N, H)``
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ChronosForecaster:
    """Zero-shot forecaster backed by a Chronos pretrained model.

    Args:
        model_id: HuggingFace model identifier (e.g. "amazon/chronos-2").
        device_map: Device placement string ("cpu", "cuda", "auto").
        torch_dtype: Data type string ("float32", "float16", "bfloat16").
    """

    def __init__(
        self,
        model_id: str = "amazon/chronos-2",
        device_map: str = "cpu",
        torch_dtype: str = "float32",
    ) -> None:
        from chronos import BaseChronosPipeline

        dtype_map: Dict[str, torch.dtype] = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        resolved_dtype = dtype_map.get(torch_dtype, torch.float32)

        logger.info(
            "Loading Chronos model '%s' on device='%s', dtype=%s",
            model_id, device_map, torch_dtype,
        )

        # Chronos2Pipeline uses 'dtype'; older pipelines use 'torch_dtype'.
        try:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_id, device_map=device_map, dtype=resolved_dtype,
            )
        except TypeError:
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_id, device_map=device_map, torch_dtype=resolved_dtype,
            )

        self.model_id = model_id
        self._pipeline_class = type(self.pipeline).__name__
        logger.info("Model loaded successfully (%s).", self._pipeline_class)

    # ------------------------------------------------------------------
    # Internal: unpack raw predict_quantiles output
    # ------------------------------------------------------------------

    @staticmethod
    def _to_np(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().float().numpy()
        return np.asarray(x, dtype=np.float64)

    def _unpack_output(
        self,
        q_raw,
        m_raw,
        num_quantiles: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalise raw predict_quantiles output to (N, Q, H) and (N, H).

        Handles both:
          - Chronos2Pipeline: list of tensors, each (D, H, Q) / (D, H)
          - ChronosPipeline / ChronosBoltPipeline: tensor (N, H, Q) / (N, H)
        """
        # --- List output (Chronos2Pipeline) ---
        if isinstance(q_raw, list):
            q_list = []
            m_list = []
            for qi, mi in zip(q_raw, m_raw):
                q_np = self._to_np(qi)  # (D, H, Q) where D=1 for univariate
                m_np = self._to_np(mi)  # (D, H)

                # Squeeze the target-dimension axis (D=1)
                while q_np.ndim > 2:
                    if q_np.shape[0] == 1:
                        q_np = q_np.squeeze(0)
                    else:
                        break
                while m_np.ndim > 1 and m_np.shape[0] == 1 and m_np.ndim > 1:
                    m_np = m_np.squeeze(0)

                # q_np should now be (H, Q), m_np should be (H,)
                q_list.append(q_np)
                m_list.append(m_np)

            q = np.stack(q_list, axis=0)  # (N, H, Q)
            m = np.stack(m_list, axis=0)  # (N, H)

            # Transpose to (N, Q, H)
            if q.ndim == 3 and q.shape[2] == num_quantiles:
                q = np.transpose(q, (0, 2, 1))

            return q, m

        # --- Tensor output (ChronosPipeline / ChronosBoltPipeline) ---
        q = self._to_np(q_raw)
        m = self._to_np(m_raw)

        if q.ndim == 3:
            if q.shape[2] == num_quantiles and q.shape[1] != num_quantiles:
                q = np.transpose(q, (0, 2, 1))  # (N, H, Q) -> (N, Q, H)

        # Handle missing batch dim
        if q.ndim == 2:
            q = q[np.newaxis, :, :]
        if m.ndim == 1:
            m = m[np.newaxis, :]

        return q, m

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def predict_quantiles(
        self,
        contexts: List[torch.Tensor],
        prediction_length: int,
        quantile_levels: List[float],
        batch_size: int = 32,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate quantile forecasts for a batch of series.

        Args:
            contexts: List of 1-D float tensors, one per series.
            prediction_length: Number of future time steps to forecast.
            quantile_levels: Quantile levels (e.g. [0.1, 0.5, 0.9]).
            batch_size: Number of series per forward pass.

        Returns:
            quantiles ``(N, Q, H)`` and mean ``(N, H)``.
        """
        all_q: List[np.ndarray] = []
        all_m: List[np.ndarray] = []
        n_q = len(quantile_levels)

        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            logger.debug(
                "Inference batch %d-%d / %d",
                start, min(start + batch_size, len(contexts)), len(contexts),
            )

            # Chronos2Pipeline uses 'inputs'; older pipelines use 'context'
            try:
                q_raw, m_raw = self.pipeline.predict_quantiles(
                    inputs=batch,
                    prediction_length=prediction_length,
                    quantile_levels=quantile_levels,
                )
            except TypeError:
                q_raw, m_raw = self.pipeline.predict_quantiles(
                    context=batch,
                    prediction_length=prediction_length,
                    quantile_levels=quantile_levels,
                )

            q, m = self._unpack_output(q_raw, m_raw, n_q)
            all_q.append(q)
            all_m.append(m)

        result_q = np.concatenate(all_q, axis=0)  # (N, Q, H)
        result_m = np.concatenate(all_m, axis=0)  # (N, H)

        logger.info(
            "Quantile forecasts: shape %s (series=%d, quantiles=%d, horizon=%d)",
            result_q.shape, result_q.shape[0], result_q.shape[1], result_q.shape[2],
        )
        return result_q, result_m

    # ------------------------------------------------------------------
    # Pseudo-sample generation (for CRPS)
    # ------------------------------------------------------------------

    @staticmethod
    def quantiles_to_pseudo_samples(
        quantile_forecast: np.ndarray,
        quantile_levels: List[float],
        num_samples: int = 100,
    ) -> np.ndarray:
        """Generate pseudo-samples from quantile forecasts for CRPS.

        Args:
            quantile_forecast: Shape ``(N, Q, H)``.
            quantile_levels: The Q quantile levels.
            num_samples: Number of pseudo-samples to generate.

        Returns:
            Array of shape ``(N, num_samples, H)``.
        """
        if quantile_forecast.ndim != 3:
            raise ValueError(
                f"quantile_forecast must be (N, Q, H), got {quantile_forecast.shape}"
            )

        n_series, n_quantiles, horizon = quantile_forecast.shape
        levels = np.asarray(quantile_levels, dtype=np.float64)

        if len(levels) != n_quantiles:
            raise ValueError(
                f"Q dimension {n_quantiles} != len(quantile_levels) {len(levels)}"
            )

        order = np.argsort(levels)
        levels = levels[order]
        quantile_forecast = quantile_forecast[:, order, :]

        if n_quantiles == 1:
            return np.repeat(quantile_forecast, repeats=num_samples, axis=1)

        # Extend to [0, 1] by linear extrapolation
        extended_levels = np.concatenate([[0.0], levels, [1.0]])

        low_slope = (
            quantile_forecast[:, 1:2, :] - quantile_forecast[:, 0:1, :]
        ) / (levels[1] - levels[0])
        high_slope = (
            quantile_forecast[:, -1:, :] - quantile_forecast[:, -2:-1, :]
        ) / (levels[-1] - levels[-2])

        low_ext = quantile_forecast[:, 0:1, :] - low_slope * levels[0]
        high_ext = quantile_forecast[:, -1:, :] + high_slope * (1.0 - levels[-1])

        extended_values = np.concatenate(
            [low_ext, quantile_forecast, high_ext], axis=1
        )

        uniform_samples = np.linspace(0.01, 0.99, num_samples)
        samples = np.zeros((n_series, num_samples, horizon), dtype=np.float64)

        for i, u in enumerate(uniform_samples):
            idx = np.searchsorted(extended_levels, u, side="right") - 1
            idx = np.clip(idx, 0, len(extended_levels) - 2)
            denom = extended_levels[idx + 1] - extended_levels[idx]
            frac = (u - extended_levels[idx]) / (denom + 1e-12)
            samples[:, i, :] = (
                extended_values[:, idx, :] * (1.0 - frac)
                + extended_values[:, idx + 1, :] * frac
            )

        return samples
