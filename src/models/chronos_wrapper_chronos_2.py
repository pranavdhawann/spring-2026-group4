"""Wrapper around Chronos forecasting pipelines.

Provides a stable interface for loading the model and running zero-shot
quantile-based inference across Chronos package versions.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ChronosForecaster:
    """Zero-shot forecaster backed by a Chronos pretrained model.

    Args:
        model_id: HuggingFace model identifier (for example "amazon/chronos-2").
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
            model_id,
            device_map,
            torch_dtype,
        )

        # Compatibility: newer versions prefer dtype, older may use torch_dtype.
        from_pretrained_errors: List[Exception] = []
        self.pipeline = None
        for dtype_kwarg in ("dtype", "torch_dtype"):
            try:
                self.pipeline = BaseChronosPipeline.from_pretrained(
                    model_id,
                    device_map=device_map,
                    **{dtype_kwarg: resolved_dtype},
                )
                logger.info("Loaded model using keyword '%s'.", dtype_kwarg)
                break
            except TypeError as exc:
                from_pretrained_errors.append(exc)
                if "unexpected keyword" not in str(exc).lower():
                    raise
            except Exception as exc:  # pragma: no cover - defensive path
                from_pretrained_errors.append(exc)
                raise

        if self.pipeline is None:
            raise RuntimeError(
                "Failed to load Chronos pipeline with supported dtype keywords. "
                f"Errors: {[str(e) for e in from_pretrained_errors]}"
            )

        self.model_id = model_id
        logger.info("Model loaded successfully.")

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Convert tensors to numpy arrays without modifying numpy inputs."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @classmethod
    def _to_numpy_stacked(cls, value: Any) -> np.ndarray:
        """Convert list outputs from Chronos2 into a single batch array."""
        if isinstance(value, (list, tuple)):
            if not value:
                return np.array([])
            return np.stack([cls._to_numpy(v) for v in value], axis=0)
        return cls._to_numpy(value)

    @staticmethod
    def _is_unexpected_kwarg_error(exc: TypeError) -> bool:
        message = str(exc).lower()
        patterns = (
            "unexpected keyword",
            "unexpected kwargs",
            "got an unexpected keyword",
            "unexpected keyword argument",
            "missing 1 required positional argument",
            "required positional argument",
            "positional arguments but",
            "takes",
        )
        return any(p in message for p in patterns)

    @staticmethod
    def _squeeze_non_batch_singletons(arr: np.ndarray, target_ndim: int) -> np.ndarray:
        """Squeeze singleton dims except batch dim until target_ndim is reached."""
        out = arr
        while out.ndim > target_ndim:
            squeezed = False
            for axis in range(1, out.ndim):
                if out.shape[axis] == 1:
                    out = np.squeeze(out, axis=axis)
                    squeezed = True
                    break
            if not squeezed:
                break
        return out

    @classmethod
    def _normalize_quantiles(
        cls,
        quantiles: np.ndarray,
        quantile_levels: List[float],
    ) -> np.ndarray:
        """Normalize quantile outputs to shape (N, Q, H)."""
        q = cls._to_numpy_stacked(quantiles)
        q = cls._squeeze_non_batch_singletons(q, target_ndim=3)

        if q.ndim != 3:
            raise ValueError(
                "Unexpected quantile output rank. Expected 3 dims after "
                f"squeezing singleton dims, got shape {q.shape}."
            )

        q_count = len(quantile_levels)
        if q.shape[1] == q_count:
            return q
        if q.shape[2] == q_count:
            return np.swapaxes(q, 1, 2)

        raise ValueError(
            "Could not locate quantile axis in output shape "
            f"{q.shape} for quantile_levels size {q_count}."
        )

    @classmethod
    def _normalize_mean(cls, mean: np.ndarray) -> np.ndarray:
        """Normalize mean outputs to shape (N, H)."""
        m = cls._to_numpy_stacked(mean)
        m = cls._squeeze_non_batch_singletons(m, target_ndim=2)
        if m.ndim == 1:
            m = m[np.newaxis, :]
        if m.ndim != 2:
            raise ValueError(
                "Unexpected mean output rank. Expected 2 dims after squeezing "
                f"singleton dims, got shape {m.shape}."
            )
        return m

    def _fallback_predict(
        self,
        batch: List[torch.Tensor],
        prediction_length: int,
        quantile_levels: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback when predict_quantiles signature differs across versions."""
        fallback_errors: List[Exception] = []

        # Try predict(context=...) then positional predict(...).
        for call in (
            lambda: self.pipeline.predict(
                inputs=batch,
                prediction_length=prediction_length,
            ),
            lambda: self.pipeline.predict(
                context=batch,
                prediction_length=prediction_length,
            ),
            lambda: self.pipeline.predict(batch, prediction_length=prediction_length),
            lambda: self.pipeline.predict(batch, prediction_length),
        ):
            try:
                output = call()
                break
            except Exception as exc:
                fallback_errors.append(exc)
        else:
            raise RuntimeError(
                "Chronos fallback predict() calls failed. "
                f"Errors: {[str(e) for e in fallback_errors]}"
            )

        if isinstance(output, (tuple, list)) and len(output) == 2:
            first = self._to_numpy_stacked(output[0])
            second = self._to_numpy_stacked(output[1])
            try:
                quantiles = self._normalize_quantiles(first, quantile_levels)
                mean = self._normalize_mean(second)
                if mean.shape != (quantiles.shape[0], quantiles.shape[2]):
                    mean = np.mean(quantiles, axis=1)
                return quantiles, mean
            except Exception:
                output = first
        else:
            output = self._to_numpy_stacked(output)

        output = self._squeeze_non_batch_singletons(output, target_ndim=3)
        if output.ndim != 3:
            raise ValueError(
                "Chronos predict() fallback returned unexpected shape "
                f"{output.shape}; expected 3 dimensions."
            )

        # Possible shapes:
        #   (N, Q, H) quantiles
        #   (N, H, Q) quantiles
        #   (N, S, H) samples
        q_count = len(quantile_levels)
        if output.shape[1] == q_count:
            quantiles = output
            mean = np.mean(quantiles, axis=1)
            return quantiles, mean
        if output.shape[2] == q_count:
            quantiles = np.swapaxes(output, 1, 2)
            mean = np.mean(quantiles, axis=1)
            return quantiles, mean

        # Assume sample trajectories and derive quantiles.
        samples = output
        mean = np.mean(samples, axis=1)
        quantiles = np.quantile(
            samples,
            q=np.asarray(quantile_levels),
            axis=1,
        ).transpose(1, 0, 2)
        return quantiles, mean

    def _predict_quantiles_compat(
        self,
        batch: List[torch.Tensor],
        prediction_length: int,
        quantile_levels: List[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Call pipeline.predict_quantiles() compatibly across versions."""
        attempts = (
            lambda: self.pipeline.predict_quantiles(
                inputs=batch,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
            ),
            lambda: self.pipeline.predict_quantiles(
                inputs=batch,
                prediction_length=prediction_length,
                quantiles=quantile_levels,
            ),
            lambda: self.pipeline.predict_quantiles(
                context=batch,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
            ),
            lambda: self.pipeline.predict_quantiles(
                contexts=batch,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
            ),
            lambda: self.pipeline.predict_quantiles(
                context=batch,
                prediction_length=prediction_length,
                quantiles=quantile_levels,
            ),
            lambda: self.pipeline.predict_quantiles(
                batch,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
            ),
            lambda: self.pipeline.predict_quantiles(
                batch,
                prediction_length,
                quantile_levels,
            ),
        )

        last_exc: Exception | None = None
        for call in attempts:
            try:
                output = call()
                if isinstance(output, (tuple, list)) and len(output) == 2:
                    quantiles, mean = output
                    quantiles = self._normalize_quantiles(
                        quantiles, quantile_levels
                    )
                    mean = self._normalize_mean(mean)
                    if mean.shape != (quantiles.shape[0], quantiles.shape[2]):
                        mean = np.mean(quantiles, axis=1)
                else:
                    quantiles = self._to_numpy_stacked(output)
                    quantiles = self._squeeze_non_batch_singletons(
                        quantiles, target_ndim=3
                    )
                    if quantiles.ndim != 3:
                        raise ValueError(
                            "Unexpected non-tuple predict_quantiles output shape: "
                            f"{quantiles.shape}"
                        )
                    if (
                        quantiles.shape[1] != len(quantile_levels)
                        and quantiles.shape[2] != len(quantile_levels)
                    ):
                        # If this is samples, derive quantiles.
                        quantiles = np.quantile(
                            quantiles,
                            q=np.asarray(quantile_levels),
                            axis=1,
                        ).transpose(1, 0, 2)
                    else:
                        quantiles = self._normalize_quantiles(
                            quantiles, quantile_levels
                        )
                    mean = np.mean(quantiles, axis=1)
                return quantiles, mean
            except AttributeError as exc:
                last_exc = exc
                break
            except TypeError as exc:
                last_exc = exc
                if not self._is_unexpected_kwarg_error(exc):
                    raise
            except Exception as exc:
                last_exc = exc
                raise

        logger.warning(
            "predict_quantiles() API mismatch detected (%s). Falling back to "
            "predict() and deriving quantiles.",
            last_exc,
        )
        return self._fallback_predict(batch, prediction_length, quantile_levels)

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
            quantile_levels: Quantile levels to predict (for example
                [0.1, 0.5, 0.9]).
            batch_size: Number of series per forward pass.

        Returns:
            A tuple of:
                - quantiles: array of shape (N, Q, H)
                - mean: array of shape (N, H)
        """
        all_quantiles: List[np.ndarray] = []
        all_means: List[np.ndarray] = []

        for start in range(0, len(contexts), batch_size):
            batch = contexts[start : start + batch_size]
            logger.debug(
                "Inference batch %d-%d / %d",
                start,
                min(start + batch_size, len(contexts)),
                len(contexts),
            )

            quantiles, mean = self._predict_quantiles_compat(
                batch=batch,
                prediction_length=prediction_length,
                quantile_levels=quantile_levels,
            )
            quantiles = self._normalize_quantiles(quantiles, quantile_levels)
            mean = self._normalize_mean(mean)
            if mean.shape != (quantiles.shape[0], quantiles.shape[2]):
                mean = np.mean(quantiles, axis=1)

            all_quantiles.append(quantiles)
            all_means.append(mean)

        result_q = np.concatenate(all_quantiles, axis=0)  # (N, Q, H)
        result_m = np.concatenate(all_means, axis=0)  # (N, H)

        logger.info(
            "Generated quantile forecasts: shape %s (series, quantiles, horizon)",
            result_q.shape,
        )
        return result_q, result_m

    @staticmethod
    def quantiles_to_pseudo_samples(
        quantile_forecast: np.ndarray,
        quantile_levels: List[float],
        num_samples: int = 100,
    ) -> np.ndarray:
        """Generate pseudo-samples from quantile forecasts for CRPS.

        Uses linear interpolation between quantile levels to approximate
        the inverse CDF, then draws deterministic uniform samples and maps
        them through that inverse CDF approximation.

        Args:
            quantile_forecast: Shape (N, Q, H) quantile values.
            quantile_levels: The Q quantile levels (for example [0.1, ..., 0.9]).
            num_samples: Number of pseudo-samples to generate.

        Returns:
            Array of shape (N, num_samples, H).
        """
        if quantile_forecast.ndim != 3:
            raise ValueError(
                "quantile_forecast must have shape (N, Q, H), got "
                f"{quantile_forecast.shape}"
            )
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        n_series, n_quantiles, horizon = quantile_forecast.shape
        levels = np.asarray(quantile_levels, dtype=np.float64)

        if len(levels) != n_quantiles:
            raise ValueError(
                "Mismatch between quantile_forecast Q dimension and "
                f"quantile_levels: {n_quantiles} vs {len(levels)}"
            )

        # Sort quantiles if user provided unsorted levels.
        order = np.argsort(levels)
        levels = levels[order]
        quantile_forecast = quantile_forecast[:, order, :]

        if np.any(np.diff(levels) <= 0):
            raise ValueError("quantile_levels must be strictly increasing")

        # If only one quantile is available, treat it as deterministic.
        if n_quantiles == 1:
            return np.repeat(quantile_forecast, repeats=num_samples, axis=1)

        # Extend to cover [0, 1] by linear extrapolation at both edges.
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
        )  # (N, Q + 2, H)

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
