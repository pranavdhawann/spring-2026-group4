"""
Chronos Model Evaluation Script

Evaluates a Chronos model (base or LoRA-finetuned) across all tickers in a
manifest.  For each ticker the last `prediction_length` values are held out as
ground truth, everything before is fed as context, and probabilistic forecasts
are generated.

Metrics computed per-ticker and in aggregate:
    - MASE  (scaled by naive lag-1 forecast on the context window)
    - RMSE
    - MAE
    - MAPE
    - WQL   (weighted quantile loss at quantiles 0.1, 0.5, 0.9)

Outputs comparison-ready CSVs to `output_dir`:
    - per_ticker_metrics.csv
    - aggregate_metrics.csv

Usage:
    python -m src.utils.evaluate_chronos \
        --model_path amazon/chronos-2-large \
        --manifest_path data/chronos_manifest.json \
        --data_dir data/multi-modal-dataset/sp500_time_series \
        --prediction_length 7 \
        --output_dir experiments/baseline/chronos/
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

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of cwd
# ---------------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.preProcessing.data_preprocessing_chronos import load_ticker
from src.utils.metrics_utils import calculate_regression_metrics
from src.utils.utils import read_json_file, set_seed

# Chronos import (optional at module level for import-check convenience)
try:
    from chronos import ChronosPipeline
except ImportError:
    ChronosPipeline = None

logger = logging.getLogger(__name__)

# Default quantiles for weighted quantile loss
WQL_QUANTILES = [0.1, 0.5, 0.9]


# ============================================================================
# Metric helpers
# ============================================================================


def compute_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    context: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    The scaling denominator is the MAE of the naive seasonal forecast on the
    *context* window (lag = ``seasonality``).  If the denominator is zero
    (constant series) the function returns ``np.inf``.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values (prediction horizon).
    y_pred : np.ndarray
        Point forecast values (same length as *y_true*).
    context : np.ndarray
        Historical context values used for the naive-forecast denominator.
    seasonality : int, optional
        Seasonal lag for the naive forecast (default ``1`` = lag-1).
    """
    numerator = np.mean(np.abs(y_true - y_pred))

    # Naive in-sample seasonal forecast error
    naive_errors = np.abs(context[seasonality:] - context[:-seasonality])
    denominator = np.mean(naive_errors)

    if denominator == 0.0:
        return np.inf

    return float(numerator / denominator)


def compute_wql(
    y_true: np.ndarray,
    quantile_forecasts: Dict[float, np.ndarray],
    quantiles: Optional[List[float]] = None,
) -> float:
    """Weighted Quantile Loss (averaged over the requested quantiles).

    For each quantile *q* the pinball loss is:

        L_q = 2 / sum(|y|) * sum( max(q*(y - ŷ_q), (q-1)*(y - ŷ_q)) )

    The final WQL is the simple average across all quantiles.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.
    quantile_forecasts : dict[float, np.ndarray]
        Mapping from quantile level to the corresponding forecast array.
    quantiles : list[float], optional
        Which quantiles to evaluate (default ``WQL_QUANTILES``).
    """
    if quantiles is None:
        quantiles = WQL_QUANTILES

    abs_sum = np.sum(np.abs(y_true))
    if abs_sum == 0.0:
        return np.inf

    total_loss = 0.0
    for q in quantiles:
        y_hat_q = quantile_forecasts[q]
        diff = y_true - y_hat_q
        pinball = np.where(diff >= 0, q * diff, (q - 1.0) * diff)
        total_loss += 2.0 * np.sum(pinball) / abs_sum

    return float(total_loss / len(quantiles))


# ============================================================================
# Model loading
# ============================================================================


def load_chronos_model(
    model_path: str,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float32,
) -> "ChronosPipeline":
    """Load a Chronos pipeline, handling LoRA adapters transparently.

    If *model_path* contains an ``adapter_config.json`` the base model name is
    read from that file, the base pipeline is loaded, and the LoRA adapter is
    merged in via PEFT.  Otherwise the model is loaded directly.
    """
    if ChronosPipeline is None:
        raise ImportError(
            "The 'chronos' package is required.  Install it with: "
            "pip install chronos-forecasting"
        )

    adapter_config_path = Path(model_path) / "adapter_config.json"

    if adapter_config_path.exists():
        logger.info("LoRA adapter detected at %s", model_path)
        with open(adapter_config_path, "r") as f:
            adapter_cfg = json.load(f)

        base_model_name = adapter_cfg.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(
                "adapter_config.json does not contain 'base_model_name_or_path'"
            )

        logger.info("Loading base model: %s", base_model_name)
        pipeline = ChronosPipeline.from_pretrained(
            base_model_name,
            device_map=device,
            torch_dtype=torch_dtype,
        )

        # Merge LoRA adapter
        from peft import PeftModel

        pipeline.model.model = PeftModel.from_pretrained(
            pipeline.model.model, model_path
        )
        pipeline.model.model = pipeline.model.model.merge_and_unload()
        logger.info("LoRA adapter merged successfully")
    else:
        logger.info("Loading model directly from %s", model_path)
        pipeline = ChronosPipeline.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch_dtype,
        )

    return pipeline


# ============================================================================
# Per-ticker evaluation
# ============================================================================


def evaluate_ticker(
    pipeline: "ChronosPipeline",
    dates: np.ndarray,
    values: np.ndarray,
    prediction_length: int,
    num_samples: int,
) -> Optional[Dict[str, float]]:
    """Evaluate a single ticker and return a metrics dict (or None on error).

    Parameters
    ----------
    pipeline : ChronosPipeline
        Loaded Chronos pipeline.
    dates, values : np.ndarray
        Full preprocessed time-series (dates and target values).
    prediction_length : int
        Number of steps to hold out and forecast.
    num_samples : int
        Number of probabilistic samples to draw.

    Returns
    -------
    dict or None
        Keys: mase, rmse, mae, mape, wql, and per-quantile wql columns.
    """
    if len(values) <= prediction_length:
        logger.warning(
            "Series too short (%d <= %d). Skipping.", len(values), prediction_length
        )
        return None

    # Split into context and ground truth
    context = values[:-prediction_length]
    y_true = values[-prediction_length:]

    # Prepare context tensor  [1, context_length]
    context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        # forecast_samples shape: [1, num_samples, prediction_length]
        forecast_samples = pipeline.predict(
            context_tensor,
            prediction_length=prediction_length,
            num_samples=num_samples,
        )

    # Squeeze batch dim → [num_samples, prediction_length]
    samples = forecast_samples.squeeze(0)

    # Point forecast (median)
    median_forecast = torch.quantile(samples, 0.5, dim=0).cpu().numpy()

    # Quantile forecasts for WQL
    quantile_forecasts = {}
    for q in WQL_QUANTILES:
        quantile_forecasts[q] = torch.quantile(samples, q, dim=0).cpu().numpy()

    # --- Compute metrics ---

    # Reuse existing util for RMSE, MAE, MAPE (also gives MSE, SMAPE as bonus)
    base_metrics = calculate_regression_metrics(y_true, median_forecast)

    mase = compute_mase(y_true, median_forecast, context, seasonality=1)
    wql = compute_wql(y_true, quantile_forecasts)

    return {
        "mase": mase,
        "rmse": base_metrics["rmse"],
        "mae": base_metrics["mae"],
        "mape": base_metrics["mape"],
        "wql": wql,
    }


# ============================================================================
# Full evaluation run
# ============================================================================


def run_evaluation(
    model_path: str,
    manifest_path: str,
    data_dir: str,
    prediction_length: int = 7,
    num_samples: int = 20,
    target_col: str = "Close",
    output_dir: str = "experiments/baseline/chronos/",
    device: str = "cuda",
    seed: int = 42,
) -> pd.DataFrame:
    """Run evaluation over all manifest tickers and write result CSVs.

    Returns
    -------
    pd.DataFrame
        Per-ticker metrics DataFrame.
    """
    set_seed(seed)

    # Derive a short model label for the CSV (useful for diffing)
    model_label = Path(model_path).name if Path(model_path).exists() else model_path

    # ------------------------------------------------------------------
    # 1. Load manifest
    # ------------------------------------------------------------------
    manifest = read_json_file(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"Could not read manifest at {manifest_path}")
    print(f"\nManifest loaded: {len(manifest)} tickers from {manifest_path}")

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print(f"Loading model from '{model_path}' on {device} ...")
    pipeline = load_chronos_model(model_path, device=device)
    print("Model loaded.\n")

    # ------------------------------------------------------------------
    # 3. Per-ticker evaluation
    # ------------------------------------------------------------------
    data_dir_path = Path(data_dir)
    rows: List[Dict] = []
    total = len(manifest)

    for idx, entry in enumerate(manifest, 1):
        ticker = entry["ticker"]
        file_path = entry.get("file_path")

        # Resolve CSV path
        if file_path and Path(file_path).exists():
            csv_path = str(file_path)
        else:
            csv_path = str(data_dir_path / f"{ticker.lower()}.csv")

        print(f"[{idx}/{total}] Evaluating {ticker} ...", end=" ")

        try:
            dates, values = load_ticker(csv_path, target_col=target_col)
        except Exception as exc:
            print(f"SKIP (load error: {exc})")
            continue

        metrics = evaluate_ticker(
            pipeline, dates, values, prediction_length, num_samples
        )

        if metrics is None:
            print("SKIP (too short)")
            continue

        metrics["ticker"] = ticker
        metrics["model"] = model_label
        rows.append(metrics)
        print(
            f"MASE={metrics['mase']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"MAE={metrics['mae']:.4f}  MAPE={metrics['mape']:.2f}%  "
            f"WQL={metrics['wql']:.4f}"
        )

    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not rows:
        print("\nNo tickers were evaluated successfully.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 4. Build DataFrames
    # ------------------------------------------------------------------
    per_ticker_df = pd.DataFrame(rows)
    col_order = ["model", "ticker", "mase", "rmse", "mae", "mape", "wql"]
    per_ticker_df = per_ticker_df[col_order]

    # Aggregate (mean across tickers)
    metric_cols = ["mase", "rmse", "mae", "mape", "wql"]
    agg_dict = {col: per_ticker_df[col].replace([np.inf, -np.inf], np.nan).mean()
                for col in metric_cols}
    agg_dict["model"] = model_label
    agg_dict["num_tickers"] = len(rows)
    aggregate_df = pd.DataFrame([agg_dict])
    aggregate_df = aggregate_df[["model", "num_tickers"] + metric_cols]

    # ------------------------------------------------------------------
    # 5. Write CSVs
    # ------------------------------------------------------------------
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    per_ticker_csv = out_path / "per_ticker_metrics.csv"
    aggregate_csv = out_path / "aggregate_metrics.csv"

    per_ticker_df.to_csv(per_ticker_csv, index=False)
    aggregate_df.to_csv(aggregate_csv, index=False)

    print(f"\nPer-ticker metrics saved to: {per_ticker_csv}")
    print(f"Aggregate metrics saved to:  {aggregate_csv}")

    # ------------------------------------------------------------------
    # 6. Print aggregate summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("AGGREGATE METRICS")
    print("=" * 60)
    print(aggregate_df.to_string(index=False))
    print("=" * 60)

    return per_ticker_df


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Chronos model (base or LoRA) on a ticker manifest."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="amazon/chronos-2-large",
        help="HuggingFace model name or local path (base or LoRA adapter).",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        required=True,
        help="Path to the JSON manifest from data_filter_chronos.py.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory containing ticker CSVs.",
    )
    parser.add_argument(
        "--prediction_length",
        type=int,
        default=7,
        help="Forecast horizon / number of holdout steps (default: 7).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=20,
        help="Number of probabilistic forecast samples (default: 20).",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="Close",
        help="Target column to forecast (default: Close).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/baseline/chronos/",
        help="Directory to write result CSVs (default: experiments/baseline/chronos/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Compute device (default: cuda).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    return parser.parse_args()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    args = parse_args()

    run_evaluation(
        model_path=args.model_path,
        manifest_path=args.manifest_path,
        data_dir=args.data_dir,
        prediction_length=args.prediction_length,
        num_samples=args.num_samples,
        target_col=args.target_col,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
