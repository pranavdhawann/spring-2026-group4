#!/usr/bin/env python3
"""Chronos 2 Zero-Shot Evaluation Pipeline — Multi-Stock Edition.

Scans a directory of {ticker}.csv files, samples a configurable fraction,
runs zero-shot Chronos 2 inference on each stock, computes metrics, and
saves aggregated results + per-stock forecast plots.

Usage::

    python evaluate_chronos_2.py
    python evaluate_chronos_2.py --config-dir ./custom_configs/
    python evaluate_chronos_2.py --device cuda
"""

import argparse
import logging
import random
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from src.data.csv_reader_chronos_2 import read_stock_directory
from src.preProcessing.preprocessor_chronos_2 import preprocess
from src.dataset.time_series_dataset_chronos_2 import from_dataframe
from src.dataLoader.loader_chronos_2 import EvaluationSplitter
from src.models.chronos_wrapper_chronos_2 import ChronosForecaster
from src.utils.config_chronos_2 import load_configs
from src.utils.metrics_chronos_2 import compute_all_metrics
from src.utils.io_chronos_2 import make_experiment_dir
from src.utils.plotting_chronos_2 import (
    plot_stock_forecast,
    plot_metrics_summary,
    plot_metric_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _compute_seasonal_error(values: np.ndarray, seasonality: int) -> float:
    """Compute mean absolute seasonal difference (for MASE denominator)."""
    if seasonality <= 0:
        seasonality = 1
    if len(values) <= seasonality:
        seasonality = 1

    diffs = np.abs(values[seasonality:] - values[:-seasonality])
    error = float(np.mean(diffs)) if len(diffs) > 0 else 0.0
    return float("nan") if error == 0.0 else error


def _returns_to_price(returns: np.ndarray, anchor_price: float) -> np.ndarray:
    """Convert log-return trajectory to price trajectory with a fixed anchor."""
    return anchor_price * np.exp(np.cumsum(returns, axis=-1))


def _apply_log_return_transform(
    df: "pd.DataFrame",
    ts_col: str,
    target_columns: List[str],
) -> Tuple["pd.DataFrame", Dict[str, np.ndarray]]:
    """Convert target columns from price space to log-return space."""
    import pandas as pd

    original_targets: Dict[str, np.ndarray] = {}
    transformed: Dict[str, np.ndarray] = {}

    for col in target_columns:
        values = df[col].to_numpy(dtype=np.float64)
        if len(values) < 2:
            raise ValueError(f"Not enough rows to compute log returns for '{col}'.")

        prev_vals = values[:-1]
        next_vals = values[1:]
        if np.any(prev_vals <= 0) or np.any(next_vals <= 0):
            raise ValueError(
                f"Column '{col}' has non-positive values; cannot compute log returns."
            )

        original_targets[col] = values
        transformed[col] = np.log(next_vals / prev_vals)

    out = pd.DataFrame({ts_col: df[ts_col].values[1:]})
    for col in target_columns:
        out[col] = transformed[col]

    return out, original_targets


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed namespace with config paths and optional overrides.
    """
    parser = argparse.ArgumentParser(
        description="Chronos 2 Zero-Shot Evaluation Pipeline",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="config",
        help=(
            "Directory containing model_chronos_2.yaml, "
            "dataset_chronos_2.yaml, evaluation_chronos_2.yaml"
        ),
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Override path for model config YAML",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Override path for dataset config YAML",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help="Override path for evaluation config YAML",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override model device_map (e.g. 'cuda', 'cpu')",
    )
    return parser.parse_args()


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Mutate *config* in-place with any CLI argument overrides.

    Args:
        config: Merged YAML configuration dictionary.
        args: Parsed CLI arguments.
    """
    if args.device:
        config.setdefault("model", {})["device_map"] = args.device


def evaluate_single_stock(
    ticker: str,
    df: "pd.DataFrame",
    dataset_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    forecaster: ChronosForecaster,
) -> Dict[str, Any]:
    """Run the full eval pipeline on a single stock.

    Args:
        ticker: Stock ticker name.
        df: Raw DataFrame for this stock.
        dataset_cfg: Dataset config section.
        eval_cfg: Evaluation config section.
        model_cfg: Model config section.
        forecaster: Already-loaded model wrapper.

    Returns:
        Dict with keys ``"ticker"``, ``"metrics"``, ``"ground_truth"``,
        ``"median_forecast"``, ``"quantile_forecast"``, ``"context_values"``,
        or ``None`` if the stock is skipped due to errors.
    """
    import pandas as pd

    ts_col = dataset_cfg.get("timestamp_column", "Date")
    prediction_length = eval_cfg["prediction_length"]
    max_ctx = model_cfg.get("max_context_length", 8192)
    quantile_levels = eval_cfg.get(
        "quantile_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    metrics_to_compute = eval_cfg.get(
        "metrics", ["mse", "mae", "mase", "smape", "crps"]
    )
    target_transform = str(dataset_cfg.get("target_transform", "none")).lower()
    if target_transform not in {"none", "log_return"}:
        raise ValueError(
            f"Unsupported target_transform='{target_transform}'. "
            "Use 'none' or 'log_return'."
        )

    # Preprocess
    df, frequency, seasonality = preprocess(df, dataset_cfg)

    # Build datasets (typically just "Close" column)
    target_columns = [c for c in df.columns if c != ts_col]
    if not target_columns:
        raise ValueError(
            f"No target columns left after preprocessing for ticker '{ticker}'."
        )

    transformed_df = df
    original_targets: Dict[str, np.ndarray] = {}
    if target_transform == "log_return":
        transformed_df, original_targets = _apply_log_return_transform(
            df, ts_col, target_columns
        )

    datasets = from_dataframe(
        transformed_df, ts_col, target_columns, frequency, seasonality
    )

    # Split context / horizon
    splitter = EvaluationSplitter(prediction_length, max_ctx)
    batches = splitter.split(datasets)

    # Inference — Chronos 2 is a quantile forecaster, not a sampler
    contexts = [b.context for b in batches]
    quantile_forecast, mean_forecast = forecaster.predict_quantiles(
        contexts=contexts,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
    )  # quantile_forecast: (N, Q, H), mean_forecast: (N, H)

    # Median = 0.5 quantile (index in quantile_levels list)
    median_idx = quantile_levels.index(0.5) if 0.5 in quantile_levels else None
    if median_idx is not None:
        median_forecast = quantile_forecast[:, median_idx, :]
    else:
        median_forecast = mean_forecast

    # Generate pseudo-samples from quantiles for CRPS computation
    num_samples = model_cfg.get("num_samples", 100)
    samples = ChronosForecaster.quantiles_to_pseudo_samples(
        quantile_forecast, quantile_levels, num_samples=num_samples,
    )

    # Ground truth + seasonal errors (in transformed space unless converted below)
    ground_truth = np.stack([b.ground_truth for b in batches])
    seasonal_errors = np.array([b.seasonal_error for b in batches])
    plot_contexts = [b.context.numpy() for b in batches]

    # Optionally convert forecasts back to price space for metrics/plots.
    if target_transform == "log_return":
        gt_prices: List[np.ndarray] = []
        median_prices: List[np.ndarray] = []
        quantile_prices: List[np.ndarray] = []
        sample_prices: List[np.ndarray] = []
        seasonal_errors_price: List[float] = []
        plot_contexts = []

        for idx, col in enumerate(target_columns):
            price_values = original_targets[col]
            end_idx = len(price_values) - 1 - prediction_length
            if end_idx < 0:
                raise ValueError(
                    f"Ticker '{ticker}' is too short after transform for '{col}'."
                )

            anchor_price = float(price_values[end_idx])

            # Match the truncated context length used by EvaluationSplitter.
            context_len = int(len(batches[idx].context))
            start_idx = max(0, end_idx - context_len)
            close_context = price_values[start_idx : end_idx + 1]
            plot_contexts.append(close_context)

            gt_prices.append(_returns_to_price(ground_truth[idx], anchor_price))
            median_prices.append(_returns_to_price(median_forecast[idx], anchor_price))
            quantile_prices.append(
                _returns_to_price(quantile_forecast[idx], anchor_price)
            )
            sample_prices.append(_returns_to_price(samples[idx], anchor_price))
            seasonal_errors_price.append(
                _compute_seasonal_error(close_context, seasonality)
            )

        ground_truth = np.stack(gt_prices, axis=0)
        median_forecast = np.stack(median_prices, axis=0)
        quantile_forecast = np.stack(quantile_prices, axis=0)
        samples = np.stack(sample_prices, axis=0)
        seasonal_errors = np.array(seasonal_errors_price)

    # Metrics
    metrics = compute_all_metrics(
        ground_truth=ground_truth,
        samples=samples,
        seasonal_errors=seasonal_errors,
        metrics_to_compute=metrics_to_compute,
    )

    # Store one representative series for plots/predictions.
    # Prefer "Close" when present, otherwise use first target column.
    plot_idx = target_columns.index("Close") if "Close" in target_columns else 0
    full_context = plot_contexts[plot_idx]

    return {
        "ticker": ticker,
        "metrics": metrics,
        "ground_truth": ground_truth[plot_idx],
        "median_forecast": median_forecast[plot_idx],
        "quantile_forecast": quantile_forecast[plot_idx],
        "context_values": full_context,
    }


def main() -> None:
    """Run the full multi-stock zero-shot evaluation pipeline."""
    args = parse_args()

    # ---- 1. Load configuration ----
    logger.info("Loading configuration...")
    config = load_configs(
        config_dir=args.config_dir,
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        eval_config_path=args.eval_config,
    )
    apply_cli_overrides(config, args)

    model_cfg: Dict = config["model"]
    dataset_cfg: Dict = config["dataset"]
    eval_cfg: Dict = config["evaluation"]

    # ---- 2. Read stock CSVs ----
    csv_dir = dataset_cfg["csv_directory"]
    sample_ratio = dataset_cfg.get("sample_ratio", 0.2)
    sample_seed = dataset_cfg.get("sample_seed", 42)

    logger.info(
        "Scanning %s (sampling %.0f%%)...", csv_dir, sample_ratio * 100
    )
    stock_dfs = read_stock_directory(
        directory=csv_dir,
        timestamp_column=dataset_cfg.get("timestamp_column", "Date"),
        target_columns=dataset_cfg.get("target_columns"),
        sample_ratio=sample_ratio,
        sample_seed=sample_seed,
        strict_target_columns=dataset_cfg.get("strict_target_columns", False),
    )
    logger.info("Loaded %d stocks for evaluation.", len(stock_dfs))

    # ---- 3. Load model (once) ----
    logger.info("Loading Chronos model...")
    forecaster = ChronosForecaster(
        model_id=model_cfg.get("model_id", "amazon/chronos-2"),
        device_map=model_cfg.get("device_map", "cpu"),
        torch_dtype=model_cfg.get("torch_dtype", "float32"),
    )

    # ---- 4. Evaluate each stock ----
    prediction_length = eval_cfg["prediction_length"]
    all_results: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for ticker, df in tqdm(stock_dfs.items(), desc="Evaluating stocks"):
        try:
            # Skip stocks with too few data points
            if len(df) <= prediction_length + 1:
                logger.warning(
                    "Skipping %s: only %d rows (need >%d)",
                    ticker, len(df), prediction_length + 1,
                )
                skipped.append(ticker)
                continue

            result = evaluate_single_stock(
                ticker=ticker,
                df=df,
                dataset_cfg=dataset_cfg,
                eval_cfg=eval_cfg,
                model_cfg=model_cfg,
                forecaster=forecaster,
            )
            all_results.append(result)

        except Exception as e:
            logger.warning("Error evaluating %s: %s", ticker, e)
            skipped.append(ticker)

    if not all_results:
        logger.error("No stocks were successfully evaluated.")
        return

    # ---- 5. Aggregate metrics across all stocks ----
    metrics_to_compute = eval_cfg.get(
        "metrics", ["mse", "mae", "mase", "smape", "crps"]
    )

    # Per-stock metric dict for aggregation
    per_stock_metrics: Dict[str, Dict[str, float]] = {}
    for r in all_results:
        per_stock_metrics[r["ticker"]] = r["metrics"]["aggregated"]

    # Global aggregation (mean of per-stock means)
    global_agg: Dict[str, float] = {}
    for metric_name in metrics_to_compute:
        values = [
            m[metric_name]
            for m in per_stock_metrics.values()
            if metric_name in m and not np.isnan(m[metric_name])
        ]
        if values:
            global_agg[metric_name] = float(np.mean(values))

    # ---- 6. Save results ----
    import pandas as pd
    import json
    import yaml

    output_dir = eval_cfg.get("output_dir", "experiments")
    exp_dir = make_experiment_dir(output_dir)

    # Aggregated metrics
    with open(exp_dir / "metrics_chronos_2.json", "w") as fp:
        json.dump(global_agg, fp, indent=2)

    # Per-stock metrics CSV
    per_stock_rows = []
    for ticker, m in per_stock_metrics.items():
        row = {"ticker": ticker, **m}
        per_stock_rows.append(row)
    per_stock_df = pd.DataFrame(per_stock_rows).sort_values("ticker")
    per_stock_df.to_csv(exp_dir / "per_stock_metrics_chronos_2.csv", index=False)

    # Predictions CSV (all stocks)
    quantile_levels = eval_cfg.get(
        "quantile_levels", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    pred_rows = []
    for r in all_results:
        for step in range(len(r["ground_truth"])):
            row = {
                "ticker": r["ticker"],
                "step": step + 1,
                "ground_truth": float(r["ground_truth"][step]),
                "median_forecast": float(r["median_forecast"][step]),
            }
            for q_idx, q_level in enumerate(quantile_levels):
                row[f"q_{q_level:.2f}"] = float(r["quantile_forecast"][q_idx, step])
            pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(exp_dir / "predictions_chronos_2.csv", index=False)

    # Config snapshot
    with open(exp_dir / "config_snapshot_chronos_2.yaml", "w") as fp:
        yaml.dump(config, fp, default_flow_style=False, sort_keys=False)

    # ---- 7. Generate plots ----
    plot_cfg = eval_cfg.get("plot", {})
    if plot_cfg.get("enabled", False):
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        num_to_plot = min(plot_cfg.get("num_stocks", 10), len(all_results))
        ctx_days = plot_cfg.get("context_days_shown", 90)

        # Pick stocks to plot (deterministic sample)
        rng = random.Random(sample_seed)
        plot_indices = sorted(rng.sample(range(len(all_results)), num_to_plot))

        logger.info("Generating forecast plots for %d stocks...", num_to_plot)
        for idx in plot_indices:
            r = all_results[idx]
            plot_stock_forecast(
                ticker=r["ticker"],
                context_values=r["context_values"],
                ground_truth=r["ground_truth"],
                median_forecast=r["median_forecast"],
                quantile_forecast=r["quantile_forecast"],
                quantile_levels=quantile_levels,
                output_dir=str(plots_dir),
                context_days_shown=ctx_days,
            )

        # Summary plots
        plot_metrics_summary(per_stock_metrics, str(plots_dir))
        plot_metric_distribution(per_stock_metrics, str(plots_dir))

    # ---- 8. Print summary ----
    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Model:             {model_cfg.get('model_id')}")
    print(f"  Data directory:    {csv_dir}")
    print(f"  Stocks sampled:    {len(stock_dfs)}")
    print(f"  Stocks evaluated:  {len(all_results)}")
    print(f"  Stocks skipped:    {len(skipped)}")
    print(f"  Prediction length: {prediction_length}")
    print(f"  Num samples:       {model_cfg.get('num_samples', 100)}")
    print("-" * 60)
    for metric_name, value in global_agg.items():
        print(f"  {metric_name.upper():>8s}:  {value:.6f}")
    print("-" * 60)
    print(f"  Results saved to:  {exp_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
