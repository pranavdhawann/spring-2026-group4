"""
grid_search_gnn.py — Coarse hyperparameter grid search for Temporal GNN stock forecasting.

Usage:
    python scripts/grid_search_gnn.py --config config/config_gnn.yaml            # full coarse grid (~16 configs)
    python scripts/grid_search_gnn.py --config config/config_gnn.yaml --quick    # minimal grid for sanity testing (~4 configs)
    python scripts/grid_search_gnn.py --config config/config_gnn.yaml --best     # run best config with full epochs after grid

Design:
    - Uses a curated coarse grid (not full Cartesian product) to cover the
      most important dimensions with a manageable number of trials.
    - Each trial trains for a limited number of epochs (max_epochs=15) with
      aggressive early stopping (patience=5) to quickly identify promising configs.
    - Results are logged to tgnn/results/grid_search_results.csv after each trial
      (so results are preserved even if the search is interrupted).
    - The best config (by val_mae_log_return) is saved to tgnn/results/best_config_gnn.yaml.
    - A summary table is printed at the end.

Grid strategy:
    Rather than sweeping all combinations of all parameters, we use a
    "one-at-a-time with interaction" approach:
      1. A base config establishes the default.
      2. Individual parameters are swept while keeping others at default.
      3. A small set of multi-parameter combinations explores key interactions
         (e.g. high lr + high dropout; Huber loss + high gamma).
    This gives ~16 diverse configs that cover the most important dimensions
    without requiring 3×3×3×2×3×2×2 = 648 trials.

Parameters swept:
    - window_size:       [20, 40, 60, 90]   — critical for stock forecasting
    - lr:                [1e-4, 3e-4, 5e-4] — learning rate
    - dropout:           [0.1, 0.2, 0.3]
    - fusion_mode:       ["concat", "cross_attention"]
    - num_gnn_snapshots: [2, 4, 8]
    - loss_type:         ["mse", "huber"]
    - gamma:             [0.01, 0.1]        — directional loss weight

Walk-forward evaluation:
    The grid search uses the standard fixed val/test split for speed.
    Walk-forward (expanding window) can be enabled via --walk-forward flag
    for the final best-config rerun, at the cost of much longer runtime.
"""

import argparse
import copy
import csv
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Grid definitions
# ══════════════════════════════════════════════════════════════════════════════

def build_coarse_grid() -> List[Dict[str, Any]]:
    """Return the curated coarse grid as a list of override dicts.

    Each entry is a dict of {param_path: value} pairs where param_path
    uses dot notation for nested config keys (e.g. "data.window_size").

    Returns ~16 diverse configs that cover the key hyperparameter dimensions.
    """
    configs = []

    # ── 1. Baseline (default config, 15 epochs) ────────────────────────────
    configs.append({
        "_name": "baseline",
        "data.window_size": 60,
        "training.lr": 3e-4,
        "model.dropout": 0.1,
        "model.fusion_mode": "concat",
        "model.num_gnn_snapshots": 4,
        "training.loss_type": "huber",
        "training.loss_gamma": 0.1,
    })

    # ── 2. Window size sweep ───────────────────────────────────────────────
    for ws in [20, 40, 90]:
        configs.append({
            "_name": f"window_{ws}",
            "data.window_size": ws,
            "training.lr": 3e-4,
            "model.dropout": 0.1,
            "model.fusion_mode": "concat",
            "model.num_gnn_snapshots": 4,
            "training.loss_type": "huber",
            "training.loss_gamma": 0.1,
        })

    # ── 3. Learning rate sweep ─────────────────────────────────────────────
    for lr in [1e-4, 5e-4]:
        configs.append({
            "_name": f"lr_{lr:.0e}",
            "data.window_size": 60,
            "training.lr": lr,
            "model.dropout": 0.1,
            "model.fusion_mode": "concat",
            "model.num_gnn_snapshots": 4,
            "training.loss_type": "huber",
            "training.loss_gamma": 0.1,
        })

    # ── 4. Dropout sweep ──────────────────────────────────────────────────
    for dp in [0.2, 0.3]:
        configs.append({
            "_name": f"dropout_{dp}",
            "data.window_size": 60,
            "training.lr": 3e-4,
            "model.dropout": dp,
            "model.fusion_mode": "concat",
            "model.num_gnn_snapshots": 4,
            "training.loss_type": "huber",
            "training.loss_gamma": 0.1,
        })

    # ── 5. Fusion mode ────────────────────────────────────────────────────
    configs.append({
        "_name": "fusion_cross_attn",
        "data.window_size": 60,
        "training.lr": 3e-4,
        "model.dropout": 0.1,
        "model.fusion_mode": "cross_attention",
        "model.num_gnn_snapshots": 4,
        "training.loss_type": "huber",
        "training.loss_gamma": 0.1,
    })

    # ── 6. GNN snapshots sweep ────────────────────────────────────────────
    for ns in [2, 8]:
        configs.append({
            "_name": f"snapshots_{ns}",
            "data.window_size": 60,
            "training.lr": 3e-4,
            "model.dropout": 0.1,
            "model.fusion_mode": "concat",
            "model.num_gnn_snapshots": ns,
            "training.loss_type": "huber",
            "training.loss_gamma": 0.1,
        })

    # ── 7. Loss type comparison ────────────────────────────────────────────
    configs.append({
        "_name": "loss_mse",
        "data.window_size": 60,
        "training.lr": 3e-4,
        "model.dropout": 0.1,
        "model.fusion_mode": "concat",
        "model.num_gnn_snapshots": 4,
        "training.loss_type": "mse",
        "training.loss_gamma": 0.1,
    })

    # ── 8. Gamma (directional loss weight) ────────────────────────────────
    configs.append({
        "_name": "gamma_low",
        "data.window_size": 60,
        "training.lr": 3e-4,
        "model.dropout": 0.1,
        "model.fusion_mode": "concat",
        "model.num_gnn_snapshots": 4,
        "training.loss_type": "huber",
        "training.loss_gamma": 0.01,
    })

    # ── 9. Key interactions ───────────────────────────────────────────────
    # Long window + small snapshots (realistic for high-freq rebalancing)
    configs.append({
        "_name": "window90_snap2",
        "data.window_size": 90,
        "training.lr": 3e-4,
        "model.dropout": 0.2,
        "model.fusion_mode": "concat",
        "model.num_gnn_snapshots": 2,
        "training.loss_type": "huber",
        "training.loss_gamma": 0.1,
    })
    # Short window + cross-attention fusion
    configs.append({
        "_name": "window20_xattn",
        "data.window_size": 20,
        "training.lr": 3e-4,
        "model.dropout": 0.1,
        "model.fusion_mode": "cross_attention",
        "model.num_gnn_snapshots": 4,
        "training.loss_type": "huber",
        "training.loss_gamma": 0.1,
    })
    # High LR + high dropout (regularisation)
    configs.append({
        "_name": "lr5e4_drop03",
        "data.window_size": 60,
        "training.lr": 5e-4,
        "model.dropout": 0.3,
        "model.fusion_mode": "concat",
        "model.num_gnn_snapshots": 4,
        "training.loss_type": "huber",
        "training.loss_gamma": 0.1,
    })
    # MSE loss + low gamma (close to original setup to measure regressions)
    configs.append({
        "_name": "mse_gamma001",
        "data.window_size": 60,
        "training.lr": 3e-4,
        "model.dropout": 0.1,
        "model.fusion_mode": "concat",
        "model.num_gnn_snapshots": 4,
        "training.loss_type": "mse",
        "training.loss_gamma": 0.01,
    })

    return configs


def build_quick_grid() -> List[Dict[str, Any]]:
    """Return a minimal 4-config grid for quick sanity testing."""
    return [
        {
            "_name": "quick_baseline",
            "data.window_size": 60,
            "training.lr": 3e-4,
            "model.dropout": 0.1,
            "model.fusion_mode": "concat",
            "model.num_gnn_snapshots": 4,
            "training.loss_type": "huber",
            "training.loss_gamma": 0.1,
        },
        {
            "_name": "quick_window20",
            "data.window_size": 20,
            "training.lr": 3e-4,
            "model.dropout": 0.1,
            "model.fusion_mode": "concat",
            "model.num_gnn_snapshots": 4,
            "training.loss_type": "huber",
            "training.loss_gamma": 0.1,
        },
        {
            "_name": "quick_lr1e4",
            "data.window_size": 60,
            "training.lr": 1e-4,
            "model.dropout": 0.1,
            "model.fusion_mode": "concat",
            "model.num_gnn_snapshots": 4,
            "training.loss_type": "huber",
            "training.loss_gamma": 0.1,
        },
        {
            "_name": "quick_mse",
            "data.window_size": 60,
            "training.lr": 3e-4,
            "model.dropout": 0.2,
            "model.fusion_mode": "concat",
            "model.num_gnn_snapshots": 4,
            "training.loss_type": "mse",
            "training.loss_gamma": 0.01,
        },
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Config utilities
# ══════════════════════════════════════════════════════════════════════════════

def _set_nested(d: dict, dotted_key: str, value: Any):
    """Set a value in a nested dict using a dot-separated key path."""
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value


def _get_nested(d: dict, dotted_key: str, default: Any = None) -> Any:
    """Get a value from a nested dict using a dot-separated key path."""
    parts = dotted_key.split(".")
    for part in parts:
        if not isinstance(d, dict):
            return default
        d = d.get(part, default)
        if d is default:
            return default
    return d


def apply_overrides(base_config: dict, overrides: Dict[str, Any]) -> dict:
    """Return a deep copy of base_config with overrides applied."""
    config = copy.deepcopy(base_config)
    for key, value in overrides.items():
        if key.startswith("_"):
            continue  # skip metadata keys like _name
        _set_nested(config, key, value)
    return config


def config_to_row(trial_name: str, overrides: Dict[str, Any]) -> dict:
    """Convert a trial's overrides to a flat dict for CSV logging."""
    row = {"trial_name": trial_name}
    for k, v in overrides.items():
        if not k.startswith("_"):
            row[k] = v
    return row


# ══════════════════════════════════════════════════════════════════════════════
# Training wrapper
# ══════════════════════════════════════════════════════════════════════════════

def run_trial(
    trial_name: str,
    config: dict,
    max_epochs: int,
    patience: int,
) -> Dict[str, Any]:
    """Run a single training trial and return metrics.

    Args:
        trial_name:  Human-readable name for this trial (for logging).
        config:      Full config dict with trial overrides already applied.
        max_epochs:  Maximum epochs for this trial.
        patience:    Early stopping patience.

    Returns:
        dict with keys: val_mae, val_dir_acc, val_loss, best_epoch,
                        train_time_s, error (if failed).
    """
    # Apply grid-search-specific training settings
    config = copy.deepcopy(config)
    config["training"]["max_epochs"] = max_epochs
    config["training"]["early_stopping_patience"] = patience

    # Suppress wandb during grid search (avoid cluttering offline runs)
    if "logging" in config:
        config["logging"]["wandb_mode"] = "disabled"

    t0 = time.time()
    try:
        # Import here so environment is set up by the time we import
        from src.dataset_gnn import build_dataloaders
        from src.loss_gnn import CombinedLoss
        from src.model_gnn import TemporalGNN
        from src.utils_gnn import count_parameters, get_device, set_seed
        from train import (
            train,
            CosineWithWarmup,
            make_grad_scaler,
            amp_autocast,
            train_one_step,
            validate,
        )

        _, best_metric = train(config, run_name=f"grid_{trial_name}")
        elapsed = time.time() - t0

        return {
            "trial_name": trial_name,
            "val_mae": best_metric,
            "train_time_s": round(elapsed, 1),
            "status": "ok",
            "error": None,
        }
    except Exception as exc:
        elapsed = time.time() - t0
        logger.error("Trial %s FAILED after %.1fs: %s", trial_name, elapsed, exc, exc_info=True)
        return {
            "trial_name": trial_name,
            "val_mae": float("inf"),
            "train_time_s": round(elapsed, 1),
            "status": "error",
            "error": str(exc),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward evaluation
# ══════════════════════════════════════════════════════════════════════════════

def run_walk_forward(
    config: dict,
    n_folds: int = 3,
    max_epochs: int = 15,
    patience: int = 5,
) -> Dict[str, Any]:
    """Expanding window walk-forward evaluation.

    Trains on progressively larger windows and evaluates on the next
    ``val_days`` period.  Returns average val MAE across all folds.

    Walk-forward setup:
        Fold 1: train=[start, fold1_end], val=[fold1_end+purge, fold1_end+purge+val_days]
        Fold 2: train=[start, fold2_end], val=[fold2_end+purge, fold2_end+purge+val_days]
        ...
    The last fold's val window should not exceed the total data range.

    This is implemented by running the standard train() once per fold,
    adjusting test_days to 0 and val_days to control the eval window.
    Because the dataset always uses the LAST val_days as val, we can
    approximate expanding windows by changing the effective train length
    via a multiplier on val_days.

    Note: Full walk-forward with proper per-fold data splits would require
    dataset changes.  This approximation (adjusting split sizes and retraining)
    captures the key validation robustness check at the cost of some leakage
    (later folds use test data as training, which should be acceptable for
    hyperparameter selection but not for final reporting).
    """
    import copy
    fold_maes = []
    for fold in range(n_folds):
        fold_config = copy.deepcopy(config)
        # Each fold uses a smaller val window to simulate expanding training
        # set: fold 0 = earliest, fold n_folds-1 = most recent (closest to
        # the standard fixed split).  We shift the split by reducing val+test
        # so that the training set covers progressively more data.
        offset_multiplier = n_folds - fold  # earlier folds have more offset
        fold_val_days = config["split"].get("val_days", 90)
        fold_config["split"]["test_days"] = 0
        # Move the val window earlier by an offset proportional to fold index
        extra_val_offset = fold_val_days * fold
        fold_config["split"]["val_days"] = fold_val_days + extra_val_offset
        fold_config["training"]["max_epochs"] = max_epochs
        fold_config["training"]["early_stopping_patience"] = patience
        fold_config["logging"]["wandb_mode"] = "disabled"

        logger.info("Walk-forward fold %d/%d | val_offset=%d days", fold + 1, n_folds, extra_val_offset)
        try:
            from train import train
            _, best_mae = train(fold_config, run_name=f"wf_fold{fold+1}")
            fold_maes.append(best_mae)
            logger.info("Fold %d val MAE: %.6f", fold + 1, best_mae)
        except Exception as exc:
            logger.error("Walk-forward fold %d FAILED: %s", fold + 1, exc)
            fold_maes.append(float("inf"))

    mean_mae = float(np.mean(fold_maes))
    std_mae = float(np.std(fold_maes))
    logger.info(
        "Walk-forward result | folds=%d | mean_MAE=%.6f | std=%.6f | per_fold=%s",
        n_folds, mean_mae, std_mae,
        [f"{m:.6f}" for m in fold_maes],
    )
    return {
        "walk_forward_mean_mae": mean_mae,
        "walk_forward_std_mae": std_mae,
        "fold_maes": fold_maes,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Results I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_results_csv(results: List[Dict[str, Any]], csv_path: str):
    """Append results to CSV, writing header if file doesn't exist."""
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)


def save_best_config(config: dict, path: str):
    """Save the best config as YAML."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Best config saved to %s", path)


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a formatted summary table of all grid search results."""
    # Sort by val_mae ascending (lower is better)
    sorted_results = sorted(results, key=lambda r: r.get("val_mae", float("inf")))

    header_cols = [
        ("Rank", 4),
        ("Trial Name", 24),
        ("window", 8),
        ("lr", 8),
        ("dropout", 8),
        ("fusion", 14),
        ("snaps", 6),
        ("loss", 6),
        ("gamma", 7),
        ("val_MAE", 10),
        ("time(s)", 8),
        ("status", 7),
    ]

    def fmt_header():
        return " | ".join(h.ljust(w) for h, w in header_cols)

    def fmt_row(rank, r):
        fields = [
            str(rank).ljust(4),
            r.get("trial_name", "?")[:24].ljust(24),
            str(r.get("data.window_size", "?")).ljust(8),
            f"{r.get('training.lr', '?'):.0e}".ljust(8) if isinstance(r.get("training.lr"), float) else str(r.get("training.lr", "?")).ljust(8),
            str(r.get("model.dropout", "?")).ljust(8),
            str(r.get("model.fusion_mode", "?"))[:14].ljust(14),
            str(r.get("model.num_gnn_snapshots", "?")).ljust(6),
            str(r.get("training.loss_type", "?"))[:6].ljust(6),
            str(r.get("training.loss_gamma", "?")).ljust(7),
            f"{r.get('val_mae', float('inf')):.6f}".ljust(10),
            str(r.get("train_time_s", "?")).ljust(8),
            str(r.get("status", "?")).ljust(7),
        ]
        return " | ".join(fields)

    sep = "-" * (sum(w for _, w in header_cols) + 3 * len(header_cols))
    print("\n" + "=" * len(sep))
    print("  GRID SEARCH RESULTS")
    print("=" * len(sep))
    print(fmt_header())
    print(sep)
    for rank, r in enumerate(sorted_results, 1):
        print(fmt_row(rank, r))
    print(sep)
    if sorted_results:
        best = sorted_results[0]
        print(f"\n  BEST: {best['trial_name']} | val_MAE = {best.get('val_mae', float('inf')):.6f}")
    print("=" * len(sep) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter grid search for Temporal GNN stock forecasting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", default=os.path.join("config", "config_gnn.yaml"),
        help="Base config file (default: config/config_gnn.yaml)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a minimal 4-config grid for quick sanity testing.",
    )
    parser.add_argument(
        "--best", action="store_true",
        help="After the grid search, retrain the best config with full epochs.",
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Use walk-forward (expanding window) evaluation for the best-config rerun.",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override max epochs per trial (default: from grid_search config or 15).",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Override early stopping patience per trial (default: 5).",
    )
    parser.add_argument(
        "--results-csv", default=None,
        help="Path for results CSV (default: from config or tgnn/results/grid_search_results.csv).",
    )
    parser.add_argument(
        "--best-config", default=None,
        help="Path to save best config YAML (default: from config or tgnn/results/best_config_gnn.yaml).",
    )
    parser.add_argument(
        "--skip-if-exists", action="store_true",
        help="Skip trials whose name already appears in the results CSV.",
    )
    args = parser.parse_args()

    # ── Load base config ───────────────────────────────────────────────────
    from src.utils_gnn import load_config
    base_config = load_config(args.config)
    grid_cfg = base_config.get("grid_search", {})

    max_epochs = args.max_epochs or grid_cfg.get("max_epochs", 15)
    patience = args.patience or grid_cfg.get("early_stopping_patience", 5)
    results_csv = args.results_csv or grid_cfg.get("results_csv", os.path.join("tgnn", "results", "grid_search_results.csv"))
    best_config_path = args.best_config or grid_cfg.get("best_config_path", os.path.join("tgnn", "results", "best_config_gnn.yaml"))

    # ── Build grid ─────────────────────────────────────────────────────────
    if args.quick:
        grid = build_quick_grid()
        logger.info("Running QUICK grid (%d configs, max_epochs=%d, patience=%d)",
                    len(grid), max_epochs, patience)
    else:
        grid = build_coarse_grid()
        logger.info("Running COARSE grid (%d configs, max_epochs=%d, patience=%d)",
                    len(grid), max_epochs, patience)

    # ── Load existing results for --skip-if-exists ─────────────────────────
    completed_names: set = set()
    if args.skip_if_exists and os.path.isfile(results_csv):
        with open(results_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_names.add(row.get("trial_name", ""))
        logger.info("Loaded %d completed trials from %s", len(completed_names), results_csv)

    # ── Run trials ─────────────────────────────────────────────────────────
    all_results: List[Dict[str, Any]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for trial_idx, overrides in enumerate(grid, 1):
        trial_name = overrides.get("_name", f"trial_{trial_idx}")

        if trial_name in completed_names:
            logger.info("Skipping trial %s (already in results CSV)", trial_name)
            continue

        logger.info(
            "\n%s\n  Trial %d/%d: %s\n%s",
            "=" * 70, trial_idx, len(grid), trial_name, "=" * 70,
        )
        logger.info("  Overrides: %s", {k: v for k, v in overrides.items() if not k.startswith("_")})

        # Apply overrides to base config
        trial_config = apply_overrides(base_config, overrides)

        # Run the trial
        result = run_trial(trial_name, trial_config, max_epochs=max_epochs, patience=patience)

        # Merge override params into result for CSV
        flat_overrides = {k: v for k, v in overrides.items() if not k.startswith("_")}
        result.update(flat_overrides)
        result["timestamp"] = timestamp

        logger.info(
            "  Trial %s finished | val_MAE=%.6f | time=%.1fs | status=%s",
            trial_name, result.get("val_mae", float("inf")),
            result.get("train_time_s", 0), result.get("status", "?"),
        )

        all_results.append(result)
        # Append to CSV immediately (fault-tolerant)
        save_results_csv([result], results_csv)

    if not all_results:
        logger.warning("No trials were run (all skipped or grid is empty).")
        return

    # ── Print summary table ────────────────────────────────────────────────
    print_summary_table(all_results)

    # ── Find best config ───────────────────────────────────────────────────
    valid_results = [r for r in all_results if r.get("status") == "ok" and r.get("val_mae", float("inf")) < float("inf")]
    if not valid_results:
        logger.error("All trials failed — cannot determine best config.")
        sys.exit(1)

    best_result = min(valid_results, key=lambda r: r["val_mae"])
    best_trial_name = best_result["trial_name"]
    logger.info("Best trial: %s | val_MAE=%.6f", best_trial_name, best_result["val_mae"])

    # Find the overrides for the best trial
    best_overrides = next((o for o in grid if o.get("_name") == best_trial_name), {})
    best_config = apply_overrides(base_config, best_overrides)
    save_best_config(best_config, best_config_path)

    # ── Walk-forward evaluation on best config ─────────────────────────────
    if args.walk_forward:
        logger.info("\nRunning walk-forward evaluation on best config: %s", best_trial_name)
        wf_result = run_walk_forward(
            config=best_config,
            n_folds=3,
            max_epochs=max_epochs,
            patience=patience,
        )
        logger.info(
            "Walk-forward | mean_MAE=%.6f | std=%.6f",
            wf_result["walk_forward_mean_mae"],
            wf_result["walk_forward_std_mae"],
        )
        wf_csv_row = {
            "trial_name": f"{best_trial_name}_walk_forward",
            "val_mae": wf_result["walk_forward_mean_mae"],
            "val_mae_std": wf_result["walk_forward_std_mae"],
            "status": "walk_forward",
            "timestamp": timestamp,
        }
        wf_csv_row.update({k: v for k, v in best_overrides.items() if not k.startswith("_")})
        save_results_csv([wf_csv_row], results_csv)

    # ── Full rerun with best config ────────────────────────────────────────
    if args.best:
        logger.info(
            "\n%s\n  Full training with best config: %s\n%s",
            "=" * 70, best_trial_name, "=" * 70,
        )
        full_config = copy.deepcopy(best_config)
        # Restore full epoch count from base config
        full_config["training"]["max_epochs"] = base_config["training"].get("max_epochs", 100)
        full_config["training"]["early_stopping_patience"] = base_config["training"].get("early_stopping_patience", 10)
        if "logging" in full_config:
            full_config["logging"]["wandb_mode"] = base_config.get("logging", {}).get("wandb_mode", "offline")

        from train import train
        _, best_full_metric = train(full_config, run_name=f"best_{best_trial_name}")
        logger.info("Full training complete | val_MAE=%.6f", best_full_metric)

        full_row = {
            "trial_name": f"{best_trial_name}_full",
            "val_mae": best_full_metric,
            "max_epochs": full_config["training"]["max_epochs"],
            "status": "full_run",
            "timestamp": timestamp,
        }
        full_row.update({k: v for k, v in best_overrides.items() if not k.startswith("_")})
        save_results_csv([full_row], results_csv)

    logger.info("\nGrid search complete.")
    logger.info("  Results CSV:  %s", os.path.abspath(results_csv))
    logger.info("  Best config:  %s", os.path.abspath(best_config_path))


if __name__ == "__main__":
    main()
