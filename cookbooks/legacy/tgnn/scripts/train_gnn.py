"""
train_gnn.py — Training loop with correct data flow through model.
"""

import argparse
import logging
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_gnn import build_dataloaders
from src.loss_gnn import CombinedLoss
from src.model_gnn import TemporalGNN
from src.utils_gnn import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_TENSORBOARD_DIR,
    count_parameters,
    get_device,
    load_config,
    log_runtime_context,
    save_checkpoint,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def make_grad_scaler(device_type: str, enabled: bool):
    """Create a GradScaler without triggering deprecated CUDA AMP APIs."""
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device_type)
    from torch.cuda.amp import GradScaler

    return GradScaler()


def amp_autocast(device_type: str, enabled: bool):
    """Compatibility wrapper around torch.amp.autocast."""
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device_type, enabled=True)
    from torch.cuda.amp import autocast

    return autocast(enabled=True)


class CosineWithWarmup:
    def __init__(self, optimizer, warmup_steps, total_steps, lr_min=1e-6):
        self.optimizer, self.warmup_steps, self.total_steps = (
            optimizer,
            warmup_steps,
            total_steps,
        )
        self.lr_min = lr_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            scale = self.step_count / max(1, self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = 0.5 * (1 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.lr_min, base_lr * scale)

    def state_dict(self):
        return {"step_count": self.step_count}

    def load_state_dict(self, s):
        self.step_count = s["step_count"]


def train_one_step(model, sample, criterion, device):
    """Forward pass through the full model pipeline and compute loss."""
    if sample["num_active"] == 0:
        return None, None

    # Model.forward now takes the raw sample dict and runs all encoders internally
    output = model(sample, device=device)

    pred_lr = output["log_returns"]
    pred_close = output["pred_close"]
    direction_logits = output.get("direction_logits")
    targets = sample["targets"].to(device)
    target_close = sample["target_close"].to(device)

    loss, components = criterion(
        pred_lr, targets, pred_close, target_close, direction_logits=direction_logits
    )
    return loss, components, pred_lr.detach()


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_pred_lr, all_target_lr = [], []
    component_sums = {}

    for sample in val_loader:
        if isinstance(sample, list):
            sample = sample[0]
        if sample["num_active"] == 0:
            continue

        output = model(sample, device=device)
        pred_lr = output["log_returns"]
        targets = sample["targets"].to(device)
        target_close = sample["target_close"].to(device)

        direction_logits = output.get("direction_logits")
        loss, components = criterion(
            pred_lr,
            targets,
            output["pred_close"],
            target_close,
            direction_logits=direction_logits,
        )
        if loss is None:
            continue

        n = targets.size(0)
        total_loss += loss.item() * n
        total_samples += n
        for k, v in components.items():
            component_sums[k] = component_sums.get(k, 0.0) + v * n

        all_pred_lr.append(pred_lr.cpu())
        all_target_lr.append(targets.cpu())

    if total_samples == 0:
        return {"val_loss": float("inf"), "val_mae_log_return": float("inf")}

    metrics = {"val_loss": total_loss / total_samples}
    for k, v in component_sums.items():
        metrics[f"val_{k}"] = v / total_samples

    if all_pred_lr:
        pred_cat = torch.cat(all_pred_lr)
        target_cat = torch.cat(all_target_lr)
        metrics["val_mae_log_return"] = (pred_cat - target_cat).abs().mean().item()
        for h in range(pred_cat.size(1)):
            metrics[f"val_mae_h{h+1}"] = (
                (pred_cat[:, h] - target_cat[:, h]).abs().mean().item()
            )
        metrics["val_directional_acc"] = (
            ((pred_cat > 0) == (target_cat > 0)).float().mean().item()
        )
        # Collapse diagnostics: if pred_std ≪ target_std the model is predicting a constant.
        metrics["val_pred_mean"] = pred_cat.mean().item()
        metrics["val_pred_std"] = pred_cat.std().item()
        metrics["val_target_std"] = target_cat.std().item()
        # Pearson correlation — MAE alone hides constant predictors.
        pc = pred_cat.flatten() - pred_cat.mean()
        tc = target_cat.flatten() - target_cat.mean()
        denom = (pc.norm() * tc.norm()).clamp(min=1e-8)
        metrics["val_corr"] = (pc @ tc / denom).item()

    return metrics


def init_tracker(config, run_name="default"):
    log_cfg = config.get("logging", {})
    backend = log_cfg.get("backend", "tensorboard")
    if backend == "wandb":
        wandb_mode = str(
            os.getenv("WANDB_MODE", log_cfg.get("wandb_mode", "offline"))
        ).lower()
        if wandb_mode not in {"online", "offline", "disabled"}:
            logger.warning("Unknown W&B mode '%s'; defaulting to offline", wandb_mode)
            wandb_mode = "offline"
        if log_cfg.get("wandb_silent", True):
            os.environ.setdefault("WANDB_SILENT", "true")
        if wandb_mode == "disabled":
            logger.info("Weights & Biases tracking disabled via config")
            return None
        try:
            import wandb

            wandb.init(
                project=log_cfg.get("project", "tgnn-stock-forecast"),
                config=config,
                name=run_name,
                mode=wandb_mode,
            )
            logger.info(
                "Initialized Weights & Biases tracker with run name %s in %s mode",
                run_name,
                wandb_mode,
            )
            return "wandb"
        except ImportError:
            logger.warning(
                "wandb requested but not installed; continuing without wandb tracking"
            )
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = os.path.join(DEFAULT_TENSORBOARD_DIR, run_name)
        logger.info("Initialized TensorBoard tracker at %s", tb_dir)
        return SummaryWriter(log_dir=tb_dir)
    except ImportError:
        logger.warning(
            "No experiment tracker available (wandb/tensorboard unavailable)"
        )
        return None


def log_metrics(tracker, metrics, step, prefix=""):
    if tracker == "wandb":
        import wandb

        wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)
    elif hasattr(tracker, "add_scalar"):
        for k, v in metrics.items():
            tracker.add_scalar(f"{prefix}{k}", v, step)


def train(config, run_name="default"):
    seed = config.get("seed", 42)
    set_seed(seed, config.get("deterministic", True))
    device = get_device()
    train_cfg = config["training"]
    log_every_n_steps = max(1, config.get("logging", {}).get("log_every_n_steps", 10))

    logger.info(
        "Training configuration | seed=%s | max_epochs=%s | lr=%s | grad_accum=%s | mixed_precision=%s",
        seed,
        train_cfg.get("max_epochs", 100),
        train_cfg.get("lr", 1e-4),
        train_cfg.get("grad_accumulation_steps", 8),
        train_cfg.get("mixed_precision", True),
    )

    logger.info("Building data loaders...")
    train_loader, val_loader, test_loader, metadata = build_dataloaders(config)
    logger.info(
        f"Train: {metadata['train_samples']}, Val: {metadata['val_samples']}, Test: {metadata['test_samples']}"
    )

    # Initialize model and compute report normalization stats
    model = TemporalGNN(config, max_nodes=metadata.get("max_nodes", 550)).to(device)
    if metadata.get("fundamentals"):
        model.reports_encoder.compute_normalization_stats(metadata["fundamentals"])
    count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-4),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    grad_accum = train_cfg.get("grad_accumulation_steps", 8)

    # FIX: total_steps must be *optimizer steps*, not raw samples × epochs.
    # Each optimizer step consumes grad_accum samples, so:
    #   steps_per_epoch ≈ ceil(train_samples / grad_accum)
    #   total_steps     = steps_per_epoch × max_epochs
    # The old code used (train_samples × max_epochs) which was ~8× too large,
    # making warmup last several epochs and the cosine decay far too slow.
    steps_per_epoch = int(np.ceil(metadata["train_samples"] / grad_accum))
    total_steps = steps_per_epoch * train_cfg.get("max_epochs", 100)
    scheduler = CosineWithWarmup(
        optimizer,
        int(total_steps * train_cfg.get("warmup_ratio", 0.05)),
        total_steps,
        train_cfg.get("lr_min", 1e-6),
    )

    # NOTE: batch_size=1 (one date per sample) with grad_accumulation_steps
    # gives effective_batch = grad_accum dates, each containing ~100 stocks.
    # E.g. grad_accum=4 → effective_batch=4 dates × 100 stocks = 400 stock-date pairs.
    criterion = CombinedLoss(
        alpha=train_cfg.get("loss_alpha", 1.0),
        beta=train_cfg.get("loss_beta", 0.1),
        gamma=train_cfg.get("loss_gamma", 0.1),
        delta=train_cfg.get("loss_delta", 0.1),
        direction_loss_type=train_cfg.get("direction_loss_type", "weighted_bce"),
        loss_type=train_cfg.get("loss_type", "huber"),
        huber_delta=train_cfg.get("huber_delta", 0.01),
        label_smoothing=train_cfg.get("label_smoothing", 0.05),
        direction_logit_scale=train_cfg.get("direction_logit_scale", 100.0),
    )

    amp_device_type = device.type if isinstance(device, torch.device) else str(device)
    use_amp = (
        train_cfg.get("mixed_precision", True)
        and amp_device_type == "cuda"
        and torch.cuda.is_available()
    )
    scaler = make_grad_scaler(amp_device_type, use_amp)
    max_grad_norm = train_cfg.get("gradient_clip", 1.0)
    patience = train_cfg.get("early_stopping_patience", 10)
    es_metric = train_cfg.get("early_stopping_metric", "val_mae_log_return")
    best_metric, patience_counter = float("inf"), 0

    tracker = init_tracker(config, run_name=run_name)
    ckpt_dir = DEFAULT_CHECKPOINT_DIR
    os.makedirs(ckpt_dir, exist_ok=True)
    max_epochs = train_cfg.get("max_epochs", 100)
    global_step = 0
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.05))
    logger.info("=" * 70)
    logger.info("TRAINING PLAN")
    logger.info("=" * 70)
    logger.info(
        "Dataset      | train_samples=%d | val_samples=%d | test_samples=%d",
        metadata["train_samples"],
        metadata["val_samples"],
        metadata["test_samples"],
    )
    logger.info(
        "Batching     | batch_size=1 (one date/sample) | grad_accum=%d | "
        "effective_batch=%d dates x ~%d stocks = ~%d stock-date pairs",
        grad_accum,
        grad_accum,
        metadata.get("num_tickers", 100),
        grad_accum * metadata.get("num_tickers", 100),
    )
    logger.info(
        "Schedule     | steps_per_epoch=%d | total_steps=%d | warmup_steps=%d (%.1f%%) | max_epochs=%d",
        steps_per_epoch,
        total_steps,
        warmup_steps,
        100.0 * warmup_steps / max(1, total_steps),
        max_epochs,
    )
    logger.info(
        "LR           | peak=%.2e | min=%.2e | warmup_ratio=%.3f | schedule=cosine",
        train_cfg.get("lr", 1e-4),
        train_cfg.get("lr_min", 1e-6),
        train_cfg.get("warmup_ratio", 0.05),
    )
    logger.info(
        "Optimizer    | AdamW | weight_decay=%.1e | gradient_clip=%.1f | mixed_precision=%s",
        train_cfg.get("weight_decay", 1e-5),
        max_grad_norm,
        use_amp,
    )
    logger.info(
        "Early stop   | patience=%d | metric=%s | checkpoint_dir=%s",
        patience,
        es_metric,
        os.path.abspath(ckpt_dir),
    )
    logger.info(
        "Tickers      | num_tickers=%d | max_nodes=%d",
        metadata["num_tickers"],
        metadata.get("max_nodes", 550),
    )
    logger.info(
        "Loss         | type=%s | huber_delta=%.4f | alpha=%.2f | beta=%.2f | gamma=%.3f | "
        "delta=%.3f | dir_type=%s | label_smoothing=%.3f",
        train_cfg.get("loss_type", "huber"),
        train_cfg.get("huber_delta", 0.01),
        train_cfg.get("loss_alpha", 1.0),
        train_cfg.get("loss_beta", 0.1),
        train_cfg.get("loss_gamma", 0.1),
        train_cfg.get("loss_delta", 0.1),
        train_cfg.get("direction_loss_type", "weighted_bce"),
        train_cfg.get("label_smoothing", 0.05),
    )
    logger.info("=" * 70)

    for epoch in range(1, max_epochs + 1):
        epoch_t0 = time.time()
        model.train()
        epoch_loss, epoch_samples, epoch_components = 0.0, 0, {}
        optimizer.zero_grad()
        accum_count = 0
        skipped_empty = 0
        skipped_none = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False)
        for batch_idx, sample in enumerate(pbar, 1):
            if isinstance(sample, list):
                sample = sample[0]
            if sample["num_active"] == 0:
                skipped_empty += 1
                continue

            if use_amp:
                with amp_autocast(amp_device_type, enabled=True):
                    loss, components, pred_lr_dbg = train_one_step(
                        model, sample, criterion, device
                    )
            else:
                loss, components, pred_lr_dbg = train_one_step(
                    model, sample, criterion, device
                )

            if loss is None:
                skipped_none += 1
                continue

            loss_scaled = loss / grad_accum
            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            accum_count += 1
            n = sample["targets"].size(0)
            epoch_loss += loss.item() * n
            epoch_samples += n
            for k, v in components.items():
                epoch_components[k] = epoch_components.get(k, 0.0) + v * n

            if accum_count >= grad_accum:
                if use_amp:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm
                    )
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                accum_count = 0
                global_step += 1
                if global_step % log_every_n_steps == 0:
                    logger.info(
                        "Step %d | epoch=%d | loss=%.5f | lr=%.3e | grad_norm=%.3f | "
                        "lr_l=%.5f price_l=%.5f dir_l=%.5f var_l=%.5f | "
                        "pred_mean=%+.2e pred_std=%.2e | tgt_std=%.2e",
                        global_step,
                        epoch,
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                        float(grad_norm),
                        components.get("lr_loss", 0.0),
                        components.get("price_loss", 0.0),
                        components.get("direction_loss", 0.0),
                        components.get("var_loss", 0.0),
                        float(pred_lr_dbg.float().mean()),
                        float(pred_lr_dbg.float().std()),
                        float(sample["targets"].float().std()),
                    )

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Flush remaining gradients
        if accum_count > 0:
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        train_loss = epoch_loss / epoch_samples if epoch_samples > 0 else float("inf")
        val_metrics = validate(model, val_loader, criterion, device)
        epoch_elapsed = time.time() - epoch_t0

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_MAE=%.4f | "
            "dir_acc=%.2f%% | corr=%+.4f | pred_std=%.2e tgt_std=%.2e | "
            "lr=%.2e | time=%.1fs",
            epoch,
            max_epochs,
            train_loss,
            val_metrics.get("val_loss", float("inf")),
            val_metrics.get("val_mae_log_return", float("inf")),
            val_metrics.get("val_directional_acc", 0) * 100,
            val_metrics.get("val_corr", 0.0),
            val_metrics.get("val_pred_std", 0.0),
            val_metrics.get("val_target_std", 0.0),
            optimizer.param_groups[0]["lr"],
            epoch_elapsed,
        )
        # Per-component train losses at epoch end (collapse diagnostics).
        if epoch_samples > 0 and epoch_components:
            logger.info(
                "Epoch %d train components | lr_loss=%.5f | price_loss=%.5f | direction_loss=%.5f | var_loss=%.5f",
                epoch,
                epoch_components.get("lr_loss", 0.0) / epoch_samples,
                epoch_components.get("price_loss", 0.0) / epoch_samples,
                epoch_components.get("direction_loss", 0.0) / epoch_samples,
                epoch_components.get("var_loss", 0.0) / epoch_samples,
            )

        if tracker:
            log_metrics(
                tracker,
                {**{"train_loss": train_loss}, **val_metrics},
                global_step,
                "epoch/",
            )

        current = val_metrics.get(es_metric, float("inf"))
        # FIX E9 (save side): strip numpy scalars from val_metrics before
        # pickling so checkpoints can be loaded with weights_only=True in
        # principle (and definitely with weights_only=False on any
        # PyTorch version).
        val_metrics_clean = {
            k: (float(v) if isinstance(v, (np.floating, np.integer, np.ndarray)) else v)
            for k, v in val_metrics.items()
        }
        if current < best_metric:
            best_metric, patience_counter = current, 0
            save_checkpoint(
                model,
                optimizer,
                epoch,
                float(current),
                os.path.join(ckpt_dir, "best.pt"),
                scaler=scaler,
                extra={"config": config, "val_metrics": val_metrics_clean},
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        logger.info(
            "Epoch %d bookkeeping | epoch_samples=%d | skipped_empty=%d | skipped_none=%d | patience=%d/%d",
            epoch,
            epoch_samples,
            skipped_empty,
            skipped_none,
            patience_counter,
            patience,
        )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            float(current),
            os.path.join(ckpt_dir, "last.pt"),
            scaler=scaler,
        )

    if tracker == "wandb":
        import wandb

        wandb.finish()
    elif hasattr(tracker, "close"):
        tracker.close()

    # Clear correlation edge caches so the next seed run starts fresh
    if hasattr(train_loader.dataset, "graph_builder"):
        train_loader.dataset.graph_builder.corr_builder.clear_cache()

    logger.info(f"Training complete. Best {es_metric}: {best_metric:.6f}")
    return model, best_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    log_path = setup_logging(
        config, command_name="train", config_path=args.config, args=args
    )
    logger.info("Loaded config from %s", os.path.abspath(args.config))
    log_runtime_context("train", config, extra={"train_log_path": log_path})

    num_seeds = config.get("num_seed_runs", 1)
    base_seed = config.get("seed", 42)
    results = []
    for i in range(num_seeds):
        config["seed"] = base_seed + i
        logger.info(
            f"\n{'='*60}\nSeed run {i+1}/{num_seeds} (seed={config['seed']})\n{'='*60}"
        )
        _, metric = train(config, run_name=f"seed_{config['seed']}")
        results.append(metric)

    if num_seeds > 1:
        logger.info(f"\nResults: {np.mean(results):.6f} ± {np.std(results):.6f}")


if __name__ == "__main__":
    main()
