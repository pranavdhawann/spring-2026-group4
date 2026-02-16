"""
Training script for LSTM multi-stock forecasting baseline.

Usage:
    python -m train.baseline_train_lstm
    python -m train.baseline_train_lstm --epochs 50 --batch_size 32
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.timeseries_lstm import LSTMForecaster
from src.preProcessing.data_preprocessing_lstm import prepare_lstm_data
from src.utils import (
    calculate_regression_metrics,
    print_metrics,
    read_yaml,
    set_seed,
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with overrides for YAML config."""
    parser = argparse.ArgumentParser(
        description="Train LSTM forecasting baseline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/lstm_baseline.yaml",
        help="Path to LSTM config YAML",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:

    main_config = read_yaml("config/config.yaml")
    lstm_config = read_yaml(args.config)

    config = {}
    config.update(main_config)
    config.update(lstm_config)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.hidden_size is not None:
        config["hidden_size"] = args.hidden_size
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.device is not None:
        config["device"] = args.device

    return config

class EarlyStopping:
    """
    Early stopping handler that monitors validation loss.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.should_stop = False

    def __call__(
        self,
        val_loss: float,
        epoch: int,
        model: nn.Module,
        save_path: Path,
    ) -> bool:
        """
        Check if training should stop.

        Returns True if patience exceeded. Also saves the best model.
        """
        if self.best_score is None or val_loss < self.best_score - self.min_delta:
            self.best_score = val_loss
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip_norm: float = 1.0,
) -> float:
    """
    Train for one epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, targets)
        loss.backward()

        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_scaler=None,
) -> Dict:
    """
    Evaluate model on a DataLoader.

    Args:
        model: LSTMForecaster
        dataloader: Val or Test DataLoader
        criterion: Loss function
        device: torch device
        target_scaler: StandardScaler for inverse transform (optional)

    Returns:
        dict with:
            loss: average loss (normalised space)
            metrics: dict from calculate_regression_metrics (original scale)
            all_preds: np.ndarray of all predictions (original scale)
            all_targets: np.ndarray of all targets (original scale)
            per_stock: dict mapping ticker_id -> (preds, targets)
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_targets = []
    all_ticker_ids = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)

            preds = model(features)
            loss = criterion(preds, targets)

            total_loss += loss.item()
            n_batches += 1

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_ticker_ids.append(batch["ticker_id"].numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_ticker_ids = np.concatenate(all_ticker_ids, axis=0)

    preds_orig = all_preds
    targets_orig = all_targets
    if target_scaler is not None:
        preds_orig = target_scaler.inverse_transform(
            all_preds.reshape(-1, 1)
        ).reshape(all_preds.shape)
        targets_orig = target_scaler.inverse_transform(
            all_targets.reshape(-1, 1)
        ).reshape(all_targets.shape)

    metrics = calculate_regression_metrics(targets_orig, preds_orig)

    per_stock = {}
    unique_ids = np.unique(all_ticker_ids)
    for tid in unique_ids:
        mask = all_ticker_ids == tid
        per_stock[int(tid)] = (preds_orig[mask], targets_orig[mask])

    return {
        "loss": total_loss / max(n_batches, 1),
        "metrics": metrics,
        "all_preds": preds_orig,
        "all_targets": targets_orig,
        "per_stock": per_stock,
    }


def create_visualizations(
    training_history: List[dict],
    per_stock_preds: Dict,
    output_dir: Path,
    ticker_id_to_name: Dict[int, str],
    top_tickers: List[str],
    max_stocks: int = 10,
) -> None:
    """
    Generate and save training visualisations.

    Args:
        training_history: list of dicts with epoch, train_loss, val_loss
        per_stock_preds: dict mapping ticker_id -> (preds, targets)
        output_dir: directory to save PNGs
        ticker_id_to_name: mapping from ticker_id int to ticker string
        top_tickers: list of top ticker names (by quality score) for viz
        max_stocks: maximum number of stocks to plot
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available, skipping visualisations.")
        return

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in training_history]
    train_losses = [h["train_loss"] for h in training_history]
    val_losses = [h["val_loss"] for h in training_history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    ax.plot(epochs, val_losses, label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(viz_dir / "loss_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {viz_dir / 'loss_curves.png'}")

    name_to_id = {v: k for k, v in ticker_id_to_name.items()}

    stocks_to_plot = []
    for ticker in top_tickers:
        tid = name_to_id.get(ticker)
        if tid is not None and tid in per_stock_preds:
            stocks_to_plot.append((ticker, tid))
        if len(stocks_to_plot) >= max_stocks:
            break

    if not stocks_to_plot:
        for tid, (preds, tgts) in list(per_stock_preds.items())[:max_stocks]:
            name = ticker_id_to_name.get(tid, f"ID_{tid}")
            stocks_to_plot.append((name, tid))

    for ticker_name, ticker_id in stocks_to_plot:
        preds, targets = per_stock_preds[ticker_id]

        n_samples = min(len(preds), 50)  
        preds_flat = preds[:n_samples].flatten()
        targets_flat = targets[:n_samples].flatten()
        x_axis = np.arange(len(preds_flat))

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        axes[0].plot(x_axis, targets_flat, label="Actual", linewidth=1.5, alpha=0.8)
        axes[0].plot(x_axis, preds_flat, label="Predicted", linewidth=1.5, alpha=0.8)
        axes[0].set_title(f"{ticker_name} - Predicted vs Actual Stock Price")
        axes[0].set_xlabel("Time Step")
        axes[0].set_ylabel("Price ($)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        errors = preds_flat - targets_flat
        axes[1].plot(x_axis, errors, color="red", linewidth=1, alpha=0.7)
        axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
        axes[1].set_title(f"{ticker_name} - Prediction Error Over Time")
        axes[1].set_xlabel("Time Step")
        axes[1].set_ylabel("Error ($)")
        axes[1].grid(True, alpha=0.3)

        axes[2].hist(errors, bins=30, edgecolor="black", alpha=0.7)
        axes[2].axvline(x=0, color="red", linestyle="--", alpha=0.7)
        axes[2].set_title(f"{ticker_name} - Residuals Distribution")
        axes[2].set_xlabel("Residual ($)")
        axes[2].set_ylabel("Frequency")
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        safe_name = ticker_name.replace("/", "_").replace("\\", "_")
        fig.savefig(viz_dir / f"pred_vs_actual_{safe_name}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: pred_vs_actual_{safe_name}.png")


def save_artifacts(
    config: dict,
    metrics: dict,
    training_history: List[dict],
    model: nn.Module,
    feature_scaler,
    target_scaler,
    output_dir: Path,
    best_epoch: int,
    total_epochs: int,
) -> None:
    """Save all training artefacts to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metrics JSON
    metrics_out = {**metrics, "best_epoch": best_epoch, "total_epochs": total_epochs}
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"  Saved: metrics.json")

    # 2. Hyperparameters JSON
    # Filter config to serialisable values
    hyper = {}
    for k, v in config.items():
        if isinstance(v, (str, int, float, bool, list, type(None))):
            hyper[k] = v
    with open(output_dir / "hyperparameters.json", "w") as f:
        json.dump(hyper, f, indent=2)
    print(f"  Saved: hyperparameters.json")

    # 3. Last model checkpoint
    torch.save(model.state_dict(), output_dir / "last_model.pt")
    print(f"  Saved: last_model.pt")

    # 4. Model summary
    if hasattr(model, "get_model_summary"):
        summary = model.get_model_summary()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        summary = f"Total parameters: {total_params:,}"
    with open(output_dir / "model_summary.txt", "w") as f:
        f.write(summary)
    print(f"  Saved: model_summary.txt")

    # 5. Training history CSV
    with open(output_dir / "training_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=training_history[0].keys())
        writer.writeheader()
        writer.writerows(training_history)
    print(f"  Saved: training_history.csv")

    # 6. Scalers
    with open(output_dir / "scalers.pkl", "wb") as f:
        pickle.dump(
            {"feature_scaler": feature_scaler, "target_scaler": target_scaler}, f
        )
    print(f"  Saved: scalers.pkl")


def get_top_tickers(config: dict, n: int = 10) -> List[str]:
    """
    Get top N tickers by quality score from stock_scores_news_1.csv.

    Falls back to an empty list if the file doesn't exist.
    """
    scores_path = config.get("STOCK_SCORE_NEWS", "")
    if not scores_path or not os.path.exists(scores_path):
        return []
    try:
        df = pd.read_csv(scores_path)
        df = df.sort_values("score", ascending=False)
        return df["ticker"].head(n).tolist()
    except Exception:
        return []


def train(train_config: dict = None) -> None:
    """
    Main training entry point.

    Orchestrates: config loading -> data preparation -> model init ->
    training loop with early stopping -> evaluation -> artefact saving ->
    visualisation.
    """
    # 1. Config
    args = parse_args()
    config = load_config(args)
    if train_config:
        config.update(train_config)

    seed = config.get("random_seed", 42)
    set_seed(seed)

    # 2. Device
    if "device" in config and config["device"]:
        device = torch.device(config["device"])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Prepare data
    from src.utils import read_json_file

    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    data_config = {
        "data_path": config["BASELINE_DATA_PATH"],
        "ticker2idx": ticker2idx,
        "test_train_split": 0.2,
        "random_seed": seed,
        "val_ratio": config.get("val_ratio", 0.15),
        "feature_groups": config.get(
            "feature_groups", ["ohlcv", "technical", "temporal"]
        ),
    }

    train_dataset, val_dataset, test_dataset, feature_scaler, target_scaler = (
        prepare_lstm_data(data_config)
    )

    actual_input_size = train_dataset.features.shape[2]
    config["input_size"] = actual_input_size

    batch_size = config.get("batch_size", 64)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Model
    model_config = {
        "input_size": config["input_size"],
        "hidden_size": config.get("hidden_size", 128),
        "num_layers": config.get("num_layers", 2),
        "dropout": config.get("dropout", 0.2),
        "output_size": config.get("output_size", 7),
        "bidirectional": config.get("bidirectional", False),
        "use_layer_norm": config.get("use_layer_norm", True),
        "use_attention": config.get("use_attention", False),
        "fc_hidden_sizes": config.get("fc_hidden_sizes", [64, 32]),
    }
    model = LSTMForecaster(model_config).to(device)
    print(f"\n{model.get_model_summary()}")

    # 5. Optimizer, scheduler, criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 0.001),
        weight_decay=config.get("weight_decay", 0.0001),
    )

    scheduler_type = config.get("lr_scheduler", "reduce_on_plateau")
    if scheduler_type == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=config.get("lr_patience", 5),
            factor=config.get("lr_factor", 0.5),
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.get("epochs", 100)
        )
    else:
        scheduler = None

    criterion = nn.MSELoss()

    # 6. Early stopping
    output_dir = Path(config.get("output_dir", "reports/experimental_results_lstm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_model.pt"

    early_stopping = EarlyStopping(
        patience=config.get("early_stopping_patience", 15)
    )

    # 7. Training loop
    n_epochs = config.get("epochs", 100)
    gradient_clip = config.get("gradient_clip_norm", 1.0)
    training_history = []

    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {config.get('learning_rate', 0.001)}")
    print(f"  Early stopping patience: {config.get('early_stopping_patience', 15)}")
    print()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model, train_dl, optimizer, criterion, device, gradient_clip
        )

        val_result = evaluate(model, val_dl, criterion, device)
        val_loss = val_result["loss"]
        val_metrics = val_result["metrics"]

        # Record history
        history_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_mae": round(val_metrics["mae"], 6),
            "val_rmse": round(val_metrics["rmse"], 6),
            "lr": optimizer.param_groups[0]["lr"],
        }
        training_history.append(history_entry)

        # Logging
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MAE: {val_metrics['mae']:.6f} | "
                f"Val RMSE: {val_metrics['rmse']:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping check
        if early_stopping(val_loss, epoch, model, best_model_path):
            print(
                f"\nEarly stopping triggered at epoch {epoch}. "
                f"Best epoch: {early_stopping.best_epoch}"
            )
            break

    # 8. Load best model and evaluate on test set
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # Build ticker_id -> name mapping
    ticker_id_to_name = {v: k for k, v in ticker2idx.items()}

    test_result = evaluate(
        model, test_dl, criterion, device, target_scaler=target_scaler
    )

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print_metrics(test_result["metrics"], prefix="Test")

    # 9. Save artefacts
    print("\nSaving artefacts...")
    save_artifacts(
        config=config,
        metrics=test_result["metrics"],
        training_history=training_history,
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        output_dir=output_dir,
        best_epoch=early_stopping.best_epoch,
        total_epochs=len(training_history),
    )

    # 10. Visualisations
    print("\nGenerating visualisations...")
    top_tickers = get_top_tickers(config, n=10)
    create_visualizations(
        training_history=training_history,
        per_stock_preds=test_result["per_stock"],
        output_dir=output_dir,
        ticker_id_to_name=ticker_id_to_name,
        top_tickers=top_tickers,
        max_stocks=10,
    )

    print(f"\nAll artefacts saved to: {output_dir}")
    print("Training complete.")


if __name__ == "__main__":
    train()
