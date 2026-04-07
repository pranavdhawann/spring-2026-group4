"""
Script to train FinBert LSTM Decoder model with teacher forcing

"""

import copy
import json
import math
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataLoader import getTrainTestDataLoader
from src.models import FinBertLSTMDecoder
from src.preProcessing import FinBertCollator
from src.utils import calculate_regression_metrics, read_json_file, read_yaml, set_seed

scaler = GradScaler()  # for stable mixed precision


def convert_X_to_tensors(X, device="cpu"):
    """
    Convert collator output to model inputs for LSTM decoder
    X format from collator: [input_ids, attention_mask, closes, mean, std]
    """
    # Stack input_ids (already tokenized)
    input_ids = torch.stack([sample[0] for sample in X])  # (B, W, L)

    # Create attention_mask from input_ids (0 for padding)
    attention_mask = (input_ids != 0).long()

    # Extract closing prices
    closes = torch.stack(
        [torch.as_tensor(sample[2], dtype=torch.float) for sample in X]
    )  # (B, N)

    # Extra features: mean and std from standardization
    extra_features = torch.tensor(
        [[sample[3], sample[4]] for sample in X], dtype=torch.float
    )  # (B, 2)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "closes": closes.to(device),
        "extra_features": extra_features.to(device),
    }


def get_fraction_subset(dataset, fraction=0.2, random_sample=True, seed=42):
    """Get a fraction of the dataset"""
    dataset_size = len(dataset)
    subset_size = max(1, math.ceil(dataset_size * fraction))

    if random_sample:
        rng = random.Random(seed)
        indices = rng.sample(range(dataset_size), subset_size)
    else:
        indices = list(range(subset_size))

    subset_dataset = Subset(dataset, indices)
    return subset_dataset


def save_losses_plot(train_losses, test_losses, save_path):
    """Save training and test loss plots"""
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", marker="o", color="blue")
    plt.plot(test_losses, label="Test Loss", marker="o", color="orange")

    plt.title("Training and Test Loss Over Epochs", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"     Loss plot saved to {save_path}")


def save_prediction_scatter(
    y_true, y_pred, save_path="pred_vs_true.png", max_points=5000
):
    """Save scatter plot of predictions vs ground truth"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Subsample if too many points
    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    plt.figure(figsize=(7, 7))

    # Scatter
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)

    # Ideal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot(
        [min_val, max_val], [min_val, max_val], linestyle="--", color="red", linewidth=2
    )

    plt.xlabel("True Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title("Predicted vs True Scatter Plot", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"     Prediction scatter plot saved to {save_path}")


def update_teacher_forcing_ratio(model, epoch, total_epochs, schedule_type="linear"):
    """
    Gradually reduce teacher forcing ratio during training

    Args:
        model: The model with teacher_forcing_ratio attribute
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        schedule_type: Type of scheduling ('linear', 'exponential', 'inverse_sigmoid', 'constant')

    Returns:
        Updated teacher forcing ratio
    """
    if not hasattr(model, "teacher_forcing_ratio"):
        return 0.5

    if schedule_type == "linear":
        # Linear decay from 0.9 to 0.1
        ratio = 0.9 - (0.8 * epoch / max(1, total_epochs - 1))
    elif schedule_type == "exponential":
        # Exponential decay
        ratio = 0.9 * (0.5 ** (epoch / max(1, total_epochs / 3)))
    elif schedule_type == "inverse_sigmoid":
        # Inverse sigmoid decay (stays high longer then drops)
        center = total_epochs / 2
        scale = total_epochs / 10
        ratio = 1.0 / (1.0 + np.exp((epoch - center) / scale))
    else:
        ratio = 0.5  # constant

    # Clamp between 0.1 and 0.9
    ratio = max(0.1, min(0.9, ratio))
    model.teacher_forcing_ratio = ratio
    return ratio


def train(train_config):
    """Main training function"""
    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/finbertForecastingLSTMDecoder",
        "load_pre_trained": False,
        "epochs": 100,
        "batch_size": 16,
        "max_length": 512,
        "lr": {
            "bert": 1e-5,
            "mlp": 1e-3,
        },
        "tokenizer_path": "ProsusAI/finbert",
        "local_files_only": True,
        "sample_fraction": 1,
        "patience": 10,
        "number_of_epochs_ran": 0,
        "y_true_vs_y_pred_max_points": 5000,
        "rand_seed": 42,
        "verbose": False,
        "scheduled_sampling": "linear",  # 'linear', 'exponential', 'inverse_sigmoid', 'constant'
        # LSTM-specific parameters
        "lstm_hidden_dim": 512,
        "lstm_num_layers": 4,
        "teacher_forcing_ratio": 0.5,  # Initial teacher forcing ratio
        "freeze_bert": False,
        "max_window_size": 14,
        "empty_news_threshold": 2,
        # Model architecture
        "mlp_hidden_dims": [128, 64],
        "news_embedding_dim": 256,
        # Optimization
        "weight_decay": 0.01,
        "weight_decay_bert": 0.01,
        "gradient_clip_val": 1.0,  # Gradient clipping for LSTM stability
    }

    config.update(train_config)
    set_seed(config["rand_seed"])

    # Load configuration
    yaml_config = read_yaml(config["yaml_config_path"])
    config.update(yaml_config)
    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    data_config = {
        "data_path": config["BASELINE_DATA_PATH"],
        "ticker2idx": ticker2idx,
        "test_train_split": 0.2,
    }
    config.update(data_config)

    # Create experiment directory if it doesn't exist
    os.makedirs(config["experiment_path"], exist_ok=True)

    # Load or create dataloaders
    dataloader_path = os.path.join(config["experiment_path"], "dataloaders.pkl")
    if not os.path.exists(dataloader_path):
        print("Creating new dataloaders...")
        train_dataloader, test_dataloader = getTrainTestDataLoader(config)
        with open(dataloader_path, "wb") as f:
            pickle.dump({"train": train_dataloader, "test": test_dataloader}, f)
        print("Dataloaders saved!")
    else:
        print("Loading existing dataloaders...")
        with open(dataloader_path, "rb") as f:
            dataloaders = pickle.load(f)
        train_dataloader = dataloaders["train"]
        test_dataloader = dataloaders["test"]
        print("Dataloaders loaded successfully!")

    # Apply subset sampling if needed
    if config["sample_fraction"] < 1.0:
        train_dataloader = get_fraction_subset(
            train_dataloader, fraction=config["sample_fraction"]
        )
        test_dataloader = get_fraction_subset(
            test_dataloader, fraction=config["sample_fraction"]
        )
        print(f"Using {config['sample_fraction']*100:.1f}% of data")

    # Create collator and data loaders
    collator = FinBertCollator(config)

    train_loader = DataLoader(
        train_dataloader,
        collate_fn=collator,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataloader,
        collate_fn=collator,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_config = copy.deepcopy(config)
    model_config["device"] = device
    model = FinBertLSTMDecoder(model_config)

    # Set initial teacher forcing ratio
    model.teacher_forcing_ratio = config.get("teacher_forcing_ratio", 0.5)

    # Load pretrained model if specified
    if config["load_pre_trained"]:
        model_path = os.path.join(config["experiment_path"], "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Loaded pretrained model!")
        else:
            print(f"Warning: Pretrained model not found at {model_path}")
            model.to(device)
        model = torch.compile(model)
    else:
        model.to(device)
        model = torch.compile(model)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer with separate parameter groups
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.finbert.parameters(),
                "lr": config["lr"]["bert"],
                "weight_decay": config.get("weight_decay_bert", 0.01),
            },
            {
                "params": model.token_projection.parameters(),
                "lr": config["lr"]["mlp"],
            },
            {
                "params": model.feature_embedding.parameters(),
                "lr": config["lr"]["mlp"],
            },
            {
                "params": model.lstm_decoder.parameters(),
                "lr": config["lr"]["mlp"],
            },
            {
                "params": model.output_projection.parameters(),
                "lr": config["lr"]["mlp"],
            },
        ],
        weight_decay=config.get("weight_decay", 0.01),
    )

    # Training state
    best_rmse = np.inf
    epochs_without_improvement = 0
    train_losses = []
    test_losses = []

    # Main training loop
    for epoch in range(config["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*60}")

        # Update teacher forcing ratio based on schedule
        tf_ratio = update_teacher_forcing_ratio(
            model,
            epoch,
            config["epochs"],
            schedule_type=config.get("scheduled_sampling", "linear"),
        )
        print(f"Teacher forcing ratio: {tf_ratio:.3f}")

        # Training phase
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc="Training", leave=False)
        st_ = time.time()

        train_true = []
        train_pred = []

        for batch_idx, (X, y) in enumerate(loop):
            if config["verbose"] and batch_idx == 0:
                print(f"Batch retrieval time: {time.time() - st_:.3f}s")

            # Move data to device
            st_ = time.time()
            y_tensor = torch.as_tensor(np.array(y), dtype=torch.float, device=device)
            X_tensor = convert_X_to_tensors(X, device=device)

            # Reshape targets to (B, H, 1) for teacher forcing
            if y_tensor.dim() == 2:
                targets = y_tensor.unsqueeze(-1)  # (B, H) -> (B, H, 1)
            else:
                targets = y_tensor

            if config["verbose"] and batch_idx == 0:
                print(f"Data conversion time: {time.time() - st_:.3f}s")

            optimizer.zero_grad()

            st_ = time.time()
            # Forward pass with teacher forcing
            with autocast(device_type="cuda", dtype=torch.float16):
                predictions = model(X_tensor, targets=targets, training=True)
                loss = criterion(predictions.squeeze(), targets.squeeze())

            # Backward pass with gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.get("gradient_clip_val", 1.0)
            )
            scaler.step(optimizer)
            scaler.update()

            if config["verbose"] and batch_idx == 0:
                print(f"Forward/backward time: {time.time() - st_:.3f}s")

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

            # Store predictions and targets for metrics
            train_true.extend(targets.detach().view(-1).cpu().numpy())
            train_pred.extend(predictions.detach().view(-1).cpu().numpy())

            st_ = time.time()

        # Calculate training metrics
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_metrics = calculate_regression_metrics(
            np.array(train_true), np.array(train_pred)
        )

        print("\nTraining Results:")
        print(f"  Loss: {avg_train_loss:.4f}")
        print(f"  Metrics: {train_metrics}")

        # Validation phase
        model.eval()
        test_loss = 0
        test_true = []
        test_pred = []

        with torch.no_grad():
            test_loop = tqdm(test_loader, desc="Validation", leave=False)
            for X_test, y_test in test_loop:
                # Move data to device
                y_test_tensor = torch.as_tensor(
                    np.array(y_test), dtype=torch.float, device=device
                )
                X_test_tensor = convert_X_to_tensors(X_test, device=device)

                # Reshape targets
                if y_test_tensor.dim() == 2:
                    targets_test = y_test_tensor.unsqueeze(-1)
                else:
                    targets_test = y_test_tensor

                # Forward pass (no teacher forcing)
                with autocast(device_type="cuda", dtype=torch.float16):
                    predictions_test = model(X_test_tensor, training=False)
                    loss_test = criterion(
                        predictions_test.squeeze(), targets_test.squeeze()
                    )

                test_loss += loss_test.item()
                test_loop.set_postfix(loss=f"{loss_test.item():.4f}")

                # Store predictions
                test_true.extend(targets_test.detach().view(-1).cpu().numpy())
                test_pred.extend(predictions_test.detach().view(-1).cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_metrics = calculate_regression_metrics(
            np.array(test_true), np.array(test_pred)
        )

        print("\nValidation Results:")
        print(f"  Loss: {avg_test_loss:.4f}")
        print(f"  Metrics: {test_metrics}")

        # Save best model based on RMSE
        if test_metrics["rmse"] < best_rmse:
            best_rmse = test_metrics["rmse"]
            # Save model state dict (unwrap from torch.compile if needed)
            model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save(
                model_to_save.state_dict(),
                os.path.join(config["experiment_path"], "best_model.pth"),
            )
            epochs_without_improvement = 0
            print("  ✓ Best model saved!")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epochs")

        # Save visualizations
        save_losses_plot(
            train_losses,
            test_losses,
            os.path.join(config["experiment_path"], "losses.png"),
        )

        save_prediction_scatter(
            test_true,
            test_pred,
            os.path.join(config["experiment_path"], "predictions.png"),
            config["y_true_vs_y_pred_max_points"],
        )

        # Save metrics
        with open(os.path.join(config["experiment_path"], "metrics.json"), "w") as f:
            metrics = {
                "train": train_metrics,
                "test": test_metrics,
                "best_rmse": best_rmse,
            }
            json.dump(metrics, f, indent=4)

        # Save configuration with current epoch
        config["number_of_epochs_ran"] = epoch + 1
        config["best_rmse"] = best_rmse
        with open(os.path.join(config["experiment_path"], "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        # Early stopping
        if epochs_without_improvement >= config["patience"]:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            config["early_stopping_initiated"] = True
            break

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Total epochs trained: {config['number_of_epochs_ran']}")
    print(f"Results saved to: {config['experiment_path']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Example configuration for LSTM decoder
    train_config = {
        "load_pre_trained": False,
        "epochs": 100,
        "batch_size": 32,  # May need smaller batch size for LSTM
        "freeze_bert": True,  # Recommended to freeze BERT initially
        "lstm_hidden_dim": 256,  # Can adjust based on your data
        "lstm_num_layers": 2,
        "teacher_forcing_ratio": 0.7,  # Start with high teacher forcing
        "scheduled_sampling": "linear",  # Gradually reduce teacher forcing
        "learning_rate": {
            "bert": 1e-5,
            "lstm": 1e-3,
        },
        "gradient_clip_val": 1.0,
        "patience": 10,  # More patience for LSTM training
    }

    train(train_config)
