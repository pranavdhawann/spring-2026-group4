"""

Script to train ChronosFinBert model

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

from src.dataLoader.chronosDataLoader import getTrainTestDataLoader
from src.models.chronosFinBert import ChronosFinBert
from src.preProcessing.chronosCollator import ChronosCollator
from src.utils import calculate_regression_metrics, read_json_file, read_yaml, set_seed

scaler = GradScaler()  # for stable mixed precision


def convert_X_to_tensors(X, device="cpu"):
    # Pad all input_ids to the same number of windows
    max_windows = max(sample[0].shape[0] for sample in X)
    L = X[0][0].shape[1]

    padded_ids = []
    padded_masks = []
    for sample in X:
        ids = sample[0]  # (W, L)
        mask = sample[
            1
        ]  # (W, L) — real mask from collator, preserves special tokens correctly
        W = ids.shape[0]
        if W < max_windows:
            pad = torch.zeros(max_windows - W, L, dtype=ids.dtype)
            ids = torch.cat([ids, pad], dim=0)
            mask = torch.cat(
                [mask, torch.zeros(max_windows - W, L, dtype=mask.dtype)], dim=0
            )
        padded_ids.append(ids)
        padded_masks.append(mask)

    input_ids = torch.stack(padded_ids)  # (B, max_windows, L)
    attention_mask = torch.stack(padded_masks)  # (B, max_windows, L)

    closes = torch.stack(
        [torch.as_tensor(sample[2], dtype=torch.float) for sample in X]
    )
    extra_features = torch.tensor(
        [[sample[3], sample[4]] for sample in X], dtype=torch.float
    )

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "closes": closes.to(device),
        "extra_features": extra_features.to(device),
    }


def get_fraction_subset(dataset, fraction=0.2, random_sample=True, seed=42):
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
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Subsample if too many points
    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    plt.figure(figsize=(7, 7))

    # Scatter
    plt.scatter(y_true, y_pred, alpha=0.5)

    # Ideal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs True Scatter Plot")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"     Prediction scatter plot saved to {save_path}")


def save_sample_forecast_plot(y_true, y_pred, epoch, save_dir, rmse=0.0):
    y_true = np.asarray(y_true).reshape(-1, 5)
    y_pred = np.asarray(y_pred).reshape(-1, 5)

    n_samples = len(y_true)
    indices = np.random.choice(n_samples, size=min(4, n_samples), replace=False)
    x_ticks = np.arange(1, 6)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(
        f"Best Model Predictions \u2014 Epoch {epoch} | RMSE: {rmse:.4f}",
        fontsize=14,
    )
    for plot_idx, sample_idx in enumerate(indices):
        ax = axes[plot_idx // 2][plot_idx % 2]
        ax.plot(x_ticks, y_true[sample_idx], color="blue", label="True")
        ax.plot(
            x_ticks,
            y_pred[sample_idx],
            color="orange",
            linestyle="--",
            label="Predicted",
        )
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.7, label="Forecast Start")
        ax.set_xlabel("Forecast Day")
        ax.set_ylabel("Scaled Price")
        ax.set_xticks(x_ticks)
        ax.set_title(f"Sample {plot_idx + 1} \u2014 Epoch {epoch}")
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(save_dir, f"best_predictions_epoch_{epoch}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"     Sample forecast plot saved to {save_path}")


def plot_batch_predictions(
    X_closes, y_true, y_pred, batch_idx, save_path, num_stocks_to_plot=None
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if hasattr(X_closes, "cpu"):
        X_closes = X_closes.cpu().numpy()
    if hasattr(y_true, "cpu"):
        y_true = y_true.cpu().numpy()
    if hasattr(y_pred, "cpu"):
        y_pred = y_pred.cpu().numpy()

    if len(y_true.shape) == 1:
        batch_size = X_closes.shape[0]
        forecast_days = len(y_true) // batch_size
        y_true = y_true.reshape(batch_size, forecast_days)
        y_pred = y_pred.reshape(batch_size, forecast_days)

    batch_size = X_closes.shape[0]
    forecast_days = y_true.shape[1]
    historical_days = X_closes.shape[1]

    if num_stocks_to_plot is None:
        num_stocks_to_plot = batch_size
    else:
        num_stocks_to_plot = min(num_stocks_to_plot, batch_size)

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    historical_idx = np.arange(historical_days)
    forecast_idx = np.arange(historical_days, historical_days + forecast_days)

    colors = plt.cm.tab10(np.linspace(0, 1, num_stocks_to_plot))

    for i in range(num_stocks_to_plot):
        color = colors[i]

        ax.plot(
            historical_idx,
            X_closes[i],
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
            label=f"Stock {i + 1} Historical",
        )

        ax.plot(
            forecast_idx,
            y_true[i],
            color=color,
            linestyle=":",
            marker="o",
            markersize=4,
            linewidth=2,
            label=f"Stock {i + 1} True",
        )

        ax.plot(
            forecast_idx,
            y_pred[i],
            color=color,
            linestyle="--",
            marker="s",
            markersize=4,
            linewidth=2,
            label=f"Stock {i + 1} Pred",
        )

    ax.axvline(
        x=historical_days - 0.5, color="gray", linestyle="-", alpha=0.5, linewidth=2
    )
    ax.text(
        historical_days - 0.5,
        ax.get_ylim()[1] * 0.95,
        "Forecast Start",
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right",
        color="gray",
        fontsize=10,
    )

    ax.set_xlabel("Days", fontsize=12)
    ax.set_ylabel("Stock Close Price", fontsize=12)
    ax.set_title(f"Batch {batch_idx}: Stock Price Predictions", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {save_path}")


def train(train_config):
    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/chronosfinbert0407",
        "load_pre_trained": False,
        "epochs": 12,
        "batch_size": 64,
        "max_length": 512,
        "lr": {
            "bert": 1e-5,
            "mlp": 1e-4,
        },
        "tokenizer_path": "ProsusAI/finbert",
        "local_files_only": False,
        "sample_fraction": 1,
        "patience": 10,
        "number_of_epochs_ran": 0,
        "y_true_vs_y_pred_max_points": 5000,
        "rand_seed": 42,
        "verbose": False,
        "scheduled_sampling": "linear",
        "freeze_bert": True,
        "freeze_chronos": True,
        "max_window_size": 90,
        "FORECAST_HORIZON": 5,
        "HISTORY_WINDOW_SIZE": 90,
    }

    config.update(train_config)
    set_seed(config["rand_seed"])

    yaml_config = read_yaml(config["yaml_config_path"])
    config.update(yaml_config)
    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    data_config = {
        "data_path": config["BASELINE_DATA_PATH"],  # path to dataset folder
        "ticker2idx": ticker2idx,  # Ticker to Id dictionary
        "test_train_split": 0.2,
    }

    config.update(data_config)

    # load or process dataloader from pkl file
    if not os.path.exists(os.path.join(config["experiment_path"], "dataloaders.pkl")):
        train_dataloader, test_dataloader = getTrainTestDataLoader(config)
        with open(
            os.path.join(config["experiment_path"], "dataloaders.pkl"), "wb"
        ) as f:
            pickle.dump({"train": train_dataloader, "test": test_dataloader}, f)
        print("Dataloaders saved!")
    else:
        with open(
            os.path.join(config["experiment_path"], "dataloaders.pkl"), "rb"
        ) as f:
            dataloaders = pickle.load(f)

        train_dataloader = dataloaders["train"]
        test_dataloader = dataloaders["test"]

        print("Dataloaders loaded successfully!")

    train_dataloader = get_fraction_subset(
        train_dataloader, fraction=config["sample_fraction"]
    )
    test_dataloader = get_fraction_subset(
        test_dataloader, fraction=config["sample_fraction"]
    )
    collator = ChronosCollator(config)

    train_loader = DataLoader(
        train_dataloader,
        collate_fn=collator,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataloader,
        collate_fn=collator,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = copy.deepcopy(config)
    model_config["device"] = device
    model = ChronosFinBert(model_config)

    if config["load_pre_trained"]:
        model_path = os.path.join(config["experiment_path"], "best_model.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(" Loaded Pre Trained Model")

    model.to(device)

    criterion = nn.HuberLoss(delta=0.01)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.finbert.parameters(),
                "lr": config["lr"]["bert"],
                "weight_decay": config.get(
                    "weight_decay_bert", 0.01
                ),  # Optional: different weight decay for BERT
            },
            {
                "params": model.news_projection.parameters(),
                "lr": config["lr"]["mlp"],
            },
            {
                "params": list(model.ts_input_proj.parameters())
                + list(model.ts_projection.parameters())
                + list(model.cross_attention.parameters())
                + list(model.lstm.parameters())
                + list(model.head.parameters()),
                "lr": config["lr"]["mlp"],
            },
        ],
        weight_decay=config.get(
            "weight_decay", 0.01
        ),  # Default weight decay for all groups
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    best_rmse = np.inf
    epochs_without_improvement = 0

    train_losses = []
    test_losses = []
    for epoch in range(config["epochs"]):
        total_loss = 0
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        st_ = time.time()
        y_true = []
        y_pred = []
        for X, y in loop:
            if config["verbose"]:
                print("     time taken to retrive one batch: ", time.time() - st_)

            # y = torch.tensor(y, dtype=torch.float, device=device)

            st_ = time.time()
            y = torch.as_tensor(np.array(y), dtype=torch.float, device=device)
            X = convert_X_to_tensors(X, device=device)
            if config["verbose"]:
                print("     time taken to convert to tensor: ", time.time() - st_)

            optimizer.zero_grad()

            st_ = time.time()
            with autocast(device_type="cuda", dtype=torch.float16):
                y_ = model(X)
                loss = criterion(y_.squeeze(), y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # skip batch if gradients are NaN to prevent training collapse
            if any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            ):
                scaler.update()
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            if config["verbose"]:
                print("     time taken to forward + backward: ", time.time() - st_)

            # skip batch if loss is NaN
            if torch.isnan(loss):
                st_ = time.time()
                continue

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            st_ = time.time()

            y_true.extend(y.detach().view(-1).cpu().numpy())
            y_pred.extend(y_.detach().view(-1).cpu().numpy())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        train_metrics = calculate_regression_metrics(y_true, y_pred)
        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")
        print(
            f"     Train | RMSE: {train_metrics['rmse']:.4f}  "
            f"MAE: {train_metrics['mae']:.4f}  "
            f"MAPE: {train_metrics['mape']:.2f}%  "
            f"SMAPE: {train_metrics['smape']:.2f}%"
        )
        train_losses.append(total_loss / len(train_loader))
        config["train_samples"] = len(train_loader) * config["batch_size"]

        model.eval()
        test_loss = 0.0
        y_true = []
        y_pred = []
        X_test_list = []
        with torch.no_grad():
            test_loop = tqdm(test_loader, desc=f"Epoch {epoch + 1} | Test", leave=False)
            for X_test_raw, y_test in test_loop:
                X_test_list.extend(X_test_raw)
                # y_test = torch.tensor(y_test, dtype=torch.float, device=device)
                y_test = torch.as_tensor(
                    np.array(y_test), dtype=torch.float, device=device
                )
                X_test = convert_X_to_tensors(X_test_raw, device=device)

                with autocast(device_type="cuda", dtype=torch.float16):
                    # y_pred = model(X)
                    # loss = criterion(y_pred.squeeze(), y)
                    y_pred_test = model(X_test)
                    loss_test = criterion(y_pred_test.squeeze(), y_test)

                test_loss += loss_test.item()
                test_loop.set_postfix(loss=f"{loss_test.item():.4f}")
                y_true.extend(y_test.detach().view(-1).cpu().numpy())
                y_pred.extend(y_pred_test.detach().view(-1).cpu().numpy())
            config["test_samples"] = len(test_loader) * config["batch_size"]
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        test_metrics = calculate_regression_metrics(y_true, y_pred)
        print(
            f"     Test  | RMSE: {test_metrics['rmse']:.4f}  "
            f"MAE: {test_metrics['mae']:.4f}  "
            f"MAPE: {test_metrics['mape']:.2f}%  "
            f"SMAPE: {test_metrics['smape']:.2f}%"
        )

        avg_test_loss = test_loss / len(test_loader)

        test_losses.append(avg_test_loss)
        scheduler.step(avg_test_loss)

        if test_metrics["rmse"] < best_rmse:
            best_rmse = test_metrics["rmse"]
            torch.save(
                model.state_dict(),
                os.path.join(config["experiment_path"], "best_model.pth"),
            )
            save_sample_forecast_plot(
                y_true.reshape(-1, 5),
                y_pred.reshape(-1, 5),
                epoch + 1,
                config["experiment_path"],
                rmse=best_rmse,
            )
            best_epoch_dir = os.path.join(
                config["experiment_path"], f"best_epoch_{epoch + 1}"
            )
            os.makedirs(best_epoch_dir, exist_ok=True)
            model.eval()
            with torch.no_grad():
                for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
                    if batch_idx >= 10:
                        break
                    y_batch = torch.as_tensor(
                        np.array(y_batch), dtype=torch.float, device=device
                    )
                    X_batch = convert_X_to_tensors(X_batch, device=device)
                    with autocast(device_type="cuda", dtype=torch.float16):
                        y_pred_batch = model(X_batch)
                    y_batch_plot = y_batch.detach().cpu().numpy()
                    y_pred_plot = y_pred_batch.detach().cpu().numpy()
                    if len(y_batch_plot.shape) == 1:
                        bs = X_batch["closes"].shape[0]
                        y_batch_plot = y_batch_plot.reshape(bs, -1)
                        y_pred_plot = y_pred_plot.reshape(bs, -1)
                    plot_batch_predictions(
                        X_closes=X_batch["closes"],
                        y_true=y_batch_plot,
                        y_pred=y_pred_plot,
                        batch_idx=batch_idx,
                        save_path=os.path.join(
                            best_epoch_dir,
                            f"batch_{batch_idx}_predictions.png",
                        ),
                        num_stocks_to_plot=min(3, X_batch["closes"].shape[0]),
                    )
            with open(
                os.path.join(config["experiment_path"], "best_metrics.json"), "w"
            ) as f:
                json.dump(
                    {
                        "best_epoch": epoch + 1,
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                    },
                    f,
                    indent=4,
                )
            epochs_without_improvement = 0
            print("     Best model saved!")
        else:
            epochs_without_improvement += 1

        # save results needed
        save_losses_plot(
            train_losses,
            test_losses,
            os.path.join(config["experiment_path"], "losses.png"),
        )

        save_prediction_scatter(
            y_true,
            y_pred,
            os.path.join(config["experiment_path"], "predictions.png"),
            config["y_true_vs_y_pred_max_points"],
        )
        with open(os.path.join(config["experiment_path"], "metrics.json"), "w") as f:
            metrics = {
                "train_": train_metrics,
                "test_": test_metrics,
            }
            json.dump(metrics, f, indent=4)
        config["number_of_epochs_ran"] = epoch
        with open(os.path.join(config["experiment_path"], "config.json"), "w") as f:
            json.dump(config, f, indent=4)

        if epochs_without_improvement >= config["patience"]:
            print("     Early stopping!")
            config["early_stopping_initiated"] = True
            break
        print(f"     Avg Test Loss: {avg_test_loss:.4f}\n")


if __name__ == "__main__":
    train_config = {
        "load_pre_trained": False,
        "sample_fraction": 1.0,
        "epochs": 12,
        "batch_size": 4,
    }
    train(train_config)
