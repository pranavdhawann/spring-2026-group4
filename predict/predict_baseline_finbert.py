"""

Script to train FinBert model

"""
import copy
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models import FinBertForecastingBL
from src.preProcessing import FinBertCollator
from src.utils import calculate_regression_metrics, read_json_file, read_yaml, set_seed

scaler = GradScaler()  # for stable mixed precision


def convert_X_to_tensors(X, device="cpu"):
    input_ids = torch.stack([sample[0] for sample in X])  # (B, W, L)

    attention_mask = (input_ids != 0).long()
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


def predict(predict_config):
    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/finbertForecasting",
        "batch_size": 7,
        "max_length": 512,
        "tokenizer_path": "ProsusAI/finbert",
        "local_files_only": True,
        "sample_fraction": 0.002,
        "bert_hidden_dim": 768,
        "lstm_hidden_dim": 32,
        "lstm_num_layers": 1,
        "y_true_vs_y_pred_max_points": 5000,
        "rand_seed": 42,
        "verbose": False,
    }

    config.update(predict_config)
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
    with open(os.path.join(config["experiment_path"], "dataloaders.pkl"), "rb") as f:
        dataloaders = pickle.load(f)

        test_dataloader = dataloaders["test"]

        print("Dataloaders loaded successfully!")
    del dataloaders

    test_dataloader = get_fraction_subset(
        test_dataloader, fraction=config["sample_fraction"]
    )
    collator = FinBertCollator(config)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = copy.deepcopy(config)
    model_config["device"] = device
    model = FinBertForecastingBL(model_config)

    model = torch.compile(model)
    model.load_state_dict(
        torch.load(
            os.path.join(config["experiment_path"], "best_model.pth"),
            map_location=device,
        )
    )
    print(" Loaded Pre Trained Model")

    criterion = nn.MSELoss()

    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        test_loop = tqdm(test_loader, desc="Predicting", leave=False)
        for batch_idx, (X_test, y_test) in enumerate(test_loop):
            y_test = torch.as_tensor(np.array(y_test), dtype=torch.float, device=device)
            X_test = convert_X_to_tensors(X_test, device=device)

            with autocast(device_type="cuda", dtype=torch.float16):
                y_pred_test = model(X_test)
                loss_test = criterion(y_pred_test.squeeze(), y_test)

            test_loss += loss_test.item()
            test_loop.set_postfix(loss=f"{loss_test.item():.4f}")
            y_true.extend(y_test.detach().view(-1).cpu().numpy())
            y_pred.extend(y_pred_test.detach().view(-1).cpu().numpy())

            if batch_idx < 10:
                y_test_plot = y_test.detach().cpu().numpy()
                y_pred_plot = y_pred_test.detach().cpu().numpy()
                if len(y_test_plot.shape) == 1:
                    batch_size = X_test["closes"].shape[0]
                    y_test_plot = y_test_plot.reshape(batch_size, -1)
                    y_pred_plot = y_pred_plot.reshape(batch_size, -1)

                plot_batch_predictions(
                    X_closes=X_test["closes"],
                    y_true=y_test_plot,
                    y_pred=y_pred_plot,
                    batch_idx=batch_idx,
                    save_path=os.path.join(
                        config["experiment_path"],
                        f"predict/batch_{batch_idx}_predictions.png",
                    ),
                    num_stocks_to_plot=min(3, X_test["closes"].shape[0]),
                )

        config["test_samples"] = len(test_loader) * config["batch_size"]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    test_metrics = calculate_regression_metrics(y_true, y_pred)
    print("     Test Metrics: ", test_metrics)
    avg_test_loss = test_loss / len(test_loader)
    print("     Average Test Loss: ", avg_test_loss)
    save_prediction_scatter(
        y_true,
        y_pred,
        os.path.join(config["experiment_path"], "predict/predictions.png"),
        config["y_true_vs_y_pred_max_points"],
    )


if __name__ == "__main__":
    predict_config = {"load_pre_trained": True}
    predict(predict_config)
