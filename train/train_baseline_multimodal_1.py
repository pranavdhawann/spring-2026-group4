"""

Script to train FinBert model

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
from src.models import MultiModalStockPredictor
from src.preProcessing import MultiModalPreProcessing
from src.utils import (
    calculate_regression_metrics,
    get_sector2Idx,
    read_json_file,
    read_yaml,
    set_seed,
)

scaler = GradScaler()  # for stable mixed precision


def prepare_batch(X, y, device="cpu", max_articles=32, pad_token_id=0, sector2idx=None):
    batch_size = len(X)
    token_len = 512
    tokenized_news = torch.full(
        (batch_size, max_articles, token_len), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros(batch_size, max_articles, token_len, dtype=torch.long)

    article_mask = torch.zeros(batch_size, max_articles, dtype=torch.bool)

    for i, x in enumerate(X):
        articles = x["tokenized_news_"]
        masks = x["attention_mask_news_"]

        num_articles = min(len(articles), max_articles)

        for j in range(num_articles):
            article_tensor = torch.tensor(articles[j], dtype=torch.long)
            mask_tensor = torch.tensor(masks[j], dtype=torch.long)
            actual_len = min(len(article_tensor), token_len)

            tokenized_news[i, j, :actual_len] = article_tensor[:actual_len]
            attention_mask[i, j, :actual_len] = mask_tensor[:actual_len]
            article_mask[i, j] = True

    time_series_list = []
    for x in X:
        ts = x["time_series_features_"]
        if isinstance(ts, np.ndarray):
            time_series_list.append(torch.from_numpy(ts).float())
        else:
            time_series_list.append(torch.tensor(ts, dtype=torch.float32))
    time_series = torch.stack(time_series_list)

    ticker_ids = torch.tensor([x["ticker_id_"] for x in X], dtype=torch.long)

    sectors = torch.tensor(
        [sector2idx[x["sector_"]] if x["sector_"] in sector2idx else 0 for x in X],
        dtype=torch.long,
    )

    if isinstance(y, np.ndarray):
        targets = torch.from_numpy(y).float()
    else:
        # Make sure y is on CPU first
        if isinstance(y, torch.Tensor):
            y = y.cpu()
        targets = torch.tensor(np.array(y), dtype=torch.float32)

    if device != "cpu":
        tokenized_news = tokenized_news.to(device)
        attention_mask = attention_mask.to(device)
        article_mask = article_mask.to(device)
        time_series = time_series.to(device)
        ticker_ids = ticker_ids.to(device)
        sectors = sectors.to(device)
        targets = targets.to(device)

    return {
        "tokenized_news_": tokenized_news,
        "attention_mask_news_": attention_mask,
        "time_series_features_": time_series,
        "ticker_id_": ticker_ids,
        "sector_": sectors,
    }, targets


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


def train(train_config):
    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/multimodal",
        "load_pre_trained": False,
        "epochs": 1,
        "batch_size": 2,
        "max_length": 512,
        "lr": {
            "bert": 1e-5,
            "mlp": 1e-3,
        },
        "bert_path": "ProsusAI/finbert",
        "local_files_only": True,
        "sample_fraction": 1,
        "patience": 5,
        "number_of_epochs_ran": 0,
        "y_true_vs_y_pred_max_points": 5000,
        "rand_seed": 42,
        "verbose": False,
        "scheduled_sampling": "linear",
        "news_embedding_dim": 256,
        "max_window_size": 14,
        "num_articles": 18,
        "time_series_features": 12,
    }
    if not os.path.exists(config["experiment_path"]):
        os.mkdir(os.path.join(config["experiment_path"]))
    config.update(train_config)
    set_seed(config["rand_seed"])

    yaml_config = read_yaml(config["yaml_config_path"])
    config.update(yaml_config)
    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    sector2idx = get_sector2Idx(config["DATA_DICTIONARY"])
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
    collator = MultiModalPreProcessing(config)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = copy.deepcopy(config)
    model_config["device"] = device
    model = MultiModalStockPredictor(model_config, len(ticker2idx), len(sector2idx))

    if config["load_pre_trained"]:
        model_path = os.path.join(config["experiment_path"], "best_model.pth")
        model = torch.compile(model)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(" Loaded Pre Trained Model")
    else:
        model.to(device)
        model = torch.compile(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

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
            # y = torch.as_tensor(np.array(y), dtype=torch.float, device=device)
            model_inputs, y = prepare_batch(
                X,
                y,
                max_articles=config["num_articles"],
                device="cuda",
                sector2idx=sector2idx,
            )
            if config["verbose"]:
                print("     time taken to convert to tensor: ", time.time() - st_)

            optimizer.zero_grad()

            st_ = time.time()
            with autocast(device_type="cuda", dtype=torch.float16):
                y_ = model(**model_inputs)
                loss = criterion(y_.squeeze(), y)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            if config["verbose"]:
                print("     time taken to forward + backward: ", time.time() - st_)

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            st_ = time.time()

            y_true.extend(y.detach().view(-1).cpu().numpy())
            y_pred.extend(y_.detach().view(-1).cpu().numpy())
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        train_metrics = calculate_regression_metrics(y_true, y_pred)
        print("     Train Metrics: ", train_metrics)
        print(f"Epoch {epoch + 1} | Loss: {total_loss / len(train_loader):.4f}")
        train_losses.append(total_loss / len(train_loader))
        config["train_samples"] = len(train_loader) * config["batch_size"]

        model.eval()
        test_loss = 0.0
        y_true = []
        y_pred = []
        with torch.no_grad():
            test_loop = tqdm(test_loader, desc=f"Epoch {epoch + 1} | Test", leave=False)
            for X_test, y_test in test_loop:
                # y_test = torch.tensor(y_test, dtype=torch.float, device=device)
                model_inputs, y_test = prepare_batch(
                    X_test,
                    y_test,
                    max_articles=config["num_articles"],
                    device="cuda",
                    sector2idx=sector2idx,
                )

                with autocast(device_type="cuda", dtype=torch.float16):
                    # y_pred = model(X)
                    # loss = criterion(y_pred.squeeze(), y)
                    y_pred_test = model(**model_inputs)
                    loss_test = criterion(y_pred_test.squeeze(), y_test)

                test_loss += loss_test.item()
                test_loop.set_postfix(loss=f"{loss_test.item():.4f}")
                y_true.extend(y_test.detach().view(-1).cpu().numpy())
                y_pred.extend(y_pred_test.detach().view(-1).cpu().numpy())
            config["test_samples"] = len(test_loader) * config["batch_size"]
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        test_metrics = calculate_regression_metrics(y_true, y_pred)
        print("     Test Metrics: ", test_metrics)
        avg_test_loss = test_loss / len(test_loader)

        test_losses.append(avg_test_loss)

        if test_metrics["rmse"] < best_rmse:
            best_rmse = test_metrics["rmse"]
            torch.save(
                model.state_dict(),
                os.path.join(config["experiment_path"], "best_model.pth"),
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
    train_config = {"load_pre_trained": False}
    train(train_config)
