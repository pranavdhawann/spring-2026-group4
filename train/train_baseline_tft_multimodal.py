"""

Script to train FinBert model

"""
import copy
import gc
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
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataLoader import getTrainTestDataLoader
from src.models.TftMultiModalBaseline import MultiModalStockPredictor
from src.preProcessing import MultiModalPreProcessing
from src.utils import (
    calculate_regression_metrics,
    get_sector2Idx,
    read_json_file,
    read_yaml,
    set_seed,
)

scaler = GradScaler()  # for stable mixed precision


torch.cuda.empty_cache()
gc.collect()

# Verify memory is cleared
print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
print(f"Memory cached: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")


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


def save_prediction_plots(
    all_ts, all_actuals, all_preds, save_dir, num_plots=10, close_idx=3
):
    """Generate per-sample prediction plots showing input history + forecast."""
    os.makedirs(save_dir, exist_ok=True)
    total_samples = len(all_actuals)
    if total_samples == 0:
        return

    indices = np.linspace(0, total_samples - 1, num_plots, dtype=int)

    for plot_num, i in enumerate(indices):
        fig, ax = plt.subplots(figsize=(12, 6))

        input_close = all_ts[i][:, close_idx]
        seq_len = len(input_close)

        actual = all_actuals[i]
        predicted = all_preds[i]
        forecast_horizon = len(actual)

        forecast_x = np.arange(seq_len, seq_len + forecast_horizon)

        ax.plot(
            range(seq_len),
            input_close,
            label="Input (Close)",
            color="blue",
            linewidth=1.5,
        )
        ax.plot(
            forecast_x,
            actual,
            label="Actual Future",
            color="green",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax.plot(
            forecast_x,
            predicted,
            label="Predicted Future",
            color="red",
            linewidth=2,
            marker="x",
            markersize=6,
            linestyle="--",
        )

        ax.axvline(
            x=seq_len - 1,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label="Forecast Start",
        )

        ax.set_title(
            f"Stock Price Prediction (TFT+FinBERT Multimodal) - Sample {plot_num + 1}",
            fontsize=13,
        )
        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel("Normalized Price (Z-Score)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        plot_path = os.path.join(save_dir, f"prediction_sample_{plot_num + 1}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"     Saved prediction plot: {plot_path}")


def train(train_config):
    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/tft_multimodal",
        "load_pre_trained": False,
        "epochs": 30,
        "batch_size": 12,
        "max_length": 512,
        "lr": 1e-4,
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
        "max_window_size": 60,
        "num_articles": 4,
        "time_series_features": 12,
    }
    if not os.path.exists(config["experiment_path"]):
        os.mkdir(os.path.join(config["experiment_path"]))
    config.update(train_config)
    set_seed(config["rand_seed"])
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

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
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataloader,
        collate_fn=collator,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=3,
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

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=0.01
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
                print("+++++++++++++++++++++ X ++++++++++++++++++++++")
                print(model_inputs)
                print("[y] :: ", y)

            optimizer.zero_grad(set_to_none=True)

            st_ = time.time()
            y_ = model(**model_inputs)
            loss = criterion(y_, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

                y_pred_test = model(**model_inputs)
                loss_test = criterion(y_pred_test, y_test)

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

    # --- Final Evaluation: Generate sample prediction plots ---
    print("\n  Generating sample prediction plots...")
    model.eval()
    plot_ts_list = []
    plot_actual_list = []
    plot_pred_list = []
    with torch.no_grad():
        for X_test, y_test in test_loader:
            model_inputs, y_batch = prepare_batch(
                X_test,
                y_test,
                max_articles=config["num_articles"],
                device="cuda",
                sector2idx=sector2idx,
            )
            preds = model(**model_inputs)

            ts_np = model_inputs["time_series_features_"].cpu().numpy()
            actuals_np = y_batch.cpu().numpy()
            preds_np = preds.cpu().numpy()

            for j in range(ts_np.shape[0]):
                plot_ts_list.append(ts_np[j])
                plot_actual_list.append(actuals_np[j])
                plot_pred_list.append(preds_np[j])

    save_prediction_plots(
        plot_ts_list,
        plot_actual_list,
        plot_pred_list,
        save_dir=config["experiment_path"],
        num_plots=10,
    )


if __name__ == "__main__":
    train_config = {"load_pre_trained": False, "epochs": 20}
    train(train_config)
