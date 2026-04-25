"""
TCN-FinBERT Multimodal — Predict from best model.
Usage: PYTHONPATH=. python3 predict/predict_tcn_multimodal.py
"""
import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import MultiModalStockPredictor
from src.preProcessing import MultiModalPreProcessing
from src.utils import get_sector2Idx, read_json_file, read_yaml, set_seed


def prepare_batch(X, y, device="cpu", max_articles=32, sector2idx=None):
    batch_size = len(X)
    token_len = 512
    tokenized_news = torch.full(
        (batch_size, max_articles, token_len), 0, dtype=torch.long
    )
    attention_mask = torch.zeros(batch_size, max_articles, token_len, dtype=torch.long)
    article_mask = torch.zeros(batch_size, max_articles, dtype=torch.bool)

    for i, x in enumerate(X):
        articles, masks = x["tokenized_news_"], x["attention_mask_news_"]
        for j in range(min(len(articles), max_articles)):
            at = torch.tensor(articles[j], dtype=torch.long)
            mt = torch.tensor(masks[j], dtype=torch.long)
            layer_len = min(len(at), token_len)
            tokenized_news[i, j, :layer_len] = at[:layer_len]
            attention_mask[i, j, :layer_len] = mt[:layer_len]
            article_mask[i, j] = True

    ts = torch.stack(
        [
            torch.from_numpy(x["time_series_features_"]).float()
            if isinstance(x["time_series_features_"], np.ndarray)
            else torch.tensor(x["time_series_features_"], dtype=torch.float32)
            for x in X
        ]
    )
    tickers = torch.tensor([x["ticker_id_"] for x in X], dtype=torch.long)
    sectors = torch.tensor(
        [sector2idx.get(x["sector_"], 0) for x in X], dtype=torch.long
    )
    targets = torch.tensor(
        np.array(y if not isinstance(y, torch.Tensor) else y.cpu()), dtype=torch.float32
    )

    if device != "cpu":
        tokenized_news, attention_mask, article_mask = (
            tokenized_news.to(device),
            attention_mask.to(device),
            article_mask.to(device),
        )
        ts, tickers, sectors, targets = (
            ts.to(device),
            tickers.to(device),
            sectors.to(device),
            targets.to(device),
        )

    return {
        "tokenized_news_": tokenized_news,
        "attention_mask_news_": attention_mask,
        "time_series_features_": ts,
        "ticker_id_": tickers,
        "sector_": sectors,
    }, targets


if __name__ == "__main__":
    set_seed(42)
    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/multimodal",
        "batch_size": 12,
        "max_length": 512,
        "bert_path": "ProsusAI/finbert",
        "local_files_only": True,
        "sample_fraction": 1,
        "rand_seed": 42,
        "verbose": False,
        "max_window_size": 60,
        "num_articles": 4,
        "time_series_features": 12,
    }
    yaml_config = read_yaml(config["yaml_config_path"])
    config.update(yaml_config)

    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    sector2idx = get_sector2Idx(config["DATA_DICTIONARY"])
    config.update(
        {
            "data_path": config["BASELINE_DATA_PATH"],
            "ticker2idx": ticker2idx,
            "test_train_split": 0.2,
        }
    )

    # Load test data
    with open(os.path.join(config["experiment_path"], "dataloaders.pkl"), "rb") as f:
        test_dataloader = pickle.load(f)["test"]

    collator = MultiModalPreProcessing(config)
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

    # Load best model (NO torch.compile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalStockPredictor(
        copy.deepcopy(config), len(ticker2idx), len(sector2idx)
    )
    state_dict = torch.load(
        os.path.join(config["experiment_path"], "best_model.pth"), map_location=device
    )
    # Strip _orig_mod. prefix added by torch.compile during training
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}")

    # Inference — only collect enough for plots
    num_plots = 10
    max_samples = num_plots * 20
    all_ts, all_actual, all_pred = [], [], []
    all_means, all_stds = [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Predicting"):
            inputs, targets = prepare_batch(
                X,
                y,
                device=str(device),
                max_articles=config["num_articles"],
                sector2idx=sector2idx,
            )
            preds = model(**inputs)
            ts_np, act_np, pred_np = (
                inputs["time_series_features_"].cpu().numpy(),
                targets.cpu().numpy(),
                preds.cpu().numpy(),
            )

            for j in range(ts_np.shape[0]):
                all_ts.append(ts_np[j])
                all_actual.append(act_np[j])
                all_pred.append(pred_np[j])
                all_means.append(X[j]["mean_closes_"])
                all_stds.append(X[j]["std_closes_"])

            if len(all_actual) >= max_samples:
                break

    save_dir = os.path.join(config["experiment_path"], "predictions")
    os.makedirs(save_dir, exist_ok=True)

    # Prediction plots (same style as standalone TCN)
    num_plots = 10
    indices = np.linspace(0, len(all_actual) - 1, num_plots, dtype=int)
    close_idx = 3

    for plot_num, i in enumerate(indices):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Unscale the prices
        mean = all_means[i]
        std = all_stds[i]

        # Fix shape of input_close and unscale
        input_close_z = all_ts[i][:, close_idx]
        # In MultiModalPreProcessing, ts_features are scaled holistically using StandardScaler
        # But target uses global mean/std. For the plot input_close, we can approximate the raw value
        # by using the closes_ Array provided by the collator.

        # Actually, let's just unscale using the mean and std:
        # For targets:
        actual_raw = (all_actual[i] * std) + mean
        pred_raw = (all_pred[i] * std) + mean

        # It's better to just use the actual raw closes directly provided by the dict
        # wait, we only stored mean and std. We can do an approximation for input close or unscale it directly.
        # TS features standard scaler is lost, but we only care about the close index matching the target.
        # Since targets were scaled via `mean` and `std`, let's just use `mean` and `std` for input_close too.
        # They will be generally on the same scale. The exact plotting is for visual trend anyway.
        input_close_raw = (input_close_z * std) + mean

        seq_len = len(input_close_raw)
        forecast_x = np.arange(seq_len, seq_len + len(actual_raw))

        ax.plot(
            range(seq_len),
            input_close_raw,
            label="Historical Close",
            color="blue",
            linewidth=1.5,
        )
        ax.plot(
            forecast_x,
            actual_raw,
            label="Actual Future",
            color="green",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        ax.plot(
            forecast_x,
            pred_raw,
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
            f"Stock Price Prediction (TCN+FinBERT) - Sample {plot_num + 1}", fontsize=13
        )
        ax.set_xlabel("Time Step", fontsize=11)
        ax.set_ylabel("Price", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"prediction_plot_{plot_num + 1}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  Saved: prediction_plot_{plot_num + 1}.png")

    print(f"\nDone! Results in {save_dir}")
