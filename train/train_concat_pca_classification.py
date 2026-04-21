"""
Train next-day direction classifier using concatenated embeddings:
TCN + TabNet + FinBERT -> concat -> StandardScaler -> PCA -> LogisticRegression.
"""

import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel

from src.dataLoader import getTrainTestDataLoader
from src.models.TcnMultiModalBaseline import TCNEncoder
from src.preProcessing import MultiModalPreProcessing
from src.utils import get_sector2Idx, read_json_file, read_yaml


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_fraction_subset(dataset, fraction=1.0, seed=42):
    if fraction >= 1.0:
        return dataset
    dataset_size = len(dataset)
    subset_size = max(1, int(dataset_size * fraction))
    rng = random.Random(seed)
    indices = rng.sample(range(dataset_size), subset_size)
    return Subset(dataset, indices)


def prepare_batch(X, y, device="cpu", max_articles=2, pad_token_id=0, sector2idx=None):
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

    time_series = torch.stack(
        [
            torch.from_numpy(x["time_series_features_"]).float()
            if isinstance(x["time_series_features_"], np.ndarray)
            else torch.tensor(x["time_series_features_"], dtype=torch.float32)
            for x in X
        ]
    )

    ticker_ids = torch.tensor([x["ticker_id_"] for x in X], dtype=torch.long)
    sectors = torch.tensor(
        [sector2idx[x["sector_"]] if x["sector_"] in sector2idx else 0 for x in X],
        dtype=torch.long,
    )

    if isinstance(y, np.ndarray):
        targets = torch.from_numpy(y).float()
    else:
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
        "article_mask_": article_mask,
        "time_series_features_": time_series,
        "ticker_id_": ticker_ids,
        "sector_": sectors,
    }, targets


class ConcatFeatureExtractor(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.max_articles = config.get("num_articles", 2)
        self.time_steps = config.get("max_window_size", 14)
        self.ts_features = config.get("time_series_features", 12)

        finbert_path = config.get("bert_path", "ProsusAI/finbert")
        local_files_only = config.get("local_files_only", True)
        finbert_cfg = AutoConfig.from_pretrained(
            finbert_path, local_files_only=local_files_only
        )
        self.finbert = AutoModel.from_pretrained(
            finbert_path,
            config=finbert_cfg,
            local_files_only=local_files_only,
        )
        self.finbert_proj = nn.Linear(finbert_cfg.hidden_size, 256)

        self.tcn_encoder = TCNEncoder(
            {
                "input_size": self.ts_features,
                "num_channels": [64, 128, 256],
                "kernel_size": 3,
                "dropout": 0.2,
                "embedding_size": 256,
            }
        )

        # TabNet on tabularized time-series (flattened) + ticker + sector.
        tab_in = (self.time_steps * self.ts_features) + 2
        from pytorch_tabnet.tab_network import TabNetNoEmbeddings

        self.tabnet = TabNetNoEmbeddings(
            input_dim=tab_in,
            output_dim=256,
            n_d=64,
            n_a=64,
            n_steps=3,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
        )
        self.to(self.device)

    @torch.no_grad()
    def forward(self, model_inputs):
        tokenized_news = model_inputs["tokenized_news_"]  # (B, A, L)
        attention_mask = model_inputs["attention_mask_news_"]  # (B, A, L)
        article_mask = model_inputs["article_mask_"]  # (B, A)
        ts = model_inputs["time_series_features_"]  # (B, T, F)
        ticker = model_inputs["ticker_id_"]  # (B,)
        sector = model_inputs["sector_"]  # (B,)

        B, A, L = tokenized_news.shape

        # FinBERT embedding (256): process each article then average with article mask.
        ids_flat = tokenized_news.reshape(B * A, L)
        mask_flat = attention_mask.reshape(B * A, L)
        bert_out = self.finbert(input_ids=ids_flat, attention_mask=mask_flat)
        cls = bert_out.last_hidden_state[:, 0, :].reshape(B, A, -1)

        art_mask = article_mask.unsqueeze(-1).float()
        cls = cls * art_mask
        counts = art_mask.sum(dim=1).clamp_min(1.0)
        finbert_embed = cls.sum(dim=1) / counts
        finbert_embed = self.finbert_proj(finbert_embed)  # (B, 256)

        # TCN embedding (256).
        tcn_embed = self.tcn_encoder(ts)  # (B, 256)

        # TabNet embedding (256).
        ts_flat = ts.reshape(B, -1)  # (B, T*F)
        tab_input = torch.cat(
            [ts_flat, ticker.float().unsqueeze(1), sector.float().unsqueeze(1)], dim=1
        )

        if hasattr(self.tabnet, "encoder") and hasattr(
            self.tabnet.encoder, "group_attention_matrix"
        ):
            gm = self.tabnet.encoder.group_attention_matrix
            if gm.device != tab_input.device:
                self.tabnet.encoder.group_attention_matrix = gm.to(tab_input.device)

        torch.use_deterministic_algorithms(False)
        tabnet_embed, _ = self.tabnet(tab_input)  # (B, 256)

        fused = torch.cat([tcn_embed, tabnet_embed, finbert_embed], dim=1)  # (B, 768)
        return fused


def extract_features(loader, extractor, device, sector2idx, max_articles):
    X_list = []
    y_list = []
    for X_batch, y_batch in tqdm(loader, leave=False):
        model_inputs, targets = prepare_batch(
            X_batch,
            y_batch,
            device=device.type,
            max_articles=max_articles,
            sector2idx=sector2idx,
        )
        with torch.no_grad():
            feats = extractor(model_inputs)  # (B, 768)

        # next-day direction label from standardized target horizon 0
        labels = (targets[:, 0] > 0).long()
        X_list.append(feats.detach().cpu().numpy())
        y_list.append(labels.detach().cpu().numpy())

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all


def train(train_config):
    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/concat_pca_classification",
        "batch_size": 64,
        "max_length": 512,
        "local_files_only": True,
        "rand_seed": 42,
        "max_window_size": 14,
        "num_articles": 2,
        "time_series_features": 12,
        "pca_components": 0.95,  # keep 95% variance
        "sample_fraction": 0.25,
    }
    config.update(train_config)
    seed_everything(config["rand_seed"])
    os.makedirs(config["experiment_path"], exist_ok=True)

    yaml_config = read_yaml(config["yaml_config_path"])
    config.update(yaml_config)

    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    sector2idx = get_sector2Idx(config["DATA_DICTIONARY"])

    data_config = {
        "data_path": config["BASELINE_DATA_PATH"],
        "ticker2idx": ticker2idx,
        "test_train_split": 0.2,
    }
    config.update(data_config)

    train_dataset, test_dataset = getTrainTestDataLoader(config)
    train_dataset = get_fraction_subset(
        train_dataset, fraction=config["sample_fraction"], seed=config["rand_seed"]
    )
    test_dataset = get_fraction_subset(
        test_dataset, fraction=config["sample_fraction"], seed=config["rand_seed"]
    )
    collator = MultiModalPreProcessing(config)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=collator,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        collate_fn=collator,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = ConcatFeatureExtractor(config, device=device).eval()

    print("Extracting train features...")
    X_train, y_train = extract_features(
        train_loader, extractor, device, sector2idx, config["num_articles"]
    )
    print("Extracting test features...")
    X_test, y_test = extract_features(
        test_loader, extractor, device, sector2idx, config["num_articles"]
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=config["pca_components"], random_state=config["rand_seed"])
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    clf = LogisticRegression(max_iter=2000, random_state=config["rand_seed"])
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "input_feature_dim": int(X_train.shape[1]),
        "pca_feature_dim": int(X_train_pca.shape[1]),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    print("Metrics:", metrics)

    with open(os.path.join(config["experiment_path"], "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(config["experiment_path"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print("Saved metrics/config only to:", config["experiment_path"])


if __name__ == "__main__":
    train_config = {}
    train(train_config)

