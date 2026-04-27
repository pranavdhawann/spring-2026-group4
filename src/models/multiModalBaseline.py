import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.models.TcnMultiModalBaseline import TCNEncoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x * math.sqrt(x.size(2))
        x = x + self.pe[:, : x.size(1), :]
        return x


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, target_idx=3):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.target_idx = target_idx
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        mean = self.mean[:, 0, self.target_idx]
        stdev = self.stdev[:, 0, self.target_idx]
        if self.affine:
            weight = self.affine_weight[self.target_idx]
            bias = self.affine_bias[self.target_idx]
            x = x - bias
            x = x / (weight + self.eps * self.eps)
        x = x * stdev.unsqueeze(1)
        x = x + mean.unsqueeze(1)
        return x


def prepare_batch(X, y, device="cpu", max_articles=32, pad_token_id=0):
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

        # Select the most recent max_articles (from the end of the history window)
        num_to_take = min(len(articles), max_articles)
        start_idx = len(articles) - num_to_take

        for j in range(num_to_take):
            article_tensor = torch.tensor(articles[start_idx + j], dtype=torch.long)
            mask_tensor = torch.tensor(masks[start_idx + j], dtype=torch.long)
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


class MultiModalStockPredictor(nn.Module):
    def __init__(
        self,
        config: Dict,
        num_tickers: Optional[int] = None,
        num_sectors: Optional[int] = None,
    ):
        super(MultiModalStockPredictor, self).__init__()

        self.config = config
        self.max_articles = config.get(
            "num_articles", 32
        )  # Maximum articles to consider
        self.time_steps = config.get("max_window_size", 14)
        self.ts_features = config.get("time_series_features", 12)

        pretrained_finbert_path = config.get("bert_path", "ProsusAI/finbert")
        local_files_only = config.get("local_files_only", True)
        self.verbose = config.get("verbose", False)

        finbert_config = AutoConfig.from_pretrained(
            pretrained_finbert_path, local_files_only=local_files_only
        )
        self.finbert = AutoModel.from_pretrained(
            pretrained_finbert_path,
            config=finbert_config,
            local_files_only=local_files_only,
        )
        self.finbert_hidden = finbert_config.hidden_size

        self.news_attention = nn.MultiheadAttention(
            embed_dim=self.finbert_hidden, num_heads=4, batch_first=True
        )

        self.ts_encoder = TCNEncoder(
            {
                "input_size": self.ts_features,
                "num_channels": [64, 128, 128, 256],
                "kernel_size": 3,
                "dropout": 0.1,
                "embedding_size": 256,
            }
        )

        ts_hidden_dim = 256
        nhead = 8
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=nhead,
            dim_feedforward=256 * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.ts_transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=2
        )

        self.ts_pos_encoder = PositionalEncoding(d_model=ts_hidden_dim)

        self.ticker_embed = (
            nn.Embedding(num_tickers, 32) if num_tickers else nn.Linear(768, 32)
        )
        self.sector_embed = (
            nn.Embedding(num_sectors, 16) if num_sectors else nn.Linear(768, 16)
        )

        # Fusion Dimension
        # FinBERT (768) + TS (256) + Ticker (32) + Sector (16) = 1072
        fusion_dim = self.finbert_hidden + ts_hidden_dim + 32 + 16

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        # 3. Multimodal Fusion Layer
        self.fusion_hidden = 256
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, self.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 4. Final projection: Single-shot output for all 5 days
        self.fc_out = nn.Linear(self.fusion_hidden, 5)

        self.revin = RevIN(
            num_features=self.ts_features, target_idx=config.get("target_idx", 3)
        )

    def forward(
        self,
        tokenized_news_: torch.Tensor,  # (batch, max_articles, token_len)
        attention_mask_news_: torch.Tensor,  # (batch, max_articles, token_len)
        time_series_features_: torch.Tensor,  # (batch, seq_len, ts_feat)
        ticker_id_: torch.Tensor,  # (batch,)
        sector_: torch.Tensor,  # (batch,)
        article_mask_: Optional[torch.Tensor] = None,  # (batch, num_articles)
    ) -> torch.Tensor:
        batch_size, num_articles, token_len = tokenized_news_.shape

        # --- A. NEWS ENCODING (FinBERT) ---
        news_flat = tokenized_news_.reshape(-1, token_len)
        mask_flat = attention_mask_news_.reshape(-1, token_len)

        news_out = self.finbert(input_ids=news_flat, attention_mask=mask_flat)
        news_features = news_out.last_hidden_state[:, 0, :]
        news_features = news_features.reshape(batch_size, num_articles, -1)

        # News Attention / Pooling
        if article_mask_ is not None:
            attn_out, _ = self.news_attention(
                news_features,
                news_features,
                news_features,
                key_padding_mask=~article_mask_.bool(),
            )
            attn_out = attn_out * article_mask_.unsqueeze(-1).float()
            article_counts = torch.clamp(
                article_mask_.sum(dim=1, keepdim=True).float(), min=1.0
            )
            news_pooled = attn_out.sum(dim=1) / article_counts  # (batch, 768)
        else:
            attn_out, _ = self.news_attention(
                news_features, news_features, news_features
            )
            news_pooled = attn_out.mean(dim=1)

        ts_norm = self.revin(time_series_features_, "norm")

        ts_seq = self.ts_encoder(ts_norm)

        ts_seq = self.ts_pos_encoder(ts_seq)

        ts_transformer_out = self.ts_transformer_encoder(ts_seq)

        ts_context_vec = ts_transformer_out[:, -1, :]

        target_idx = self.revin.target_idx
        last_known = ts_norm[:, -1, target_idx].unsqueeze(1)

        ticker_features = self.ticker_embed(ticker_id_)
        sector_features = self.sector_embed(sector_)

        combined = torch.cat(
            [news_pooled, ts_context_vec, ticker_features, sector_features], dim=-1
        )

        fused_norm = self.fusion_norm(combined)

        fusion_summary = self.fusion_bottleneck(fused_norm)

        diff_pred = self.fc_out(fusion_summary)

        out_norm = diff_pred + last_known

        out = self.revin(out_norm, "denorm")

        return out


if __name__ == "__main__":
    print("MULTIMODAL TRANSFORMER MODEL TEST")

    # Mock Config
    mock_config = {
        "bert_path": "ProsusAI/finbert",
        "local_files_only": True,
        "max_window_size": 14,
        "num_articles": 8,
        "time_series_features": 12,
        "verbose": False,
    }

    # Mock Model
    print("Initializing model (Mocking FinBERT)...")
    from unittest.mock import MagicMock

    # Mock Inputs
    batch_size = 4
    num_articles = 8
    token_len = 512
    seq_len = 14
    ts_feat = 12

    # Pre-mocking to avoid AutoModel.from_pretrained errors
    original_from_pretrained = AutoModel.from_pretrained
    original_config_from_pretrained = AutoConfig.from_pretrained

    # Create a dummy config
    mock_bert_config = MagicMock()
    mock_bert_config.hidden_size = 768
    AutoConfig.from_pretrained = MagicMock(return_value=mock_bert_config)

    # Create a dummy model
    mock_bert_model = MagicMock()
    # Mocking the return value of the forward pass
    mock_last_hidden_state = torch.randn(batch_size * num_articles, token_len, 768)
    mock_news_out = MagicMock()
    mock_news_out.last_hidden_state = mock_last_hidden_state
    mock_bert_model.return_value = mock_news_out

    AutoModel.from_pretrained = MagicMock(return_value=mock_bert_model)

    try:
        model = MultiModalStockPredictor(mock_config, num_tickers=100, num_sectors=10)

        # Mock Inputs
        tokenized_news = torch.zeros(
            (batch_size, num_articles, token_len), dtype=torch.long
        )
        attention_mask = torch.ones(
            (batch_size, num_articles, token_len), dtype=torch.long
        )
        time_series = torch.randn((batch_size, seq_len, ts_feat))
        ticker_ids = torch.zeros(batch_size, dtype=torch.long)
        sectors = torch.zeros(batch_size, dtype=torch.long)
        article_mask = torch.ones((batch_size, num_articles), dtype=torch.bool)

        print("\nRunning forward pass with mock data...")
        output = model(
            tokenized_news_=tokenized_news,
            attention_mask_news_=attention_mask,
            time_series_features_=time_series,
            ticker_id_=ticker_ids,
            sector_=sectors,
            article_mask_=article_mask,
        )
        print(f"  Output shape: {output.shape}")
        assert output.shape == (
            batch_size,
            5,
        ), f"Expected (batch_size, 5), got {output.shape}"
        print("Passed!")
    finally:
        # Restore originals
        AutoModel.from_pretrained = original_from_pretrained
        AutoConfig.from_pretrained = original_config_from_pretrained

    print("\nStarting full data-loading test (requires local dataset)...")
    import os
    import pickle

    import numpy as np
    from torch.utils.data import DataLoader

    from src.dataLoader import getTrainTestDataLoader
    from src.preProcessing import MultiModalPreProcessing
    from src.utils import get_sector2Idx, read_json_file, read_yaml

    # Configuration
    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/multimodal",
        "load_pre_trained": False,
        "batch_size": 32,
        "max_length": 512,
        "bert_path": "ProsusAI/finbert",
        "local_files_only": True,
        "sample_fraction": 1,
        "rand_seed": 42,
        "verbose": False,
        "max_window_size": 14,
        "num_articles": 18,
        "time_series_features": 12,
    }

    if not os.path.exists(config["experiment_path"]):
        os.makedirs(config["experiment_path"])

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

    dataloaders_path = os.path.join(config["experiment_path"], "dataloaders.pkl")
    if not os.path.exists(dataloaders_path):
        train_dataloader, test_dataloader = getTrainTestDataLoader(config)
        with open(dataloaders_path, "wb") as f:
            pickle.dump({"train": train_dataloader, "test": test_dataloader}, f)
        print("Dataloaders saved!")
    else:
        with open(dataloaders_path, "rb") as f:
            dataloaders = pickle.load(f)
        train_dataloader = dataloaders["train"]
        test_dataloader = dataloaders["test"]
        print("Dataloaders loaded successfully!")

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

    print("\n" + "=" * 50)
    print("Inspecting actual data batch:")
    print("=" * 50)
    for X, y in train_loader:
        print(f"batch_size: {len(X)}")
        print(f"each data has: {X[0].keys()}")
        print(f"article: {len(X[0]['tokenized_news_'])}")
        print(f"token len: {len(X[0]['tokenized_news_'][0])}")
        print(f"time series: {len(X[0]['time_series_features_'])}")
        print(f"time series features len: {len(X[0]['time_series_features_'][0])}")
        print(f"target length: {len(y[0])}")
        print(f"target[0]: {y[0]}")

        actual_batch_size = len(X)
        actual_num_articles = len(X[0]["tokenized_news_"])
        actual_token_len = len(X[0]["tokenized_news_"][0])
        actual_time_steps = len(X[0]["time_series_features_"])
        actual_ts_features = len(X[0]["time_series_features_"][0])
        actual_target_len = len(y[0])

        print("\nActual shapes:")
        print(f"  - Batch size: {actual_batch_size}")
        print(f"  - Num articles: {actual_num_articles}")
        print(f"  - Token length: {actual_token_len}")
        print(f"  - Time steps: {actual_time_steps}")
        print(f"  - TS features: {actual_ts_features}")
        print(f"  - Target length: {actual_target_len}")
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = MultiModalStockPredictor(
        config=config,
        num_tickers=len(ticker2idx),  # Actual number of tickers
        num_sectors=len(sector2idx)
        if "sector2idx" in locals()
        else 10,  # Actual sectors
    )
    model = model.to(device)

    print("\nModel created with:")
    print(f"  - Num tickers: {len(ticker2idx)}")
    print(f"  - Num sectors: {len(sector2idx) if 'sector2idx' in locals() else 10}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 50)
    print("Testing forward pass with actual batch:")
    print("=" * 50)

    counter = 0
    for X, y in train_loader:
        counter += 1
        model_inputs, targets = prepare_batch(
            X, y, device, max_articles=config["num_articles"]
        )

        with torch.no_grad():
            output = model(**model_inputs)
            print(output)
        print(f"Batch {counter}")
        print("Input shapes:")
        print(f"  tokenized_news_: {model_inputs['tokenized_news_'].shape}")
        print(f"  attention_mask_news_: {model_inputs['attention_mask_news_'].shape}")
        print(f"  time_series_features_: {model_inputs['time_series_features_'].shape}")
        print(f"  ticker_id_: {model_inputs['ticker_id_'].shape}")
        print(f"  sector_: {model_inputs['sector_'].shape}")
        print(f"  targets: {targets.shape}")
        print(f"\nOutput shape: {output.shape}")

        if counter >= 15:
            break
