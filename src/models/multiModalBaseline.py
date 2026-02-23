from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


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
            "max_articles", 32
        )  # Maximum articles to consider
        self.time_steps = config.get("max_window_size", 14)
        self.ts_features = config.get("time_series_features", 12)

        pretrained_finbert_path = config.get("bert_path", "ProsusAI/finbert")
        local_files_only = config.get("local_files_only", True)

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

        self.ts_encoder = nn.LSTM(  # < --change for time sereies
            input_size=self.ts_features,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        self.ticker_embed = (
            nn.Embedding(num_tickers, 32) if num_tickers else nn.Linear(768, 32)
        )
        self.sector_embed = (
            nn.Embedding(num_sectors, 16) if num_sectors else nn.Linear(768, 16)
        )

        fusion_dim = (
            self.finbert_hidden + 256 + 32 + 16
        )  # news + ts (128*2) + ticker + sector

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.predictor = nn.Linear(128, 7)  # Predict 7 days

    def forward(
        self,
        tokenized_news_: torch.Tensor,  # (batch, max_articles, token_len)
        attention_mask_news_: torch.Tensor,  # (batch, max_articles, token_len)
        time_series_features_: torch.Tensor,  # (batch, 14, 12)
        ticker_id_: torch.Tensor,  # (batch,)
        sector_: torch.Tensor,  # (batch,)
        article_mask_: Optional[
            torch.Tensor
        ] = None,  # (batch, max_articles) - indicates real vs padding
    ) -> torch.Tensor:
        batch_size, num_articles, token_len = tokenized_news_.shape
        # Reshape for FinBERT: (batch * num_articles, token_len)
        news_flat = tokenized_news_.reshape(-1, token_len)
        mask_flat = attention_mask_news_.reshape(-1, token_len)

        print(f"news_flat shape: {news_flat.shape}")
        print(f"mask_flat shape: {mask_flat.shape}")

        # with torch.no_grad():  # Freeze FinBERT to save memory
        news_out = self.finbert(input_ids=news_flat, attention_mask=mask_flat)

        news_features = news_out.last_hidden_state[
            :, 0, :
        ]  # (batch * num_articles, hidden)
        print(f"news_features before reshape: {news_features.shape}")

        news_features = news_features.reshape(batch_size, num_articles, -1)
        print(f"news_features after reshape: {news_features.shape}")

        if article_mask_ is not None:
            attn_out, _ = self.news_attention(
                news_features,
                news_features,
                news_features,
                key_padding_mask=~article_mask_.bool(),
            )

            attn_out = attn_out * article_mask_.unsqueeze(-1).float()

            article_counts = article_mask_.sum(dim=1, keepdim=True).float()
            article_counts = torch.clamp(article_counts, min=1.0)
            news_pooled = attn_out.sum(dim=1) / article_counts
        else:
            attn_out, _ = self.news_attention(
                news_features, news_features, news_features
            )
            news_pooled = attn_out.mean(dim=1)

        print(f"news_pooled shape: {news_pooled.shape}")

        #  time sereies changes
        ts_out, (h_n, _) = self.ts_encoder(time_series_features_)
        ts_features = ts_out[:, -1, :]
        print(f"ts_features shape: {ts_features.shape}")

        ticker_features = self.ticker_embed(ticker_id_)  # (batch, 32)
        sector_features = self.sector_embed(sector_)  # (batch, 16)
        print(f"ticker_features shape: {ticker_features.shape}")
        print(f"sector_features shape: {sector_features.shape}")

        combined = torch.cat(
            [news_pooled, ts_features, ticker_features, sector_features], dim=-1
        )
        print(f"combined shape: {combined.shape}")

        fused = self.fusion(combined)
        print(f"fused shape: {fused.shape}")

        output = self.predictor(fused)  # (batch, 7)
        print(f"output shape: {output.shape}")

        return output


if __name__ == "__main__":
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
        "experiment_path": "experiments/baseline/multiModal",
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
