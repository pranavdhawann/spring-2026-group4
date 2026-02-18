from typing import Dict

import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModel, AutoTokenizer

from src.utils import set_seed


class FinBertForecastingBL(nn.Module):
    def __init__(self, config: Dict):
        super(FinBertForecastingBL, self).__init__()
        set_seed()

        self.config = {
            "finbert_name_or_path": "ProsusAI/finbert",
            "device": torch.device("cpu"),
            "local_files_only": True,
            "bert_hidden_dim": 768,
            "news_embedding_dim": 256,
            "FORECAST_HORIZON": 7,
            "freeze_bert": True,
            "mlp_hidden_dims": [128, 64],
            "dropout_rate": 0.2,
            "max_window_size": 10,
            "empty_news_threshold": 2,
        }

        self.config.update(config)
        self.device = self.config["device"]

        self.current_epoch = 0
        self.total_epochs = self.config.get("num_epochs", 100)
        self.max_window_size = self.config.get("max_window_size", 10)
        self.empty_news_threshold = self.config.get("empty_news_threshold", 2)

        self.finbert = AutoModel.from_pretrained(
            self.config["finbert_name_or_path"],
            local_files_only=self.config["local_files_only"],
        )

        if self.config.get("freeze_bert", True):
            for param in self.finbert.parameters():
                param.requires_grad = False

        self.news_projection = nn.Linear(
            self.config["bert_hidden_dim"], self.config["news_embedding_dim"]
        )

        mlp_dims = self.config["mlp_hidden_dims"]
        dropout_rate = self.config["dropout_rate"]
        input_dim = self.config["news_embedding_dim"] + 3

        mlp_layers = []
        for hidden_dim in mlp_dims:
            mlp_layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            )
            input_dim = hidden_dim

        # Final output layer
        mlp_layers.append(nn.Linear(input_dim, self.config["FORECAST_HORIZON"]))
        self.mlp_regressor = nn.Sequential(*mlp_layers)
        self.to(self.device)

    def is_empty_news(self, input_ids: torch.Tensor) -> torch.Tensor:
        non_padding = (input_ids != 0).sum(dim=-1)  # (B, W)
        return non_padding <= self.empty_news_threshold

    def find_latest_news_indices(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, W, L = input_ids.shape

        is_empty = self.is_empty_news(input_ids)  # (B, W)
        positions = torch.arange(W, device=input_ids.device).expand(B, W)  # (B, W)
        valid_positions = torch.where(~is_empty, positions, -1)  # (B, W)
        latest_indices = valid_positions.max(dim=1)[0]  # (B,)
        return latest_indices

    def forward(self, inputs):
        input_ids = inputs["input_ids"]  # (B, W, L)
        attention_mask = inputs["attention_mask"]  # (B, W, L)
        closes = inputs["closes"]  # (B, N)
        extra_features = inputs["extra_features"]  # (B, 2)

        B, W, L = input_ids.shape

        last_close = closes[:, -1].unsqueeze(1)  # (B, 1)
        combined_features = torch.cat([last_close, extra_features], dim=1)  # (B, 3)
        latest_indices = self.find_latest_news_indices(input_ids)  # (B,)
        has_news_mask = latest_indices >= 0  # (B,)
        news_embeddings = torch.zeros(
            B, self.config["bert_hidden_dim"], device=self.device
        )

        if has_news_mask.any():
            news_batch_indices = torch.where(has_news_mask)[0]  # (M,)
            news_window_indices = latest_indices[news_batch_indices]  # (M,)
            latest_input_ids = input_ids[
                news_batch_indices, news_window_indices
            ]  # (M, L)
            latest_attention_mask = attention_mask[
                news_batch_indices, news_window_indices
            ]  # (M, L)

            with autocast(device_type="cuda"):
                bert_outputs = self.finbert(
                    input_ids=latest_input_ids,
                    attention_mask=latest_attention_mask,
                )

            cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # (M, 768)
            news_embeddings[news_batch_indices] = cls_embeddings

        projected_news = self.news_projection(
            news_embeddings
        )  # (B, news_embedding_dim)
        combined = torch.cat(
            [projected_news, combined_features], dim=1
        )  # (B, news_embedding_dim + 3)
        predictions = self.mlp_regressor(combined)  # (B, FORECAST_HORIZON)

        return predictions


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "finbert_name_or_path": "ProsusAI/finbert",
        "device": device,
        "FORECAST_HORIZON": 7,
        "news_embedding_dim": 256,
        "mlp_hidden_dims": [128, 64],
        "dropout_rate": 0.2,
        "freeze_bert": True,
        "max_window_size": 14,
        "empty_news_threshold": 2,
    }

    model = FinBertForecastingBL(config)
    tokenizer = AutoTokenizer.from_pretrained(config["finbert_name_or_path"])

    print("\n" + "=" * 50)
    print("TESTING VECTORIZED EMPTY NEWS DETECTION")
    print("=" * 50)

    texts = [
        "Apple stock rose 3% after strong earnings report.",  # Real news
        "",  # Empty string
        "Tesla shares drop following delivery miss.",  # Real news
        "",  # Empty string
        "Market volatility increases.",  # Real news
    ]

    tokenized = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    window_size = 3
    batch_size = len(texts)

    input_ids = tokenized["input_ids"].unsqueeze(1).repeat(1, window_size, 1)
    attention_mask = tokenized["attention_mask"].unsqueeze(1).repeat(1, window_size, 1)

    print(f"\nInput shape: {input_ids.shape}")

    print("\nVectorized empty news detection:")
    is_empty = model.is_empty_news(input_ids)
    for b in range(batch_size):
        for w in range(window_size):
            status = "EMPTY" if is_empty[b, w] else "HAS NEWS"
            tokens = (input_ids[b, w] != 0).sum().item()
            print(f"  Batch {b}, Window {w}: {status} ({tokens} tokens)")

    latest_indices = model.find_latest_news_indices(input_ids)
    print(f"\nLatest news indices: {latest_indices.cpu().numpy()}")

    has_news = latest_indices >= 0
    print(f"Batches with news: {has_news.cpu().numpy()}")

    dummy_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "closes": torch.randn(batch_size, 30),
        "extra_features": torch.randn(batch_size, 2),
    }

    try:
        compiled_model = torch.compile(model)
        with torch.no_grad():
            predictions = compiled_model(dummy_inputs)
        print("\n✓ torch.compile successful - no graph breaks!")
        print(f"Predictions shape: {predictions.shape}")
    except Exception as e:
        print(f"\n✗ torch.compile failed: {e}")
