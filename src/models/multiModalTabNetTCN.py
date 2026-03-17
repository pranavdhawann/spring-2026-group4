from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.TcnMultiModalBaseline import TCNEncoder
from src.utils import set_seed


class TabNetTCNMultiModal(nn.Module):
    """
    Multimodal model: simple text encoder + TCN time-series encoder + TabNet head.

    - Text: token embedding + pooling over tokens and articles.
    - Time-series: TCNEncoder over price/indicator window.
    - Categorical: ticker and sector embeddings.
    - Fusion: concatenate all features and feed into a TabNet block, then a linear
      layer to forecast the next 7 days.
    """

    def __init__(
        self,
        config: Dict,
        num_tickers: Optional[int] = None,
        num_sectors: Optional[int] = None,
    ):
        super().__init__()
        set_seed()

        self.config = {
            "device": torch.device("cpu"),
            "vocab_size": 30522,
            "token_embed_dim": 128,
            "news_embedding_dim": 256,
            "FORECAST_HORIZON": 7,
            "max_window_size": 14,
            "time_series_features": 12,
            "num_articles": 2,
            "tabnet_n_d": 64,
            "tabnet_n_a": 64,
            "tabnet_n_steps": 3,
            "tabnet_gamma": 1.5,
            "tabnet_n_independent": 2,
            "tabnet_n_shared": 2,
        }
        self.config.update(config)

        self.device = self.config.get("device", torch.device("cpu"))
        self.max_articles = self.config.get("num_articles", 2)
        self.time_steps = self.config.get("max_window_size", 14)
        self.ts_features = self.config.get("time_series_features", 12)

        # Text encoder: token embedding + pooling
        self.token_embedding = nn.Embedding(
            num_embeddings=self.config["vocab_size"],
            embedding_dim=self.config["token_embed_dim"],
            padding_idx=0,
        )

        self.news_projection = nn.Linear(
            self.config["token_embed_dim"], self.config["news_embedding_dim"]
        )

        # Time-series encoder: TCN
        self.ts_encoder = TCNEncoder(
            {
                "input_size": self.ts_features,
                "num_channels": [64, 128, 256],
                "kernel_size": 3,
                "dropout": 0.2,
                "embedding_size": 256,
            }
        )

        # Categorical embeddings
        self.ticker_embed = (
            nn.Embedding(num_tickers, 32) if num_tickers else nn.Identity()
        )
        self.sector_embed = (
            nn.Embedding(num_sectors, 16) if num_sectors else nn.Identity()
        )

        ticker_dim = 32 if num_tickers else 0
        sector_dim = 16 if num_sectors else 0

        # TabNet head over concatenated features (continuous-only: use TabNetNoEmbeddings
        # to avoid EmbeddingGenerator and group_matrix issues).
        fused_dim = self.config["news_embedding_dim"] + 256 + ticker_dim + sector_dim

        from pytorch_tabnet.tab_network import TabNetNoEmbeddings

        self.tabnet = TabNetNoEmbeddings(
            input_dim=fused_dim,
            output_dim=128,
            n_d=self.config["tabnet_n_d"],
            n_a=self.config["tabnet_n_a"],
            n_steps=self.config["tabnet_n_steps"],
            gamma=self.config["tabnet_gamma"],
            n_independent=self.config["tabnet_n_independent"],
            n_shared=self.config["tabnet_n_shared"],
        )

        self.predictor = nn.Linear(128, self.config["FORECAST_HORIZON"])

        self.to(self.device)

    def encode_news(
        self,
        tokenized_news_: torch.Tensor,
        attention_mask_news_: Optional[torch.Tensor] = None,
        article_mask_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode news articles with token embeddings and mean pooling.

        tokenized_news_: (batch, max_articles, token_len)
        attention_mask_news_: currently unused; pooling uses token != 0.
        article_mask_: (batch, max_articles) indicating real articles.
        """
        batch_size, num_articles, token_len = tokenized_news_.shape

        tokens = tokenized_news_.reshape(batch_size * num_articles, token_len)
        token_emb = self.token_embedding(tokens)  # (B*A, L, D)

        token_mask = (tokens != 0).unsqueeze(-1)  # (B*A, L, 1)
        summed = (token_emb * token_mask).sum(dim=1)  # (B*A, D)
        counts = token_mask.sum(dim=1).clamp_min(1)  # (B*A, 1)
        pooled = summed / counts  # (B*A, D)

        pooled = pooled.reshape(batch_size, num_articles, -1)  # (B, A, D)

        if article_mask_ is not None:
            article_mask = article_mask_.unsqueeze(-1).float()  # (B, A, 1)
            pooled = pooled * article_mask
            counts_articles = article_mask.sum(dim=1).clamp_min(1.0)
            news_vec = pooled.sum(dim=1) / counts_articles  # (B, D)
        else:
            news_vec = pooled.mean(dim=1)  # (B, D)

        news_vec = self.news_projection(news_vec)  # (B, news_embedding_dim)
        return news_vec

    def forward(
        self,
        tokenized_news_: torch.Tensor,
        attention_mask_news_: torch.Tensor,
        time_series_features_: torch.Tensor,
        ticker_id_: torch.Tensor,
        sector_: torch.Tensor,
        article_mask_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Text encoding
        news_features = self.encode_news(
            tokenized_news_, attention_mask_news_, article_mask_
        )  # (B, news_embedding_dim)

        # Time-series encoding
        ts_features = self.ts_encoder(time_series_features_)  # (B, 256)

        # Categorical embeddings
        parts = [news_features, ts_features]
        if isinstance(self.ticker_embed, nn.Embedding):
            parts.append(self.ticker_embed(ticker_id_))  # (B, 32)
        if isinstance(self.sector_embed, nn.Embedding):
            parts.append(self.sector_embed(sector_))  # (B, 16)

        fused = torch.cat(parts, dim=-1)  # (B, fused_dim)

        # pytorch_tabnet encoder keeps group_attention_matrix on CPU; ensure same device as input
        if hasattr(self.tabnet, "encoder") and hasattr(
            self.tabnet.encoder, "group_attention_matrix"
        ):
            gm = self.tabnet.encoder.group_attention_matrix
            if gm.device != fused.device:
                self.tabnet.encoder.group_attention_matrix = gm.to(fused.device)

        tabnet_out, _ = self.tabnet(fused)  # (B, 128)
        output = self.predictor(tabnet_out)  # (B, FORECAST_HORIZON)

        return output


__all__ = ["TabNetTCNMultiModal"]

