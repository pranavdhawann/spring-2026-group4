from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.models.tft_model import (
    GatedLinearUnit,
    GatedResidualNetwork,
    InterpretableMultiHeadAttention,
)
from src.models.tft_model import PositionalEncoding as TFTPositionalEncoding
from src.models.tft_model import VariableSelectionNetwork


class TFTEncoder(nn.Module):
    """
    Standalone TFT Encoder heavily optimized for multimodal fusion.
    Extracts purely the 128-dimensional pooled temporal features.
    """

    def __init__(self, config: Dict):
        super(TFTEncoder, self).__init__()

        self.config = {
            "input_size": 12,
            "hidden_size": 128,
            "num_heads": 4,
            "dropout": 0.1,
            "lstm_layers": 1,
        }
        self.config.update(config)

        input_size = self.config["input_size"]
        hidden_size = self.config["hidden_size"]
        num_heads = self.config["num_heads"]
        dropout = self.config["dropout"]
        lstm_layers = self.config["lstm_layers"]

        self.vsn = VariableSelectionNetwork(
            input_size=input_size,
            num_vars=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.lstm_glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.lstm_layer_norm = nn.LayerNorm(hidden_size)

        self.pos_encoder = TFTPositionalEncoding(d_model=hidden_size)

        self.attention = InterpretableMultiHeadAttention(
            d_model=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.attn_glu = GatedLinearUnit(hidden_size, hidden_size, dropout)
        self.attn_layer_norm = nn.LayerNorm(hidden_size)

        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
        )

    def forward(self, x):
        selected, var_weights = self.vsn(x)

        lstm_out, _ = self.lstm_encoder(selected)
        lstm_gated = self.lstm_glu(lstm_out)
        temporal_features = self.lstm_layer_norm(lstm_gated + selected)

        enriched = self.pos_encoder(temporal_features)
        attn_out, attn_weights = self.attention(enriched, enriched, enriched)

        attn_gated = self.attn_glu(attn_out)
        enriched_features = self.attn_layer_norm(attn_gated + temporal_features)

        enriched_features = self.output_grn(enriched_features)

        pooled = enriched_features[:, -1, :]
        return pooled


class MultiModalStockPredictor(nn.Module):
    def __init__(
        self,
        config: Dict,
        num_tickers: Optional[int] = None,
        num_sectors: Optional[int] = None,
    ):
        super(MultiModalStockPredictor, self).__init__()

        self.config = config
        self.max_articles = config.get("num_articles", 32)
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

        # Freeze FinBERT — use as fixed feature extractor
        for param in self.finbert.parameters():
            param.requires_grad = False
        self.finbert.eval()

        self.news_attention = nn.MultiheadAttention(
            embed_dim=self.finbert_hidden, num_heads=4, batch_first=True
        )

        self.ts_encoder = TFTEncoder(
            {
                "input_size": self.ts_features,
                "hidden_size": 128,
                "num_heads": 4,
                "dropout": 0.1,
                "lstm_layers": 1,
            }
        )
        ts_hidden_dim = 128

        self.ticker_embed = (
            nn.Embedding(num_tickers, 32) if num_tickers else nn.Linear(768, 32)
        )
        self.sector_embed = (
            nn.Embedding(num_sectors, 16) if num_sectors else nn.Linear(768, 16)
        )

        fusion_dim = self.finbert_hidden + ts_hidden_dim + 32 + 16

        self.fusion_norm = nn.LayerNorm(fusion_dim)

        self.fusion_hidden = 256
        self.fusion_bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, self.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.fc_out = nn.Linear(self.fusion_hidden, 5)

        self.target_idx = config.get("target_idx", 3)

    def forward(
        self,
        tokenized_news_: torch.Tensor,
        attention_mask_news_: torch.Tensor,
        time_series_features_: torch.Tensor,
        ticker_id_: torch.Tensor,
        sector_: torch.Tensor,
        article_mask_: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_articles, token_len = tokenized_news_.shape

        # --- A. NEWS ENCODING (FinBERT) ---
        news_flat = tokenized_news_.reshape(-1, token_len)
        mask_flat = attention_mask_news_.reshape(-1, token_len)

        news_out = self.finbert(input_ids=news_flat, attention_mask=mask_flat)
        news_features = news_out.last_hidden_state[
            :, 0, :
        ]  # (batch * num_articles, 768)
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

        ts_context_vec = self.ts_encoder(time_series_features_)

        last_known = time_series_features_[:, -1, self.target_idx].unsqueeze(1)

        ticker_features = self.ticker_embed(ticker_id_)
        sector_features = self.sector_embed(sector_)

        combined = torch.cat(
            [news_pooled, ts_context_vec, ticker_features, sector_features], dim=-1
        )

        fused_norm = self.fusion_norm(combined)
        fusion_summary = self.fusion_bottleneck(fused_norm)

        diff_pred = self.fc_out(fusion_summary)
        out = diff_pred + last_known

        return out
