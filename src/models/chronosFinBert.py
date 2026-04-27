from typing import Dict

import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModel

from src.utils import set_seed


class ChronosFinBert(nn.Module):
    def __init__(self, config: Dict):
        super(ChronosFinBert, self).__init__()
        set_seed()

        self.config = {
            "chronos_model_path": "amazon/chronos-t5-base",
            "local_files_only": False,
            "bert_hidden_dim": 768,
            "d_fusion": 512,
            "lstm_hidden": 256,
            "lstm_num_layers": 2,
            "dropout_rate": 0.2,
            "FORECAST_HORIZON": 5,
            "freeze_bert": True,
            "freeze_chronos": True,
            "empty_news_threshold": 2,
        }

        self.config.update(config)
        self.device = self.config.get("device", torch.device("cpu"))

        self.empty_news_threshold = self.config["empty_news_threshold"]

        # ── Chronos encoder (T5 encoder only) ─────────────────────────────
        chronos_full = AutoModel.from_pretrained(
            self.config["chronos_model_path"],
            local_files_only=self.config["local_files_only"],
        )
        self.chronos_encoder = chronos_full.encoder

        if self.config["freeze_chronos"]:
            for param in self.chronos_encoder.parameters():
                param.requires_grad = False
            for block in self.chronos_encoder.block[-4:]:
                for param in block.parameters():
                    param.requires_grad = True

        chronos_d = self.chronos_encoder.config.d_model

        # Projects scalar close prices into the encoder's embedding space
        self.ts_input_proj = nn.Linear(1, chronos_d)

        # ── FinBERT ────────────────────────────────────────────────────────
        self.finbert = AutoModel.from_pretrained(
            "ProsusAI/finbert",
            local_files_only=self.config["local_files_only"],
        )

        if self.config["freeze_bert"]:
            for param in self.finbert.parameters():
                param.requires_grad = False
            for layer in self.finbert.encoder.layer[-4:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # ── Projections ───────────────────────────────────────────────────
        d_fusion = self.config["d_fusion"]

        self.ts_projection = nn.Linear(chronos_d, d_fusion)
        self.news_projection = nn.Linear(self.config["bert_hidden_dim"], d_fusion)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_fusion,
            num_heads=8,
            dropout=self.config["dropout_rate"],
            batch_first=True,
        )

        # ── Fusion LSTM ───────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=d_fusion,
            hidden_size=self.config["lstm_hidden"],
            num_layers=self.config["lstm_num_layers"],
            batch_first=True,
            dropout=self.config["dropout_rate"]
            if self.config["lstm_num_layers"] > 1
            else 0,
        )

        # ── Head ──────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(self.config["dropout_rate"])
        self.head = nn.Linear(
            self.config["lstm_hidden"], self.config["FORECAST_HORIZON"]
        )

        nn.init.xavier_uniform_(self.ts_projection.weight)
        nn.init.xavier_uniform_(self.news_projection.weight)
        nn.init.xavier_uniform_(self.ts_input_proj.weight)
        nn.init.xavier_uniform_(self.head.weight)

        self.to(self.device)

    # ── News helpers (copied exactly from FinBertForecastingBL) ───────────

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

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]  # (B, W, L)
        attention_mask = inputs["attention_mask"]  # (B, W, L)
        closes = inputs["closes"]  # (B, N)
        extra_features = inputs[
            "extra_features"
        ]  # (B, 2)  # noqa: F841 — kept for API compatibility

        B, W, L = input_ids.shape
        N = closes.shape[1]

        # ── Chronos: encode close price sequence ──────────────────────────
        # Project scalar floats → encoder embedding space, then pass as
        # inputs_embeds so no vocabulary quantisation is required.
        closes_embeds = self.ts_input_proj(closes.unsqueeze(-1))  # (B, N, chronos_d)
        chronos_out = self.chronos_encoder(inputs_embeds=closes_embeds)
        ts_hidden = chronos_out.last_hidden_state  # (B, N, chronos_d)

        ts_proj = self.ts_projection(ts_hidden)  # (B, N, d_fusion)

        # ── FinBERT: encode ALL non-empty news windows → sequence of CLS embeddings ─
        # is_empty: (B, W) — True where window has no real tokens
        is_empty = self.is_empty_news(input_ids)  # (B, W)
        # news_seq: (B, W, bert_hidden_dim) — CLS embedding per window, zero for empty
        news_seq = torch.zeros(B, W, self.config["bert_hidden_dim"], device=self.device)

        # Flatten all non-empty (batch, window) pairs and run FinBERT in one pass
        non_empty_b, non_empty_w = torch.where(~is_empty)  # each shape (M,)
        if non_empty_b.numel() > 0:
            flat_input_ids = input_ids[non_empty_b, non_empty_w]  # (M, L)
            flat_attention_mask = attention_mask[non_empty_b, non_empty_w]  # (M, L)

            with autocast(device_type="cuda"):
                bert_outputs = self.finbert(
                    input_ids=flat_input_ids,
                    attention_mask=flat_attention_mask,
                )

            cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]  # (M, 768)
            news_seq[non_empty_b, non_empty_w] = cls_embeddings  # scatter back

        news_proj = self.news_projection(news_seq)  # (B, W, d_fusion)

        # ── Fusion: cross attention — TS sequence attends to full news sequence ─
        # ts_proj: (B, N, d_fusion)  news_proj: (B, W, d_fusion)
        fused, _ = self.cross_attention(ts_proj, news_proj, news_proj)

        # ── LSTM over fused sequence ───────────────────────────────────────
        _, (h_n, _) = self.lstm(fused)
        last_hidden = h_n[-1]

        # ── Prediction head ───────────────────────────────────────────────
        out = self.dropout(last_hidden)
        predictions = self.head(out)  # (B, FORECAST_HORIZON)

        return predictions
