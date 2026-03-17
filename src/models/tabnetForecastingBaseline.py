


from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModel

from src.utils import set_seed


class tabet_forcasting(nn.Module):


    def __init__(self, config: Dict):
        super().__init__()
        set_seed()

        self.config = {
            # task: "regression" (7-day forecasting) or "classification" (next-day direction)
            "task": "regression",
            "num_classes": 2,  # only used when task == "classification"
            "finbert_name_or_path": "ProsusAI/finbert",
            "device": torch.device("cpu"),
            "local_files_only": True,
            "bert_hidden_dim": 768,
            "news_embedding_dim": 256,
            "FORECAST_HORIZON": 7,
            "freeze_bert": True,
            "max_window_size": 10,
            "empty_news_threshold": 2,
            "tabnet_n_d": 64,
            "tabnet_n_a": 64,
            "tabnet_n_steps": 3,
            "tabnet_gamma": 1.5,
            "tabnet_n_independent": 2,
            "tabnet_n_shared": 2,
        }

        self.config.update(config)
        self.device = self.config["device"]

        self.max_window_size = self.config.get("max_window_size", 10)
        self.empty_news_threshold = self.config.get("empty_news_threshold", 2)

        # Text encoder: FinBERT (same as baseline)
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

        # TabNet head (continuous-only: use TabNetNoEmbeddings to avoid EmbeddingGenerator).
        input_dim = self.config["news_embedding_dim"] + 3

        from pytorch_tabnet.tab_network import TabNetNoEmbeddings

        task: Literal["regression", "classification"] = self.config.get(
            "task", "regression"
        )
        if task == "classification":
            # Binary by default: output a single logit. Use BCEWithLogitsLoss in training.
            output_dim = 1
        else:
            output_dim = self.config["FORECAST_HORIZON"]

        self.tabnet = TabNetNoEmbeddings(
            input_dim=input_dim,
            output_dim=output_dim,
            n_d=self.config["tabnet_n_d"],
            n_a=self.config["tabnet_n_a"],
            n_steps=self.config["tabnet_n_steps"],
            gamma=self.config["tabnet_gamma"],
            n_independent=self.config["tabnet_n_independent"],
            n_shared=self.config["tabnet_n_shared"],
        )

        self.to(self.device)

    def is_empty_news(self, input_ids: torch.Tensor) -> torch.Tensor:
        non_padding = (input_ids != 0).sum(dim=-1)
        return non_padding <= self.empty_news_threshold

    def find_latest_news_indices(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, W, _ = input_ids.shape
        is_empty = self.is_empty_news(input_ids)
        positions = torch.arange(W, device=input_ids.device).expand(B, W)
        valid_positions = torch.where(~is_empty, positions, -1)
        latest_indices = valid_positions.max(dim=1)[0]
        return latest_indices

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        closes = inputs["closes"]
        extra_features = inputs["extra_features"]

        B, W, _ = input_ids.shape

        last_close = closes[:, -1].unsqueeze(1)
        combined_features = torch.cat([last_close, extra_features], dim=1)

        latest_indices = self.find_latest_news_indices(input_ids)
        has_news_mask = latest_indices >= 0

        news_embeddings = torch.zeros(
            B, self.config["bert_hidden_dim"], device=self.device
        )

        if has_news_mask.any():
            news_batch_indices = torch.where(has_news_mask)[0]
            news_window_indices = latest_indices[news_batch_indices]
            latest_input_ids = input_ids[news_batch_indices, news_window_indices]
            latest_attention_mask = attention_mask[news_batch_indices, news_window_indices]

            with autocast(device_type="cuda"):
                bert_outputs = self.finbert(
                    input_ids=latest_input_ids,
                    attention_mask=latest_attention_mask,
                )

            cls_embeddings = bert_outputs.last_hidden_state[:, 0, :]
            news_embeddings[news_batch_indices] = cls_embeddings

        projected_news = self.news_projection(news_embeddings)
        tabnet_input = torch.cat([projected_news, combined_features], dim=1)

        out, _ = self.tabnet(tabnet_input)
        return out

    @torch.no_grad()
    def predict_proba(self, inputs) -> torch.Tensor:
        """
        For classification only: returns probabilities in [0,1] with shape (B,).
        """
        task = self.config.get("task", "regression")
        if task != "classification":
            raise ValueError("predict_proba is only valid when task='classification'")
        logits = self.forward(inputs).view(-1)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_label(self, inputs, threshold: float = 0.5) -> torch.Tensor:
        """
        For classification only: returns hard labels {0,1} with shape (B,).
        """
        probs = self.predict_proba(inputs)
        return (probs >= threshold).long()


__all__ = ["tabet_forcasting"]

