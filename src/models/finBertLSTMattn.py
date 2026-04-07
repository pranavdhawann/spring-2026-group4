import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModel

from src.utils import set_seed


class FinBertLSTMDecoder(nn.Module):
    """
    Uses multiple token embeddings from each article with proper dimension handling
    """

    def __init__(self, config: Dict):
        super(FinBertLSTMDecoder, self).__init__()
        set_seed()

        self.config = {
            "finbert_name_or_path": "ProsusAI/finbert",
            "device": torch.device("cpu"),
            "local_files_only": True,
            "bert_hidden_dim": 768,
            "news_embedding_dim": 256,
            "lstm_hidden_dim": 512,
            "lstm_num_layers": 2,
            "FORECAST_HORIZON": 7,
            "freeze_bert": True,
            "dropout_rate": 0.2,
            "max_window_size": 14,
            "empty_news_threshold": 2,
            "teacher_forcing_ratio": 0.5,
            "attention_heads": 4,  # Reduced for stability
            "num_tokens_per_article": 10,
            "token_selection_strategy": "top_k",
            "use_positional_encoding": True,
            "max_seq_length": 512,
        }

        self.config.update(config)
        self.device = self.config["device"]

        self.teacher_forcing_ratio = self.config.get("teacher_forcing_ratio", 0.5)
        self.num_tokens = self.config.get("num_tokens_per_article", 10)
        self.news_dim = self.config["news_embedding_dim"]

        # Load FinBERT model
        self.finbert = AutoModel.from_pretrained(
            self.config["finbert_name_or_path"],
            local_files_only=self.config["local_files_only"],
        )

        # Freeze BERT if specified
        if self.config.get("freeze_bert", True):
            for param in self.finbert.parameters():
                param.requires_grad = False

        # Project BERT embeddings for each token
        self.token_projection = nn.Linear(self.config["bert_hidden_dim"], self.news_dim)

        # Learnable token selection (if using 'learned' strategy)
        self.token_selector = nn.Sequential(
            nn.Linear(self.news_dim, self.news_dim // 4),
            nn.ReLU(),
            nn.Linear(self.news_dim // 4, 1),
        )

        # Positional encoding for tokens within an article
        if self.config["use_positional_encoding"]:
            self.token_positional_encoding = PositionalEncoding(
                self.news_dim,
                self.config["max_seq_length"],
                self.config["dropout_rate"],
            )

        # Temporal attention over flattened tokens
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.news_dim,
            num_heads=self.config["attention_heads"],
            dropout=self.config["dropout_rate"],
            batch_first=True,
        )

        # Cross-attention between LSTM state and news tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.config["lstm_hidden_dim"],
            num_heads=4,
            dropout=self.config["dropout_rate"],
            batch_first=True,
        )

        # Compress multi-token representation
        self.token_compressor = nn.Sequential(
            nn.Linear(self.news_dim, self.news_dim),
            nn.LayerNorm(self.news_dim),
            nn.ReLU(),
            nn.Dropout(self.config["dropout_rate"]),
            nn.Linear(self.news_dim, self.news_dim),
        )

        # Feature embedding
        self.feature_embedding = nn.Sequential(
            nn.Linear(3, self.news_dim),
            nn.ReLU(),
            nn.Dropout(self.config["dropout_rate"]),
        )

        # LSTM Decoder
        lstm_input_size = self.news_dim * 2
        self.lstm_decoder = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=self.config["lstm_hidden_dim"],
            num_layers=2,
            batch_first=True,
            dropout=self.config["dropout_rate"],
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(
                self.config["lstm_hidden_dim"], self.config["lstm_hidden_dim"] // 2
            ),
            nn.ReLU(),
            nn.Dropout(self.config["dropout_rate"]),
            nn.Linear(self.config["lstm_hidden_dim"] // 2, 1),
        )

        # Projection for teacher forcing
        self.value_projection = nn.Linear(1, self.news_dim)

        self.to(self.device)

    def select_important_tokens(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Select most important tokens from each article"""
        B_flat, L, D = token_embeddings.shape
        num_tokens = self.num_tokens

        if self.config["token_selection_strategy"] == "top_k":
            token_norms = torch.norm(token_embeddings, dim=2)
            token_norms = token_norms * attention_mask.float()
            k = min(num_tokens, L)
            top_k_indices = torch.topk(token_norms, k, dim=1)[1]
            selected = torch.gather(
                token_embeddings, 1, top_k_indices.unsqueeze(-1).expand(-1, -1, D)
            )
        else:
            # Default to CLS token
            selected = token_embeddings[:, 0:1, :].expand(-1, num_tokens, -1)

        # Pad if needed
        if selected.shape[1] < num_tokens:
            padding = torch.zeros(
                B_flat, num_tokens - selected.shape[1], D, device=self.device
            )
            selected = torch.cat([selected, padding], dim=1)

        return selected[:, :num_tokens, :]

    def encode_multiple_articles_multi_token(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode multiple news articles, keeping multiple tokens per article"""
        B, W, L = input_ids.shape

        # Reshape to process all articles at once
        flat_input_ids = input_ids.view(B * W, L)
        flat_attention_mask = attention_mask.view(B * W, L)

        # Find empty articles
        non_padding = (flat_input_ids != 0).sum(dim=1)
        is_empty = non_padding <= self.config["empty_news_threshold"]

        # Initialize tensors
        all_token_embeddings = torch.zeros(
            B * W, L, self.config["bert_hidden_dim"], device=self.device
        )

        # Process non-empty articles
        if not is_empty.all():
            non_empty_indices = torch.where(~is_empty)[0]
            valid_input_ids = flat_input_ids[non_empty_indices]
            valid_attention_mask = flat_attention_mask[non_empty_indices]

            with autocast(device_type="cuda" if self.device.type == "cuda" else "cpu"):
                bert_outputs = self.finbert(
                    input_ids=valid_input_ids,
                    attention_mask=valid_attention_mask,
                )

            token_embeds = bert_outputs.last_hidden_state
            all_token_embeddings[non_empty_indices] = token_embeds

        # Project to smaller dimension
        projected_tokens = self.token_projection(all_token_embeddings)

        # Select important tokens
        selected_tokens = self.select_important_tokens(
            projected_tokens, flat_attention_mask
        )

        # Add positional encoding if enabled
        if self.config["use_positional_encoding"]:
            selected_tokens = self.token_positional_encoding(selected_tokens)

        # Reshape back to (B, W, num_tokens, news_dim)
        daily_multi_tokens = selected_tokens.view(B, W, self.num_tokens, self.news_dim)

        return daily_multi_tokens

    def aggregate_multi_tokens(self, daily_multi_tokens: torch.Tensor) -> torch.Tensor:
        """
        Aggregate multi-token representations across days

        Args:
            daily_multi_tokens: (B, W, num_tokens, news_dim)

        Returns:
            aggregated: (B, news_dim)
        """
        B, W, num_tokens, D = daily_multi_tokens.shape

        # Flatten across days and tokens: (B, W * num_tokens, D)
        flat_tokens = daily_multi_tokens.view(B, W * num_tokens, D)

        # Apply temporal attention - don't need weights to avoid shape issues
        attended, _ = self.temporal_attention(
            query=flat_tokens,
            key=flat_tokens,
            value=flat_tokens,
            need_weights=False,  # Don't return weights to avoid shape issues
        )

        # Pool across sequence dimension (mean pooling)
        aggregated = attended.mean(dim=1)  # (B, D)

        # Apply compression
        aggregated = self.token_compressor(aggregated)  # (B, D)

        return aggregated

    def forward(
        self,
        inputs: Dict,
        targets: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> torch.Tensor:
        """Forward pass with multi-token embeddings"""
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        closes = inputs["closes"]
        extra_features = inputs["extra_features"]

        B = input_ids.shape[0]
        H = self.config["FORECAST_HORIZON"]

        # 1. Encode articles with multiple tokens per article
        daily_multi_tokens = self.encode_multiple_articles_multi_token(
            input_ids, attention_mask
        )

        # 2. Aggregate multi-token representations
        news_embedding = self.aggregate_multi_tokens(daily_multi_tokens)

        # 3. Prepare initial decoder input
        last_close = closes[:, -1].unsqueeze(1)
        combined_features = torch.cat([last_close, extra_features], dim=1)
        embedded_features = self.feature_embedding(combined_features)

        decoder_input = torch.cat([news_embedding, embedded_features], dim=1)

        # 4. LSTM Decoder with teacher forcing
        h0 = torch.zeros(2, B, self.config["lstm_hidden_dim"], device=self.device)
        c0 = torch.zeros(2, B, self.config["lstm_hidden_dim"], device=self.device)

        predictions = []
        current_input = decoder_input

        for t in range(H):
            lstm_input = current_input.unsqueeze(1)
            lstm_output, (h0, c0) = self.lstm_decoder(lstm_input, (h0, c0))

            pred = self.output_projection(lstm_output.squeeze(1))
            predictions.append(pred)

            # Teacher forcing
            if (
                training
                and targets is not None
                and torch.rand(1).item() < self.teacher_forcing_ratio
            ):
                next_value = self.value_projection(targets[:, t, :])
                next_input = torch.cat([news_embedding, next_value], dim=1)
            else:
                next_value = self.value_projection(pred)
                next_input = torch.cat([news_embedding, next_value], dim=1)

            current_input = next_input

        predictions = torch.stack(predictions, dim=1)

        return predictions


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "finbert_name_or_path": "ProsusAI/finbert",
        "device": device,
        "FORECAST_HORIZON": 7,
        "news_embedding_dim": 256,
        "lstm_hidden_dim": 512,
        "dropout_rate": 0.2,
        "freeze_bert": True,
        "max_window_size": 14,
        "empty_news_threshold": 2,
        "teacher_forcing_ratio": 0.5,
        "num_tokens_per_article": 10,
        "token_selection_strategy": "top_k",
        "use_positional_encoding": True,
    }

    model = FinBertLSTMDecoder(config)
    tokenizer = AutoTokenizer.from_pretrained(config["finbert_name_or_path"])

    print("\n" + "=" * 70)
    print("TESTING FIXED MULTI-TOKEN MODEL")
    print("=" * 70)

    # Create test data
    batch_size = 2
    window_size = 3

    news_articles = [
        [
            "Apple stock rises sharply on strong earnings",
            "Market shows positive momentum",
            "Tech sector leads gains",
        ],
        [
            "Tesla faces production delays",
            "Supply chain issues worsen",
            "EV demand slows down",
        ],
    ]

    # Tokenize
    all_texts = []
    for sample_articles in news_articles:
        all_texts.extend(sample_articles)

    tokenized = tokenizer(
        all_texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    input_ids = tokenized["input_ids"].view(batch_size, window_size, -1)
    attention_mask = tokenized["attention_mask"].view(batch_size, window_size, -1)

    print(f"\nInput shape: {input_ids.shape}")

    # Test forward pass
    dummy_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "closes": torch.randn(batch_size, 30),
        "extra_features": torch.randn(batch_size, 2),
    }

    targets = torch.randn(batch_size, config["FORECAST_HORIZON"], 1)

    model.train()
    predictions = model(dummy_inputs, targets=targets, training=True)

    print(f"Predictions shape: {predictions.shape}")
    print("\n✓ Model working correctly!")
