from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.utils import set_seed


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for encoder outputs"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert (
            self.head_dim * num_heads == hidden_dim
        ), "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        mask: (batch_size, seq_len) optional padding mask
        """
        batch_size, seq_len, _ = x.shape

        # Linear projections
        Q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if mask is not None:
            # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_dim)
        )

        return self.out_proj(context), attn_weights


class CrossAttention(nn.Module):
    """Cross-attention mechanism for decoder to attend to encoder outputs"""

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(CrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert (
            self.head_dim * num_heads == hidden_dim
        ), "hidden_dim must be divisible by num_heads"

        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, decoder_hidden, encoder_outputs, encoder_mask=None):
        """
        decoder_hidden: (batch_size, 1, hidden_dim) or (batch_size, seq_len, hidden_dim)
        encoder_outputs: (batch_size, seq_len, hidden_dim)
        encoder_mask: (batch_size, seq_len) optional padding mask
        """
        batch_size, decoder_seq_len, _ = decoder_hidden.shape
        encoder_seq_len = encoder_outputs.shape[1]

        # Linear projections
        Q = (
            self.query(decoder_hidden)
            .view(batch_size, decoder_seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key(encoder_outputs)
            .view(batch_size, encoder_seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(encoder_outputs)
            .view(batch_size, encoder_seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if encoder_mask is not None:
            # encoder_mask: (batch_size, encoder_seq_len) -> (batch_size, 1, decoder_seq_len, encoder_seq_len)
            encoder_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(encoder_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, decoder_seq_len, self.hidden_dim)
        )

        return self.out_proj(context), attn_weights


class FinBertForecastingBL(nn.Module):
    """
    FinBERT + Encoder LSTM (time) + Feature-conditioned Decoder LSTM with Attention
    """

    def __init__(self, config: Dict):
        super(FinBertForecastingBL, self).__init__()
        set_seed()

        self.config = {
            "finbert_name_or_path": "ProsusAI/finbert",
            "device": torch.device("cpu"),
            "local_files_only": True,
            "bert_hidden_dim": 768,
            "lstm_hidden_dim": 256,
            "lstm_num_layers": 1,
            "FORECAST_HORIZON": 7,
            "attention_num_heads": 8,
            "attention_dropout": 0.1,
            "use_self_attention": True,
            "use_cross_attention": True,
        }

        self.config.update(config)
        self.device = self.config["device"]

        self.finbert = AutoModel.from_pretrained(
            self.config["finbert_name_or_path"],
            local_files_only=self.config["local_files_only"],
        )

        self.encoder_lstm = nn.LSTM(
            input_size=self.config["bert_hidden_dim"],
            hidden_size=self.config["lstm_hidden_dim"],
            num_layers=self.config["lstm_num_layers"],
            batch_first=True,
        )

        # Self-attention for encoder outputs
        self.use_self_attention = self.config["use_self_attention"]
        if self.use_self_attention:
            self.self_attention = MultiHeadSelfAttention(
                self.config["lstm_hidden_dim"],
                num_heads=self.config["attention_num_heads"],
                dropout=self.config["attention_dropout"],
            )
            self.self_attention_norm = nn.LayerNorm(self.config["lstm_hidden_dim"])
            self.self_attention_dropout = nn.Dropout(self.config["attention_dropout"])

        # (hidden + 3 floats) → hidden_dim
        self.feature_projection = nn.Linear(
            self.config["lstm_hidden_dim"] + 3, self.config["lstm_hidden_dim"]
        )

        self.decoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.config["lstm_hidden_dim"],
            num_layers=self.config["lstm_num_layers"],
            batch_first=True,
        )

        # Cross-attention for decoder
        self.use_cross_attention = self.config["use_cross_attention"]
        if self.use_cross_attention:
            self.cross_attention = CrossAttention(
                self.config["lstm_hidden_dim"],
                num_heads=self.config["attention_num_heads"],
                dropout=self.config["attention_dropout"],
            )
            self.cross_attention_norm = nn.LayerNorm(self.config["lstm_hidden_dim"])
            self.cross_attention_dropout = nn.Dropout(self.config["attention_dropout"])

        self.regressor = nn.Linear(self.config["lstm_hidden_dim"], 1)

        self.to(self.device)

    def forward(self, inputs):
        """
        inputs:
        {
            "input_ids": (B, W, L)
            "attention_mask": (B, W, L)
            "extra_features": (B, 3)
        }
        """

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        extra_features = inputs["extra_features"].to(self.device)  # (B, 2)

        closes = inputs["closes"][:, -1].to(self.device)  # (B,)
        closes = closes.unsqueeze(1)  # (B, 1)
        extra_features = torch.cat([closes, extra_features], dim=1)

        B, W, L = input_ids.shape

        input_ids = input_ids.view(B * W, L)
        attention_mask = attention_mask.view(B * W, L)

        outputs = self.finbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings.view(B, W, -1)  # (B, W, 768)

        # Pass through encoder LSTM
        encoder_outputs, (h_n, c_n) = self.encoder_lstm(cls_embeddings)
        # encoder_outputs: (B, W, hidden_dim)

        # Apply self-attention to encoder outputs
        if self.use_self_attention:
            # Create mask based on actual content (assuming no padding in sequence dimension)
            # For a more sophisticated mask, you might want to track which time steps are valid
            attn_output, attn_weights = self.self_attention(encoder_outputs)
            encoder_outputs = self.self_attention_norm(
                encoder_outputs + self.self_attention_dropout(attn_output)
            )

        encoder_hidden = h_n[-1]  # (B, hidden_dim)

        conditioned_hidden = torch.cat(
            [encoder_hidden, extra_features], dim=1
        )  # (B, hidden_dim + 3)

        projected_hidden = torch.tanh(
            self.feature_projection(conditioned_hidden)
        )  # (B, hidden_dim)

        decoder_hidden = (
            projected_hidden.unsqueeze(0),  # h_0
            c_n,  # reuse encoder cell
        )

        forecast_steps = self.config["FORECAST_HORIZON"]

        decoder_input = torch.zeros(B, 1, 1, device=self.device)
        outputs_list = []

        # Store the last encoder outputs for cross-attention
        encoder_context = encoder_outputs

        for t in range(forecast_steps):
            out, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)

            # Apply cross-attention between decoder output and encoder outputs
            if self.use_cross_attention:
                # out shape: (B, 1, hidden_dim)
                attn_out, attn_weights = self.cross_attention(out, encoder_context)
                out = self.cross_attention_norm(
                    out + self.cross_attention_dropout(attn_out)
                )

            pred = self.regressor(out)  # (B,1,1)
            outputs_list.append(pred)
            decoder_input = pred  # autoregressive

        outputs = torch.cat(outputs_list, dim=1)  # (B, T, 1)
        return outputs.squeeze(-1)  # (B, T)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = {
        "finbert_name_or_path": "ProsusAI/finbert",
        "device": device,
        "FORECAST_HORIZON": 7,
        "lstm_hidden_dim": 256,
        "use_self_attention": True,
        "use_cross_attention": True,
        "attention_num_heads": 8,
    }
    model = FinBertForecastingBL(config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["finbert_name_or_path"])

    # Create dummy inputs for testing
    texts = [
        "Apple stock rose 3% after strong earnings report.",
        "Market volatility increases amid inflation concerns.",
        "Tesla shares drop following delivery miss.",
        "Federal Reserve signals potential rate cuts.",
        "Oil prices surge due to supply constraints.",
    ]

    # Simulate the input structure expected by the model
    inputs = {
        "input_ids": tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )["input_ids"].unsqueeze(1),
        "attention_mask": tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        )["attention_mask"].unsqueeze(1),
        "extra_features": torch.randn(len(texts), 2),
        "closes": torch.randn(len(texts), 10),  # Simulate historical closes
    }

    with torch.no_grad():
        outputs = model(inputs)

    print("Input batch size:", len(texts))
    print("Forecast output shape:", outputs.shape)
    print("Forecast output:")
    print(outputs)
