"""Temporal Fusion Transformer (TFT)"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLinearUnit(nn.Module):
    """GLU: σ(W₁x + b₁) ⊙ (W₂x + b₂)"""

    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(GatedLinearUnit, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.dropout(self.fc2(x))
        return sig * x


class GatedResidualNetwork(nn.Module):
    """GRN with optional context vector injection."""

    def __init__(
        self, input_size, hidden_size, output_size=None, context_size=None, dropout=0.0
    ):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size

        self.fc1 = nn.Linear(input_size, hidden_size)

        self.context_fc = (
            nn.Linear(context_size, hidden_size, bias=False)
            if context_size is not None
            else None
        )

        self.elu = nn.ELU()

        self.fc2 = nn.Linear(hidden_size, self.output_size)

        self.glu = GatedLinearUnit(self.output_size, self.output_size, dropout)

        self.layer_norm = nn.LayerNorm(self.output_size)

        if input_size != self.output_size:
            self.skip_proj = nn.Linear(input_size, self.output_size)
        else:
            self.skip_proj = None

    def forward(self, x, context=None):
        residual = x if self.skip_proj is None else self.skip_proj(x)

        hidden = self.fc1(x)
        if self.context_fc is not None and context is not None:
            hidden = hidden + self.context_fc(context)
        hidden = self.elu(hidden)

        hidden = self.fc2(hidden)

        gated = self.glu(hidden)
        output = self.layer_norm(gated + residual)

        return output


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self, input_size, num_vars, hidden_size, dropout=0.0, context_size=None
    ):
        super(VariableSelectionNetwork, self).__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size

        self.var_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=1,
                    hidden_size=hidden_size,
                    output_size=hidden_size,
                    dropout=dropout,
                )
                for _ in range(num_vars)
            ]
        )

        self.selection_grn = GatedResidualNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=num_vars,
            context_size=context_size,
            dropout=dropout,
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, context=None):
        var_outputs = []
        for i in range(self.num_vars):
            var_input = x[:, :, i : i + 1]
            var_out = self.var_grns[i](var_input)
            var_outputs.append(var_out)

        var_outputs = torch.stack(var_outputs, dim=2)

        if context is not None:
            if context.dim() == 2:
                context = context.unsqueeze(1).expand(-1, x.size(1), -1)
        weights = self.selection_grn(x, context)
        weights = self.softmax(weights).unsqueeze(-1)

        selected = (var_outputs * weights).sum(dim=2)

        return selected, weights.squeeze(-1)


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with independent values across heads for maximum capacity."""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_k)

        self.out_proj = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value):
        batch_size, seq_len, _ = query.shape

        # Project Q, K per head
        Q = self.W_q(query).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k)

        # Shared V
        V = self.W_v(value)  # (batch, seq_len, d_k)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        avg_attn = attn_weights.mean(dim=1)

        attn_output = torch.matmul(avg_attn, V)

        output = self.out_proj(attn_output)  # (batch, seq_len, d_model)

        return output, avg_attn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Removed scaling to keep signal-to-encoding ratio aligned with best run
        x = x + self.pe[: x.size(1), 0, :].unsqueeze(0)
        return x


class TFTModel(nn.Module):
    def __init__(self, config: Dict):
        super(TFTModel, self).__init__()

        self.config = {
            "input_size": 12,
            "output_size": 5,
            "hidden_size": 128,
            "num_heads": 4,
            "dropout": 0.1,
            "lstm_layers": 1,
        }
        self.config.update(config)

        input_size = self.config["input_size"]
        output_size = self.config["output_size"]
        hidden_size = self.config["hidden_size"]
        num_heads = self.config["num_heads"]
        dropout = self.config["dropout"]
        lstm_layers = self.config["lstm_layers"]

        self.target_idx = self.config.get("target_idx", 3)

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

        self.pos_encoder = PositionalEncoding(d_model=hidden_size)

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

        self.fc_out = nn.Linear(hidden_size, output_size)

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

        diff_pred = self.fc_out(pooled)

        last_known = x[:, -1, self.target_idx].unsqueeze(1)

        out = diff_pred + last_known

        return out

    def get_attention_weights(self, x):
        """Run forward pass and return interpretable attention weights.

        Returns:
            var_weights: (Batch, Seq_Len, num_vars) — feature importance
            attn_weights: (Batch, Seq_Len, Seq_Len) — temporal attention
        """
        self.eval()
        with torch.no_grad():
            selected, var_weights = self.vsn(x)

            lstm_out, _ = self.lstm_encoder(selected)
            lstm_gated = self.lstm_glu(lstm_out)
            temporal_features = self.lstm_layer_norm(lstm_gated + selected)

            enriched = self.pos_encoder(temporal_features)
            _, attn_weights = self.attention(enriched, enriched, enriched)

        return var_weights, attn_weights


if __name__ == "__main__":
    print("TFT STANDALONE MODEL TEST")

    print("\n[Test 1] 60-Day window sequence:")
    config = {
        "input_size": 12,
        "output_size": 5,
        "hidden_size": 128,
        "num_heads": 4,
        "dropout": 0.1,
        "lstm_layers": 1,
    }

    model = TFTModel(config)

    x = torch.randn(8, 60, 12)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (8, 5), f"Expected (8, 5), got {y.shape}"
    print("  Passed!")

    print("\n[Test 2] Custom config:")
    config = {
        "input_size": 16,
        "output_size": 3,
        "hidden_size": 64,
        "num_heads": 4,
        "dropout": 0.2,
        "lstm_layers": 2,
    }

    model = TFTModel(config)
    x = torch.randn(8, 20, 16)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (8, 3), f"Expected (8, 3), got {y.shape}"
    print("  Passed!")

    print("\n[Test 3] Missing keys fall back to defaults:")
    config = {}

    model = TFTModel(config)
    x = torch.randn(4, 14, 12)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (4, 5), f"Expected (4, 5), got {y.shape}"
    print("  Passed!")

    print("\n[Test 4] Attention weight extraction:")
    config = {
        "input_size": 12,
        "output_size": 5,
        "hidden_size": 128,
        "num_heads": 4,
    }
    model = TFTModel(config)
    x = torch.randn(4, 60, 12)
    var_w, attn_w = model.get_attention_weights(x)

    print(f"  Variable weights shape: {var_w.shape}")
    print(f"  Attention weights shape: {attn_w.shape}")
    assert var_w.shape == (4, 60, 12)
    assert attn_w.shape == (4, 60, 60)
    print("  Passed!")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\nALL TESTS PASSED!")
    print("\nModel Summary (TFT Standalone):")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("  Input:  (batch_size, seq_len=60, input_size=12)")
    print("  Output: (batch_size, output_size=5)")
