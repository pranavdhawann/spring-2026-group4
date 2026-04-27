"""TCN Baseline Model"""

import math
from typing import Dict

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.utils.weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
        )

        self.conv2 = nn.utils.weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=padding,
            )
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(out_channels)
        self.ln2 = nn.LayerNorm(out_channels)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, : x.size(2)]
        out = out.permute(0, 2, 1)
        out = self.ln1(out)
        out = out.permute(0, 2, 1)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(2)]
        out = out.permute(0, 2, 1)
        out = self.ln2(out)
        out = out.permute(0, 2, 1)
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.residual is None else self.residual(x)

        return self.relu(out + res)


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
        x = x + self.pe[: x.size(1), 0, :].unsqueeze(0)
        return x


class TCNModel(nn.Module):
    def __init__(self, config: Dict):
        super(TCNModel, self).__init__()

        self.config = {
            "input_size": 12,
            "output_size": 5,
            "num_channels": [64, 128, 128, 256],
            "kernel_size": 3,
            "dropout": 0.1,
            "hidden_size": 256,
            "nhead": 8,
            "num_layers": 2,
        }
        self.config.update(config)

        input_size = self.config["input_size"]
        output_size = self.config["output_size"]
        num_channels = self.config["num_channels"]
        kernel_size = self.config["kernel_size"]
        dropout = self.config["dropout"]
        hidden_size = self.config["hidden_size"]
        nhead = self.config["nhead"]
        num_layers = self.config["num_layers"]

        temporal_blocks = []
        in_channels = input_size

        for i, out_channels in enumerate(num_channels):
            dilation = 2**i
            temporal_blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*temporal_blocks)

        self.feature_extractor = nn.Sequential(
            nn.Linear(num_channels[-1], hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        self.pos_encoder = PositionalEncoding(d_model=hidden_size)

        self.fc_out = nn.Linear(hidden_size, output_size)

        self.target_idx = self.config.get("target_idx", 3)

    def forward(self, x):
        x_tcn = x.permute(0, 2, 1)

        x_tcn = self.tcn(x_tcn)

        x_tcn = x_tcn.permute(0, 2, 1)
        x_features = self.feature_extractor(x_tcn)

        x_features = self.pos_encoder(x_features)

        encoder_outputs = self.transformer_encoder(x_features)

        pooled_out = encoder_outputs[:, -1, :]

        diff_pred = self.fc_out(pooled_out)

        last_known = x[:, -1, self.target_idx].unsqueeze(1)

        out = diff_pred + last_known

        return out


if __name__ == "__main__":
    print("TCN STANDALONE MODEL TEST")

    print("\n[Test 1] 60-Day window sequence:")
    config = {
        "input_size": 12,
        "output_size": 5,
        "hidden_size": 64,
        "dropout": 0.1,
    }

    model = TCNModel(config)

    # Batch size 8, Sequence length 60, Features 12
    x = torch.randn(8, 60, 12)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (8, 5), f"Expected (8, 5), got {y.shape}"
    print("Passed!")

    print("\n[Test 2] Custom config:")
    config = {
        "input_size": 16,
        "output_size": 3,
        "num_channels": [32, 64, 128],
        "kernel_size": 5,
        "dropout": 0.2,
        "hidden_size": 32,
    }

    model = TCNModel(config)
    x = torch.randn(8, 20, 16)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (8, 3), f"Expected (8, 3), got {y.shape}"
    print("Passed!")

    print("\n[Test 3] Missing keys fall back to defaults:")
    config = {}

    model = TCNModel(config)
    x = torch.randn(4, 14, 12)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (4, 5), f"Expected (4, 5), got {y.shape}"
    print("Passed!")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("ALL TESTS PASSED!")
    print("\nModel Summary (Transformer Baseline):")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("  Input:  (batch_size, seq_len=60, input_size=12)")
    print("  Output: (batch_size, output_size=5)")
