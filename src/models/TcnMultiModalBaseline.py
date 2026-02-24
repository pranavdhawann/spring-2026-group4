"""TCN Multimodal Model"""

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
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.residual is None else self.residual(x)

        return self.relu(out + res)


class TCNEncoder(nn.Module):

    def __init__(self, config: Dict):
        super(TCNEncoder, self).__init__()

        self.config = {
            'input_size': 12,
            'num_channels': [64, 128, 256],
            'kernel_size': 3,
            'dropout': 0.2,
            'embedding_size': 256,
        }
        self.config.update(config)

        input_size = self.config['input_size']
        num_channels = self.config['num_channels']
        kernel_size = self.config['kernel_size']
        dropout = self.config['dropout']
        embedding_size = self.config['embedding_size']

        temporal_blocks = []
        in_channels = input_size

        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
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

        if num_channels[-1] != embedding_size:
            self.projection = nn.Linear(num_channels[-1], embedding_size)
        else:
            self.projection = nn.Identity()

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.tcn(x)

        x = x.mean(dim=2)


        embedding = self.projection(x)

        return embedding


if __name__ == "__main__":

    print("TCN ENCODER TEST")


    print("\n[Test 1] Default config (match LSTM output):")
    config = {
        'input_size': 12,
        'num_channels': [64, 128, 256],
        'kernel_size': 3,
        'dropout': 0.2,
        'embedding_size': 256,
    }

    model = TCNEncoder(config)

    batch_size = 32
    seq_len = 14
    input_size = 12

    x = torch.randn(batch_size, seq_len, input_size)

    embedding = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {embedding.shape}")
    print(f"  Expected:     (32, 256)")

    assert embedding.shape == (batch_size, 256), f"Expected (32, 256), got {embedding.shape}"
    print("Passed!")

    print("\n[Test 2] Custom config:")
    config = {
        'input_size': 12,
        'num_channels': [32, 64, 128],
        'kernel_size': 5,
        'dropout': 0.3,
        'embedding_size': 128,
    }

    model = TCNEncoder(config)
    x = torch.randn(16, 14, 12)
    embedding = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {embedding.shape}")
    print(f"  Expected:     (16, 128)")

    assert embedding.shape == (16, 128), f"Expected (16, 128), got {embedding.shape}"
    print("Passed!")

    # Test 3: Missing config keys use defaults
    print("\n[Test 3] Missing keys fall back to defaults:")
    config = {}  # Empty config

    model = TCNEncoder(config)
    x = torch.randn(8, 14, 12)
    embedding = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {embedding.shape}")
    print(f"  Expected:     (8, 256)")

    assert embedding.shape == (8, 256), f"Expected (8, 256), got {embedding.shape}"
    print("Passed!")

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    print("ALL TESTS PASSED!")

    print(f"\nModel Summary:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Input:  (batch_size, seq_len=14, input_size=12)")
    print(f"  Output: (batch_size, embedding_size=256)")

