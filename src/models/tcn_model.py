"""TCN Baseline Model"""

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
        out = out[:, :, : x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = out[:, :, : x.size(2)]
        out = self.relu(out)
        out = self.dropout(out)

        res = x if self.residual is None else self.residual(x)

        return self.relu(out + res)


class TCNModel(nn.Module):

    def __init__(self, config: Dict):
        super(TCNModel, self).__init__()

        self.config = {
            'input_size': 12,
            'output_size': 7,
            'num_channels': [64, 64, 128, 128],
            'kernel_size': 3,
            'dropout': 0.1,
            'hidden_size': 64,
        }
        self.config.update(config)

        input_size = self.config['input_size']
        output_size = self.config['output_size']
        num_channels = self.config['num_channels']
        kernel_size = self.config['kernel_size']
        dropout = self.config['dropout']
        hidden_size = self.config['hidden_size']

        temporal_blocks = []
        in_channels = input_size

        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i  # 1, 2, 4, 8 ...
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

        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 1)

        x = self.tcn(x)

        x = x.mean(dim=2)

        out = self.fc(x)

        return out


if __name__ == "__main__":

    print("TCN MODEL TEST")


    # Test 1: Default config
    print("\n[Test 1] Default config:")
    config = {
        'input_size': 12,
        'output_size': 7,
    }

    model = TCNModel(config)
    x = torch.randn(8, 14, 12)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (8, 7), f"Expected (8, 7), got {y.shape}"
    print("Passed!")

    # Test 2: Custom config
    print("\n[Test 2] Custom config:")
    config = {
        'input_size': 16,
        'output_size': 3,
        'num_channels': [32, 64, 128],
        'kernel_size': 5,
        'dropout': 0.2,
        'hidden_size': 32,
    }

    model = TCNModel(config)
    x = torch.randn(8, 20, 16)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (8, 3), f"Expected (8, 3), got {y.shape}"
    print("Passed!")

    # Test 3: Missing config keys use defaults
    print("\n[Test 3] Missing keys fall back to defaults:")
    config = {}

    model = TCNModel(config)
    x = torch.randn(4, 14, 12)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (4, 7), f"Expected (4, 7), got {y.shape}"
    print("Passed!")

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("ALL TESTS PASSED!")
    print(f"\nModel Summary:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Input:  (batch_size, seq_len=14, input_size=12)")
    print(f"  Output: (batch_size, output_size=7)")
   