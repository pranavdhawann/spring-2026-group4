"""TSMixer Model for Time-Series Stock Price Forecasting

TSMixer uses alternating time-mixing and feature-mixing MLPs to capture
both temporal patterns and cross-feature interactions. Designed for
short-sequence multivariate forecasting (14 timesteps, 23 features).

Reference: Google Research, "TSMixer: An All-MLP Architecture for
Time Series Forecasting" (2023)
"""

from typing import Dict
import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """Single TSMixer block with time-mixing and feature-mixing MLPs.

    Time-mixing operates across the temporal dimension (seq_len).
    Feature-mixing operates across the feature dimension (d_model).
    Both have residual connections and LayerNorm.

    V2: Separate d_ff_time for time-mixing to avoid over-parameterization
    when seq_len is small (e.g. 14 → 256 was 18x expansion; now 14 → 56 = 4x).
    """

    def __init__(self, seq_len, d_model, d_ff, dropout, d_ff_time=None):
        super(MixerBlock, self).__init__()

        # V2: Use separate (smaller) expansion for time-mixing MLP
        # Default: 4x seq_len if not specified, to avoid 18x over-expansion
        if d_ff_time is None:
            d_ff_time = max(seq_len * 4, 32)

        # Time-mixing MLP (operates across temporal dim)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, d_ff_time),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff_time, seq_len),
            nn.Dropout(dropout),
        )

        # Feature-mixing MLP (operates across feature dim)
        self.feat_norm = nn.LayerNorm(d_model)
        self.feat_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        # Time-mixing
        residual = x
        x = self.time_norm(x)
        x = x.permute(0, 2, 1)       # (batch, d_model, seq_len)
        x = self.time_mlp(x)
        x = x.permute(0, 2, 1)       # (batch, seq_len, d_model)
        x = x + residual

        # Feature-mixing
        residual = x
        x = self.feat_norm(x)
        x = self.feat_mlp(x)
        x = x + residual

        return x


class TSMixerModel(nn.Module):

    def __init__(self, config: Dict):
        super(TSMixerModel, self).__init__()

        self.config = {
            'input_size': 23,
            'output_size': 7,
            'seq_len': 14,
            'd_model': 64,
            'd_ff': 128,
            'd_ff_time': None,    # V2: separate time-mixing FF dim (default: 4x seq_len)
            'num_blocks': 3,
            'dropout': 0.15,
            'output_strategy': 'temporal_proj',  # V2: default changed from 'flatten'
        }
        self.config.update(config)

        input_size = self.config['input_size']
        output_size = self.config['output_size']
        seq_len = self.config['seq_len']
        d_model = self.config['d_model']
        d_ff = self.config['d_ff']
        d_ff_time = self.config['d_ff_time']
        num_blocks = self.config['num_blocks']
        dropout = self.config['dropout']
        output_strategy = self.config['output_strategy']

        # Feature projection: input_size -> d_model
        self.feature_proj = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Mixer blocks (V2: pass separate d_ff_time)
        self.mixer_blocks = nn.ModuleList([
            MixerBlock(seq_len, d_model, d_ff, dropout, d_ff_time=d_ff_time)
            for _ in range(num_blocks)
        ])

        # Output head
        self.output_norm = nn.LayerNorm(d_model)

        if output_strategy == 'temporal_proj':
            # Temporal projection: project seq_len → output_size in time dim
            # Each forecast day gets a learned weighted mix of all input timesteps
            self.temporal_proj = nn.Linear(seq_len, output_size)
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        elif output_strategy == 'flatten':
            self.fc = nn.Sequential(
                nn.Linear(seq_len * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_size),
            )
        else:  # 'last' -- take last timestep
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, output_size),
            )

        self._output_strategy = output_strategy
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Break zero symmetry in output layer — prevents mean-collapse trap
        # where LayerNorm + zero bias → model predicts 0 → MSE stays at zero
        if self._output_strategy == 'flatten':
            nn.init.normal_(self.fc[-1].bias, mean=0.0, std=0.1)
        elif self._output_strategy == 'temporal_proj':
            nn.init.normal_(self.output_proj[-1].bias, mean=0.0, std=0.1)
        else:  # 'last'
            nn.init.normal_(self.fc[-1].bias, mean=0.0, std=0.1)

    def forward(self, x, **kwargs):
        # x: (batch, seq_len, input_size)

        x = self.feature_proj(x)
        x = self.input_norm(x)
        # x: (batch, seq_len, d_model)

        for block in self.mixer_blocks:
            x = block(x)

        x = self.output_norm(x)
        # x: (batch, seq_len, d_model)

        if self._output_strategy == 'temporal_proj':
            # Project 14 input steps → 7 output steps
            x = x.permute(0, 2, 1)        # (batch, d_model, seq_len)
            x = self.temporal_proj(x)      # (batch, d_model, output_size)
            x = x.permute(0, 2, 1)        # (batch, output_size, d_model)
            out = self.output_proj(x)      # (batch, output_size, 1)
            return out.squeeze(-1)         # (batch, output_size)
        elif self._output_strategy == 'flatten':
            x = x.reshape(x.size(0), -1)
        else:
            x = x[:, -1, :]  # take last timestep

        out = self.fc(x)
        return out


if __name__ == "__main__":

    print("TSMIXER MODEL TEST")

    # Test 1: Default config (temporal_proj)
    print("\n[Test 1] Default config (temporal_proj):")
    config = {
        'input_size': 23,
        'output_size': 7,
    }

    model = TSMixerModel(config)
    x = torch.randn(8, 14, 23)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Strategy:     {model._output_strategy}")
    assert y.shape == (8, 7), f"Expected (8, 7), got {y.shape}"
    # Verify temporal_proj produces varied outputs per forecast day
    std_across_days = y.std(dim=1).mean().item()
    print(f"  Avg std across 7 forecast days: {std_across_days:.4f}")
    print("Passed!")

    # Test 2: Flatten strategy
    print("\n[Test 2] Flatten strategy:")
    config = {
        'input_size': 16,
        'output_size': 3,
        'seq_len': 20,
        'd_model': 32,
        'd_ff': 64,
        'num_blocks': 2,
        'dropout': 0.2,
        'output_strategy': 'flatten',
    }

    model = TSMixerModel(config)
    x = torch.randn(8, 20, 16)
    y = model(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    assert y.shape == (8, 3), f"Expected (8, 3), got {y.shape}"
    print("Passed!")

    # Test 3: Last strategy
    print("\n[Test 3] Last strategy:")
    config = {
        'output_strategy': 'last',
    }

    model = TSMixerModel(config)
    x = torch.randn(4, 14, 23)
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
    print(f"  Input:  (batch_size, seq_len=14, input_size=23)")
    print(f"  Output: (batch_size, output_size=7)")
