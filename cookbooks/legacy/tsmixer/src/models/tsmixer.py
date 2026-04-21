"""TSMixer with RevIN for multi-step log-return forecasting."""
from __future__ import annotations
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022). Per-sample, per-feature."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def _stats(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        return mean, std

    def normalize(self, x: torch.Tensor):
        mean, std = self._stats(x)
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std

    def denormalize(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, feat_idx: int):
        # y: [B, H] predictions for a single feature (target index)
        if self.affine:
            y = (y - self.beta[feat_idx]) / self.gamma[feat_idx]
        y = y * std[:, 0, feat_idx : feat_idx + 1] + mean[:, 0, feat_idx : feat_idx + 1]
        return y


class TimeMixing(nn.Module):
    def __init__(self, T: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(T)
        self.fc1 = nn.Linear(T, T)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(T, T)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, C]
        y = x.transpose(1, 2)  # [B, C, T]
        y = self.norm(y)
        y = self.drop(self.fc2(self.drop(self.act(self.fc1(y)))))
        y = y.transpose(1, 2)
        return x + y


class FeatureMixing(nn.Module):
    def __init__(self, C: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(C)
        self.fc1 = nn.Linear(C, ff_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ff_dim, C)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, C]
        y = self.norm(x)
        y = self.drop(self.fc2(self.drop(self.act(self.fc1(y)))))
        return x + y


class MixerBlock(nn.Module):
    def __init__(self, T: int, C: int, ff_dim: int, dropout: float):
        super().__init__()
        self.time_mix = TimeMixing(T, dropout)
        self.feat_mix = FeatureMixing(C, ff_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feat_mix(self.time_mix(x))


class TSMixer(nn.Module):
    """Forecasts `horizon` future values of feature at `target_idx`."""

    def __init__(
        self,
        lookback: int,
        n_features: int,
        horizon: int,
        target_idx: int,
        n_blocks: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.1,
        num_tickers: int = 0,
        ticker_embed_dim: int = 8,
    ):
        super().__init__()
        self.target_idx = target_idx
        self.horizon = horizon
        self.num_tickers = num_tickers
        self.ticker_embed_dim = ticker_embed_dim if num_tickers > 0 else 0
        self.time_weights = nn.Parameter(torch.linspace(0.5, 1.0, lookback, dtype=torch.float32))
        self.revin = RevIN(n_features)
        mixer_features = n_features + self.ticker_embed_dim
        self.ticker_embed = (
            nn.Embedding(num_tickers, self.ticker_embed_dim) if self.ticker_embed_dim > 0 else None
        )
        self.blocks = nn.ModuleList(
            [MixerBlock(lookback, mixer_features, ff_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor, ticker_id: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, T, C]
        x = x * self.time_weights.view(1, -1, 1)
        x_norm, mean, std = self.revin.normalize(x)
        h = x_norm
        if self.ticker_embed is not None:
            if ticker_id is None:
                ticker_id = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
            embed = self.ticker_embed(ticker_id.long()).unsqueeze(1).expand(-1, x.shape[1], -1)
            h = torch.cat([h, embed], dim=-1)
        for blk in self.blocks:
            h = blk(h)
        # Take target-feature trajectory and map T -> H
        target_series = h[..., self.target_idx]  # [B, T]
        y_norm = self.head(target_series)  # [B, H]
        y = self.revin.denormalize(y_norm, mean, std, self.target_idx)
        return y
