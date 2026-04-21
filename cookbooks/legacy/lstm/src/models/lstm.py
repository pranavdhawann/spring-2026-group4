"""Stacked LSTM forecaster: LSTM(128) -> LSTM(64) -> Linear(horizon)."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int = 5,
        hidden1: int = 128,
        hidden2: int = 64,
        dropout: float = 0.2,
        output_bound: float | None = 5.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.horizon = horizon
        self.output_bound = output_bound
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Linear(hidden2, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)           # return_sequences=True
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x[:, -1, :])
        out = self.head(x)
        if self.output_bound is not None:
            b = self.output_bound
            out = b * torch.tanh(out / b)  # smooth clamp into [-b, b]
        return out
