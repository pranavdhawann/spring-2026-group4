from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedReturnLoss(nn.Module):
    """MSE plus quantile pinball penalty to discourage variance collapse."""

    def __init__(self, quantile_lambda: float = 0.3, quantiles: Sequence[float] = (0.1, 0.9)):
        super().__init__()
        self.ql = float(quantile_lambda)
        self.quantiles = tuple(float(q) for q in quantiles)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        q_loss = pred.new_tensor(0.0)
        errors = target - pred
        for q in self.quantiles:
            q_loss = q_loss + torch.maximum(q * errors, (q - 1.0) * errors).mean()
        return mse + self.ql * q_loss


class ReturnLoss(nn.Module):
    """Configurable return loss. Default is mixed (MSE + quantile)."""

    def __init__(
        self,
        loss_type: str = "mixed",
        delta: float = 0.003,
        lam: float = 0.3,
        q: float | Iterable[float] = 0.9,
    ):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type == "mixed":
            quantiles: tuple[float, ...]
            if isinstance(q, (tuple, list)):
                quantiles = tuple(float(v) for v in q)
            else:
                # Keep backward compatibility: when given a single q, pair it with lower tail.
                quantiles = (1.0 - float(q), float(q))
            self.base: nn.Module = MixedReturnLoss(quantile_lambda=lam, quantiles=quantiles)
        elif self.loss_type == "mse":
            self.base = nn.MSELoss()
        elif self.loss_type == "huber":
            self.base = nn.HuberLoss(delta=delta)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.base(pred, target)


# Backwards-compat alias
HuberPlusQuantile = ReturnLoss
