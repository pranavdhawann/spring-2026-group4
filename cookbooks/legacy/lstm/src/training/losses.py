"""Custom losses for log-return forecasting.

BoundedAntiZeroHuber combines:
  - Huber base term (robust to fat tails)
  - Target-magnitude weighting (upweights larger-magnitude targets)
  - Anti-zero penalty: a Gaussian bump centered at 0
  - Directional auxiliary loss: BCE on sign(target > 0)

Predictions are assumed to already be bounded to a sane range by the model head
(see `LSTMForecaster.output_bound`), so no explicit clamp is applied here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundedAntiZeroHuber(nn.Module):
    def __init__(
        self,
        delta: float = 1.0,
        zero_alpha: float = 0.05,
        zero_sigma: float = 0.25,
        mag_weight_alpha: float = 0.0,
        mag_weight_power: float = 1.0,
        mag_weight_scale: float = 1.0,
        mag_weight_cap: float = 5.0,
        direction_alpha: float = 0.0,
    ):
        super().__init__()
        if zero_sigma <= 0:
            raise ValueError("zero_sigma must be > 0")
        if mag_weight_scale <= 0:
            raise ValueError("mag_weight_scale must be > 0")
        self.huber = nn.HuberLoss(delta=delta, reduction="none")
        self.zero_alpha = zero_alpha
        self.zero_sigma = zero_sigma
        self.mag_weight_alpha = mag_weight_alpha
        self.mag_weight_power = mag_weight_power
        self.mag_weight_scale = mag_weight_scale
        self.mag_weight_cap = mag_weight_cap
        self.direction_alpha = direction_alpha

    def _weighted_huber(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber_elem = self.huber(pred, target)
        if self.mag_weight_alpha <= 0:
            return huber_elem.mean()
        norm_mag = torch.abs(target) / self.mag_weight_scale
        weights = 1.0 + self.mag_weight_alpha * (norm_mag ** self.mag_weight_power)
        if self.mag_weight_cap is not None:
            weights = torch.clamp(weights, max=self.mag_weight_cap)
        return (weights * huber_elem).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self._weighted_huber(pred, target)
        # Gaussian bump at 0: high when |pred| << sigma, decays to 0 as |pred| grows.
        zero_penalty = torch.exp(-0.5 * (pred / self.zero_sigma) ** 2).mean()
        total = base + self.zero_alpha * zero_penalty
        if self.direction_alpha > 0:
            direction_target = (target > 0).to(dtype=pred.dtype)
            direction_loss = F.binary_cross_entropy_with_logits(pred, direction_target)
            total = total + self.direction_alpha * direction_loss
        return total
