"""
loss_gnn.py — Combined loss function for Temporal GNN stock forecasting.

Components:
    1. Log-return regression loss: MSE (original) or Huber/SmoothL1 (recommended)
    2. Relative price error (scale-invariant)
    3. Directional loss (weighted BCE on sign of log return, magnitude-weighted)

FIX L1: Previous ``direction_loss = (1 - tanh(10*pred) * sign(tgt)) / 2`` was the
primary reason the model collapsed to ŷ≈0.  At ŷ=0 the tanh derivative is 10
(because of the 10× scaling), so its gradient dominated the combined loss and
pushed predictions toward 0 whenever the sign of the target was noisy (which is
most of the time for 1-day log returns).  The new formulation uses BCE on the
sign of the target, weighted by |target| so tiny moves don't dominate, and is
off by default (gamma=0.01 in config_gnn.yaml).

NEW: Huber loss (SmoothL1) replaces MSE as the primary regression loss.
Daily log returns contain outliers (earnings surprises, macro shocks).
MSE penalises these outliers quadratically and can dominate training;
Huber is quadratic near zero (for typical small returns) but linear in
the tails, giving more robust gradients. Set loss_type="huber" in config.

NEW: Label smoothing for directional targets.  Daily returns are
approximately 50/50 up/down, making the hard-label BCE very noisy.
Smoothed labels (e.g. 0.05/0.95 instead of 0/1) reduce over-confidence
and help the model not collapse to predicting the most common direction.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Multi-component loss for stock forecasting.

    Args:
        alpha:              Weight for regression loss (default: 1.0)
        beta:               Weight for relative price error loss (default: 0.1)
        gamma:              Weight for directional loss (default: 0.1)
        direction_loss_type: "weighted_bce" (new) or "soft_tanh" (legacy,
            retained for reproducibility but do not use — see fix note above).
        loss_type:          "mse" or "huber" — primary regression loss type.
            "huber" (SmoothL1) is more robust to log-return outliers.
        huber_delta:        Delta parameter for Huber loss (default: 0.01).
            Roughly: errors smaller than delta are quadratic, larger are linear.
            For log returns (typical magnitude ~0.01), delta=0.01 means the
            transition from quadratic to linear happens at a 1% daily return,
            which is a reasonable threshold.
        label_smoothing:    Smoothing for directional BCE targets (default: 0.05).
            Target 0 becomes ``label_smoothing``, target 1 becomes
            ``1 - label_smoothing``.  Set to 0.0 to disable.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.1,
        gamma: float = 0.1,
        delta: float = 0.1,
        direction_loss_type: str = "weighted_bce",
        loss_type: str = "huber",
        huber_delta: float = 0.01,
        label_smoothing: float = 0.05,
        direction_logit_scale: float = 100.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.direction_loss_type = direction_loss_type
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.label_smoothing = label_smoothing
        # FIX L2: log-return predictions have magnitude ~1e-2, so using them
        # directly as BCE logits puts sigmoid≈0.5 and pins the loss at ln(2).
        # Scale by ~1/typical_daily_std so logits land in a learnable range.
        self.direction_logit_scale = direction_logit_scale

    def _regression_loss(
        self, pred_lr: torch.Tensor, target_lr: torch.Tensor
    ) -> torch.Tensor:
        """Primary regression loss: Huber (recommended) or MSE."""
        if self.loss_type == "huber":
            # F.huber_loss uses 'mean' reduction by default.
            # delta controls the transition from quadratic to linear.
            return F.huber_loss(pred_lr, target_lr, delta=self.huber_delta, reduction="mean")
        elif self.loss_type == "smooth_l1":
            # Alias for Huber with beta=1 (PyTorch's SmoothL1 uses 'beta' not 'delta').
            return F.smooth_l1_loss(pred_lr, target_lr, reduction="mean")
        else:
            # Default: MSE (original behaviour, kept for backward compatibility).
            return F.mse_loss(pred_lr, target_lr)

    def _direction_loss(
        self, logits: torch.Tensor, target_lr: torch.Tensor
    ) -> torch.Tensor:
        """Return a differentiable scalar directional loss in [0, ~1].

        weighted_bce: BCEWithLogits on sign(target), weighted by |target|.
            - Uses pred_lr directly as the logit (no 10× scaling), so at
              ŷ=0 the gradient is bounded and does not dominate the MSE term.
            - |target| weighting means trivially-small moves contribute
              proportionally little gradient noise.
            - Label smoothing reduces over-confidence on the noisy 0/1 targets.

        soft_tanh: legacy implementation, preserved only for reproducibility.
        """
        if self.direction_loss_type == "soft_tanh":
            # Legacy — see fix note at top of file.  Retained only for reproducibility.
            d = -torch.mean(torch.tanh(logits * 10.0) * torch.sign(target_lr))
            return (1.0 + d) / 2.0

        # Default: weighted BCE on sign(target) with optional label smoothing.
        # When a separate DirectionHead is used, `logits` are already on a
        # natural scale (initialized ~N(0,1)) so no rescaling is needed.
        target_sign = (target_lr > 0).float()

        # Apply label smoothing: 0 → smoothing, 1 → (1 - smoothing)
        if self.label_smoothing > 0.0:
            eps = float(self.label_smoothing)
            target_sign = target_sign * (1.0 - eps) + (1.0 - target_sign) * eps

        weight = target_lr.abs().clamp(min=1e-4)
        weight = weight / weight.mean().clamp(min=1e-8)  # normalize so mean weight == 1
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target_sign,
            weight=weight,
            reduction="mean",
        )
        return bce

    def forward(
        self,
        pred_lr: torch.Tensor,
        target_lr: torch.Tensor,
        pred_close: torch.Tensor,
        target_close: torch.Tensor,
        direction_logits: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Compute combined loss.

        Args:
            pred_lr:           (N, H) predicted log returns
            target_lr:         (N, H) target log returns
            pred_close:        (N, H) predicted close prices (reconstructed)
            target_close:      (N, H) target close prices
            direction_logits:  (N, H) logits from separate direction head (optional).
                               If provided, used for directional BCE instead of pred_lr.
                               This decouples the direction and regression gradients.

        Returns:
            total_loss: scalar tensor
            components: dict with individual loss values
        """
        # 1. Primary: regression loss on log returns (Huber or MSE)
        lr_loss = self._regression_loss(pred_lr, target_lr)

        # 2. Relative price error (scale-invariant, no bias toward high-priced stocks)
        relative_error = (pred_close - target_close) / (target_close.abs() + 1e-8)
        price_loss = F.mse_loss(relative_error, torch.zeros_like(relative_error))

        # 3. Directional loss — uses separate head logits if available,
        #    so BCE gradients don't interfere with the regression head.
        if self.gamma > 0:
            dir_input = direction_logits if direction_logits is not None else pred_lr
            direction_loss = self._direction_loss(dir_input, target_lr)
        else:
            direction_loss = torch.zeros((), device=pred_lr.device, dtype=pred_lr.dtype)

        # 4. Variance regularizer (FIX L4): prevent prediction collapse to
        # near-zero.  Both Huber and price_loss reward predicting the
        # unconditional mean (≈0), creating a positive feedback loop where
        # shrinking predictions → shrinking gradients → stuck at zero.
        # Penalise when pred_std < target_std to keep predictions spread.
        if self.delta > 0 and pred_lr.numel() > 1:
            pred_std = pred_lr.std()
            tgt_std = target_lr.std().detach()
            var_ratio = pred_std / (tgt_std + 1e-8)
            var_loss = F.relu(1.0 - var_ratio)  # 0 when pred_std >= tgt_std
        else:
            var_loss = torch.zeros((), device=pred_lr.device, dtype=pred_lr.dtype)

        total = (self.alpha * lr_loss + self.beta * price_loss
                 + self.gamma * direction_loss + self.delta * var_loss)

        components = {
            "lr_loss": lr_loss.item(),
            "price_loss": price_loss.item(),
            "direction_loss": direction_loss.detach().item(),
            "var_loss": var_loss.item(),
            "total_loss": total.item(),
        }

        return total, components


def reconstruct_prices(
    log_returns: torch.Tensor,
    last_close: torch.Tensor,
) -> torch.Tensor:
    """
    Convert log-return forecasts back to close prices.
    
    Args:
        log_returns: (N, H) predicted log returns for t+1..t+H
        last_close:  (N, 1) or (N,) last known close price at time t
    
    Returns:
        (N, H) predicted close prices
    """
    if last_close.dim() == 1:
        last_close = last_close.unsqueeze(-1)  # (N, 1)
    
    cum_returns = torch.cumsum(log_returns, dim=-1)  # (N, H)
    return last_close * torch.exp(cum_returns)        # (N, H)
