"""PatchTST with channel-independence and RevIN (Nie et al., 2023; Kim et al., 2022).

Input  : (B, L, C)  past_values — L context, C feature channels.
Output : (B, H)     forecast of the target channel (channel 0 = log_ret).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Channel-independent Reversible Instance Normalization."""

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        self._mean = None
        self._std = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        # x: (B, L, C) for norm, (B, H, C) for denorm
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.gamma + self.beta
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            x = x * self._std + self._mean
            return x
        raise ValueError(mode)


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class PatchTST(nn.Module):
    def __init__(
        self,
        num_channels: int,
        context_length: int,
        horizon: int,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 128,
        n_heads: int = 8,
        encoder_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        fc_dropout: float = 0.1,
        revin_affine: bool = True,
        learn_pos_embed: bool = True,
        target_index: int = 0,
    ):
        super().__init__()
        self.C = num_channels
        self.L = context_length
        self.H = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.target_index = target_index

        # Spec states num_patches = 14 for L=60, patch_len=8, stride=4, which
        # corresponds to floor((L - patch_len) / stride) + 1 (no padding).
        self.num_patches = (context_length - patch_len) // stride + 1
        pad_total = (self.num_patches - 1) * stride + patch_len - context_length
        self.pad_left = max(pad_total, 0)

        self.revin = RevIN(num_channels, affine=revin_affine)
        self.patch_proj = nn.Linear(patch_len, d_model)

        if learn_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            pe = torch.zeros(self.num_patches, d_model)
            pos = torch.arange(self.num_patches).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pos_embed", pe.unsqueeze(0))

        self.input_drop = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([
            _EncoderBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(encoder_layers)
        ])

        self.fc_drop = nn.Dropout(fc_dropout)
        self.head = nn.Linear(self.num_patches * d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        x = self.revin(x, "norm")                                    # (B, L, C)
        x = x.permute(0, 2, 1)                                       # (B, C, L)
        if self.pad_left > 0:
            x = nn.functional.pad(x, (self.pad_left, 0), mode="replicate")
        # unfold into patches: (B, C, num_patches, patch_len)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, C, P, PL = patches.shape
        tokens = self.patch_proj(patches.reshape(B * C, P, PL))      # (B*C, P, d)
        tokens = tokens + self.pos_embed
        tokens = self.input_drop(tokens)
        for blk in self.encoder:
            tokens = blk(tokens)
        flat = tokens.reshape(B, C, P * tokens.size(-1))             # (B, C, P*d)
        out = self.head(self.fc_drop(flat))                          # (B, C, H)
        out = out.permute(0, 2, 1)                                   # (B, H, C)

        # Denormalize in feature space, then pick target channel.
        # RevIN stored mean/std over L; apply same per-channel affine inversion.
        out = self.revin(out, "denorm")                              # (B, H, C)
        return out[..., self.target_index]                           # (B, H)
