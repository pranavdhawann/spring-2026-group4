"""
fusion_gnn.py — Cross-modal fusion module for combining time series, news, and reports embeddings.

Two modes (configurable via config_gnn.yaml: fusion_mode):
    - "concat": Simple concatenation + linear projection (default for v1)
    - "cross_attention": Cross-modal attention between TS and news, then concat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention module: Q from one modality, K/V from another.
    Includes skip connection.
    """

    def __init__(self, d_model: int = 256, nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (N, d_model) or (N, 1, d_model)
            key:   (N, d_model) or (N, 1, d_model)
            value: (N, d_model) or (N, 1, d_model)

        Returns:
            (N, d_model) — attended + skip connection
        """
        # Add sequence dimension if needed (cross-attention over single vectors)
        if query.dim() == 2:
            query = query.unsqueeze(1)  # (N, 1, d_model)
        if key.dim() == 2:
            key = key.unsqueeze(1)
        if value.dim() == 2:
            value = value.unsqueeze(1)

        attn_out, _ = self.attn(query, key, value)  # (N, 1, d_model)
        attn_out = self.dropout(attn_out)

        # Skip connection + LayerNorm
        out = self.norm(query + attn_out)
        return out.squeeze(1)  # (N, d_model)


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion: [h_ts | h_news | h_rep] → Linear → LayerNorm → GELU → h_fused
    """

    def __init__(
        self, d_ts: int = 256, d_news: int = 256, d_rep: int = 256, d_fused: int = 256
    ):
        super().__init__()
        total_dim = d_ts + d_news + d_rep
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
        )

    def forward(
        self,
        h_ts: torch.Tensor,
        h_news: torch.Tensor,
        h_rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_ts:   (N, d_ts)
            h_news: (N, d_news)
            h_rep:  (N, d_rep)

        Returns:
            (N, d_fused)
        """
        h_cat = torch.cat([h_ts, h_news, h_rep], dim=-1)  # (N, d_ts + d_news + d_rep)
        return self.fusion(h_cat)  # (N, d_fused)


class CrossAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion:
        1. TS attends to News → h_ts_att (with skip connection)
        2. News attends to TS → h_news_att (with skip connection)
        3. When h_rep is zeros (v1), pass through unchanged
        4. Concat [h_ts_att | h_news_att | h_rep] → Linear → LayerNorm → GELU → h_fused
    """

    def __init__(
        self,
        d_ts: int = 256,
        d_news: int = 256,
        d_rep: int = 256,
        d_fused: int = 256,
        nhead: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Cross-attention: TS attends to News
        self.ts_attend_news = MultiHeadCrossAttention(
            d_model=d_ts, nhead=nhead, dropout=dropout
        )

        # Cross-attention: News attends to TS
        self.news_attend_ts = MultiHeadCrossAttention(
            d_model=d_news, nhead=nhead, dropout=dropout
        )

        # Final projection
        total_dim = d_ts + d_news + d_rep
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, d_fused),
            nn.LayerNorm(d_fused),
            nn.GELU(),
        )

    def forward(
        self,
        h_ts: torch.Tensor,
        h_news: torch.Tensor,
        h_rep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_ts:   (N, d_ts)
            h_news: (N, d_news)
            h_rep:  (N, d_rep) — zeros in v1

        Returns:
            (N, d_fused)
        """
        # Cross-attention between TS and News
        h_ts_att = self.ts_attend_news(query=h_ts, key=h_news, value=h_news)
        h_news_att = self.news_attend_ts(query=h_news, key=h_ts, value=h_ts)

        # h_rep passes through unchanged (zeros in v1 — no attention needed)
        h_cat = torch.cat([h_ts_att, h_news_att, h_rep], dim=-1)
        return self.fusion(h_cat)


# ===========================================================================
# Factory function
# ===========================================================================


def build_fusion(config: dict) -> nn.Module:
    """Build fusion module based on config."""
    model_cfg = config["model"]
    mode = model_cfg.get("fusion_mode", "concat")
    d_ts = model_cfg.get("d_ts", 256)
    d_news = model_cfg.get("d_news", 256)
    d_rep = model_cfg.get("d_rep", 256)
    d_fused = model_cfg.get("d_fused", 256)
    dropout = model_cfg.get("dropout", 0.1)

    if mode == "concat":
        return ConcatFusion(d_ts, d_news, d_rep, d_fused)
    elif mode == "cross_attention":
        return CrossAttentionFusion(
            d_ts, d_news, d_rep, d_fused, nhead=4, dropout=dropout
        )
    else:
        raise ValueError(f"Unknown fusion_mode: {mode}")
