"""
encoders_gnn.py — Modality encoders for the Temporal GNN.

1. TSEncoder: Time series encoder (TCN or Transformer) — now accepts 21 features
2. NewsEncoder: News article encoder (FinBERT embeddings → attention aggregation)
3. ReportsEncoder: Financial reports encoder — real SEC XBRL fundamentals
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Time Series Encoder — TCN
# ===========================================================================

class CausalConv1d(nn.Module):
    """Causal convolution: output at time t only depends on inputs at times ≤ t."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = out.transpose(1, 2)
        out = self.norm1(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout2(out)
        return out + residual


class TCNEncoder(nn.Module):
    """
    Temporal Convolution Network for time series.
    Input: (B, W, input_dim) where input_dim = 21 (6 base + 15 technical indicators)
    Output: (B, output_dim) — attention-pooled across the full W-day window.

    FIX E7: The previous implementation returned ``out[:, :, -1]`` only, which
    discarded 59/60 of the temporal information computed by the stacked causal
    convolutions.  The entire point of the TCN is to model multi-scale
    temporal patterns; throwing away all but the last step was a silent
    bottleneck that prevented the model from learning anything beyond
    t-1 → t+1 correlations.  We now apply a softmax attention pool over the
    time axis (learned per-feature) with a residual on the last step, which:
      (a) retains the causal structure,
      (b) lets the encoder choose which timesteps matter for the forecast,
      (c) gracefully reduces to the old behaviour if attention collapses.
    """

    def __init__(self, input_dim=21, channels=None, kernel_size=3, dropout=0.1, output_dim=256):
        super().__init__()
        if channels is None:
            channels = [64, 128, 128, 256]

        dilations = [2 ** i for i in range(len(channels))]

        layers = []
        for i, (out_ch, dilation) in enumerate(zip(channels, dilations)):
            in_ch = input_dim if i == 0 else channels[i - 1]
            layers.append(TCNResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        # Attention pool over the time axis: scalar score per timestep.
        self.temporal_attn = nn.Linear(channels[-1], 1)
        self.output_proj = nn.Linear(channels[-1], output_dim) if channels[-1] != output_dim else nn.Identity()

    def forward(self, x):
        """x: (B, W, input_dim) → (B, output_dim)

        Returns an attention-weighted pool across the full W-day window plus
        a residual on the last timestep.
        """
        x = x.transpose(1, 2)            # (B, input_dim, W)
        out = self.network(x)             # (B, C, W)
        out_tw = out.transpose(1, 2)      # (B, W, C)

        # Attention weights over time.
        attn_scores = self.temporal_attn(out_tw)           # (B, W, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)   # (B, W, 1)
        attn_pooled = (out_tw * attn_weights).sum(dim=1)   # (B, C)

        # Residual on the last timestep (the causal-conv invariant path).
        last_step = out[:, :, -1]                          # (B, C)
        pooled = attn_pooled + last_step                    # (B, C)
        return self.output_proj(pooled)


# ===========================================================================
# 1b. Time Series Encoder — Transformer
# ===========================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTSEncoder(nn.Module):
    def __init__(self, input_dim=21, d_model=256, nhead=4, num_layers=4, dropout=0.1, output_dim=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # FIX E7: attention pool over time instead of last-step-only.
        self.temporal_attn = nn.Linear(d_model, 1)
        self.output_proj = nn.Linear(d_model, output_dim) if d_model != output_dim else nn.Identity()

    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, x):
        """x: (B, W, input_dim) → (B, output_dim)

        Returns an attention-weighted pool over the full sequence plus a
        residual on the last timestep (see TCNEncoder fix note).
        """
        x = self.input_proj(x)
        x = self.pos_enc(x)
        mask = self._generate_causal_mask(x.size(1), x.device)
        x = self.transformer(x, mask=mask)                 # (B, W, d_model)

        attn_scores = self.temporal_attn(x)                 # (B, W, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)    # (B, W, 1)
        pooled = (x * attn_weights).sum(dim=1)              # (B, d_model)
        last_step = x[:, -1, :]
        pooled = pooled + last_step
        return self.output_proj(pooled)


def build_ts_encoder(config: dict) -> nn.Module:
    """Build TS encoder. Input dim now = 21 (base + technical indicators)."""
    model_cfg = config["model"]
    encoder_type = model_cfg.get("ts_encoder", "tcn")
    dropout = model_cfg.get("dropout", 0.1)
    output_dim = model_cfg.get("d_ts", 256)
    input_dim = model_cfg.get("ts_input_dim", 21)

    if encoder_type == "tcn":
        return TCNEncoder(
            input_dim=input_dim,
            channels=model_cfg.get("tcn_channels", [64, 128, 128, 256]),
            kernel_size=model_cfg.get("tcn_kernel_size", 3),
            dropout=dropout,
            output_dim=output_dim,
        )
    elif encoder_type == "transformer":
        return TransformerTSEncoder(
            input_dim=input_dim, d_model=output_dim,
            nhead=model_cfg.get("transformer_heads", 4),
            num_layers=model_cfg.get("transformer_layers", 4),
            dropout=dropout, output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown ts_encoder: {encoder_type}")


# ===========================================================================
# 2. News Encoder
# ===========================================================================

class NewsEncoder(nn.Module):
    """
    News encoder using pre-computed FinBERT/MiniLM embeddings.
    Attention-weighted mean over same-day articles, with [NO_NEWS] fallback.
    """

    def __init__(self, embedding_dim: int = 768, output_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.attn_linear = nn.Linear(embedding_dim, 1)
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.no_news_embedding = nn.Parameter(torch.randn(output_dim) * 0.02)

    def aggregate_daily(self, article_embeddings):
        if article_embeddings.size(0) == 0:
            return None
        if article_embeddings.size(0) == 1:
            return article_embeddings.squeeze(0)
        attn_scores = self.attn_linear(article_embeddings)
        attn_weights = F.softmax(attn_scores, dim=0)
        return (attn_weights * article_embeddings).sum(dim=0)

    def forward(self, news_embeddings: list) -> torch.Tensor:
        """
        Args:
            news_embeddings: List of N elements, each is either
                tensor(num_articles, embed_dim) or None.
        Returns:
            (N, output_dim)
        """
        batch_outputs = []
        for emb in news_embeddings:
            if emb is not None and emb.numel() > 0:
                h_day = self.aggregate_daily(emb)
                if h_day is not None:
                    h_news = self.projection(h_day)
                else:
                    h_news = self.no_news_embedding
            else:
                h_news = self.no_news_embedding
            batch_outputs.append(h_news)
        return torch.stack(batch_outputs, dim=0)


# ===========================================================================
# 3. Reports Encoder — Real SEC XBRL fundamentals
# ===========================================================================

# Common fundamental metrics across the 3 financial statements
# These are the union of metrics found in the SEC XBRL filings
FUNDAMENTAL_METRICS = [
    # Balance Sheet
    "balance_sheets_Assets",
    "balance_sheets_AssetsCurrent",
    "balance_sheets_Liabilities",
    "balance_sheets_LiabilitiesCurrent",
    "balance_sheets_StockholdersEquity",
    "balance_sheets_AccountsPayableCurrent",
    "balance_sheets_AccountsReceivableNetCurrent",
    "balance_sheets_InventoryNet",
    "balance_sheets_PropertyPlantAndEquipmentNet",
    "balance_sheets_CommonStockValue",
    # Cash Flow
    "statement_of_cash_flows_NetCashProvidedByUsedInOperatingActivities",
    "statement_of_cash_flows_NetCashProvidedByUsedInInvestingActivities",
    "statement_of_cash_flows_NetCashProvidedByUsedInFinancingActivities",
    "statement_of_cash_flows_CashAndCashEquivalentsPeriodIncreaseDecrease",
    "statement_of_cash_flows_PaymentsToAcquirePropertyPlantAndEquipment",
    "statement_of_cash_flows_ProceedsFromSaleOfPropertyPlantAndEquipment",
    # Equity
    "statement_of_equity_RetainedEarningsAccumulatedDeficit",
    "statement_of_equity_DividendsCommonStockCash",
    "statement_of_equity_StockIssuedDuringPeriodValueShareBasedCompensation",
    "statement_of_equity_DividendsPayableAmountPerShare",
]

NUM_FUNDAMENTAL_FEATURES = len(FUNDAMENTAL_METRICS)


class ReportsEncoder(nn.Module):
    """
    Financial reports encoder using real SEC XBRL fundamental data.

    Extracts point-in-time fundamental features from parsed filing JSONs.
    Carries the last available report forward in time until a new filing appears.

    Input: raw fundamental dict {filing_date: {metric: value}}
    Output: (N, d_rep) embedding per stock

    Architecture:
        1. Standardize metrics to fixed-size vector (20 dims)
        2. Log-transform + z-score normalization
        3. MLP: 20 → 128 → d_rep
    """

    def __init__(self, d_rep: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_rep = d_rep
        self.n_metrics = NUM_FUNDAMENTAL_FEATURES
        self.metric_to_idx = {m: i for i, m in enumerate(FUNDAMENTAL_METRICS)}

        self.encoder = nn.Sequential(
            nn.Linear(self.n_metrics, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, d_rep),
            nn.LayerNorm(d_rep),
            nn.GELU(),
        )

        # Learned embedding for stocks with no fundamental data.
        # FIX E8: was initialized to zeros, which was a silent bug — 10/100
        # tickers have no fundamentals, so all 10 received an identical zero
        # slice that could not be distinguished from a real all-zero
        # fundamentals vector.  Now initialized with a small Gaussian so it
        # starts as a genuine learnable "missing" token.
        self.no_report_embedding = nn.Parameter(torch.randn(d_rep) * 0.02)

        # Running statistics for normalization (set during preprocessing)
        self.register_buffer("metric_mean", torch.zeros(self.n_metrics))
        self.register_buffer("metric_std", torch.ones(self.n_metrics))

    def compute_normalization_stats(
        self,
        all_fundamentals: Dict[str, Dict[str, Dict[str, float]]],
    ):
        """Compute mean/std for each metric across all tickers and dates."""
        metric_values = {m: [] for m in FUNDAMENTAL_METRICS}

        for ticker, filings in all_fundamentals.items():
            for filing_date, metrics in filings.items():
                for metric_name in FUNDAMENTAL_METRICS:
                    val = metrics.get(metric_name)
                    if val is not None and val != 0:
                        # Log transform for large values
                        log_val = np.sign(val) * np.log1p(abs(val))
                        metric_values[metric_name].append(log_val)

        means = []
        stds = []
        for m in FUNDAMENTAL_METRICS:
            vals = metric_values[m]
            if vals:
                means.append(np.mean(vals))
                stds.append(max(np.std(vals), 1e-8))
            else:
                means.append(0.0)
                stds.append(1.0)

        # ── FIX E4: use .copy_() to preserve registered buffers ──
        # Plain assignment (self.metric_mean = ...) replaces the buffer with a
        # regular tensor, breaking .to(device), state_dict(), and checkpoint
        # save/load.  .copy_() updates the existing buffer in-place.
        self.metric_mean.copy_(torch.tensor(means, dtype=torch.float))
        self.metric_std.copy_(torch.tensor(stds, dtype=torch.float))

    def get_point_in_time_features(
        self,
        fundamentals: Dict[str, Dict[str, float]],
        date: pd.Timestamp,
    ) -> Optional[torch.Tensor]:
        """
        Get the most recent fundamental features available before `date`.
        Point-in-time safe: only uses filings with filing_date <= date.

        Returns:
            (n_metrics,) tensor or None if no data available
        """
        if not fundamentals:
            return None

        # Find the most recent filing before the given date
        valid_dates = []
        for fd_str in fundamentals.keys():
            try:
                fd = pd.Timestamp(fd_str)
                if fd <= date:
                    valid_dates.append((fd, fd_str))
            except (ValueError, TypeError):
                continue

        if not valid_dates:
            return None

        # Most recent filing
        valid_dates.sort(reverse=True)
        _, latest_date_str = valid_dates[0]
        metrics = fundamentals[latest_date_str]

        # Build feature vector on the same device as the normalization buffers
        # (buffers follow the model to CUDA via .to(device))
        feature_vec = torch.zeros(self.n_metrics, device=self.metric_mean.device)
        for metric_name, idx in self.metric_to_idx.items():
            val = metrics.get(metric_name)
            if val is not None:
                # Log transform
                log_val = np.sign(val) * np.log1p(abs(val))
                feature_vec[idx] = log_val

        # Normalize
        feature_vec = (feature_vec - self.metric_mean) / self.metric_std

        return feature_vec

    def forward(
        self,
        fundamental_features: List[Optional[torch.Tensor]],
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Args:
            fundamental_features: List of N tensors (n_metrics,) or None per stock
            device: target device

        Returns:
            (N, d_rep) embeddings
        """
        dev = device or torch.device("cpu")
        outputs = []

        for feat in fundamental_features:
            if feat is not None:
                feat = feat.to(dev)
                h = self.encoder(feat)
                outputs.append(h)
            else:
                outputs.append(self.no_report_embedding.to(dev))

        return torch.stack(outputs, dim=0)
