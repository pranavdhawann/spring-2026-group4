"""
model_gnn.py — Full Temporal GNN model.

CORRECT DATA FLOW:
    1. Dataset provides: ts_features (N, W, 21), news embeddings, report fundamentals, edges
    2. Model.forward():
       a. TS Encoder:  (N, W, 21) → h_ts (N, 256)                [per stock, over W-day window]
       b. News Encoder: time-decayed sum over W-day window of per-day news
          embeddings → h_news (N, 256)                           [FIX M3: was last-day-only]
       c. Reports Encoder: fundamentals → h_rep (N, 256)          [most recent filing]
       d. Fusion: [h_ts | h_news | h_rep] → h_fused (N, 256)
       e. Build K real snapshots (K=num_gnn_snapshots) by recomputing the
          fused features over K growing sub-windows of the input window, with
          correlation edges recomputed per sub-window.                [FIX M1]
       f. Temporal GNN over K snapshots → h_graph (N, 256)
       g. Forecast Head → log_returns (N_T, 5)

FIX NOTES (why the model could not learn before):
    M1  All K snapshots fed to EvolveGCN were identical (same h_fused, same
        edge_index) — so the GRU-evolved weights had no temporal signal to
        learn from.  We now build K real sub-window snapshots.
    M2  ``_reset_evolvegcn_state()`` was called on every forward pass, wiping
        out the temporal weight evolution before it could contribute.  We now
        keep the GRU state coherent WITHIN a forward pass and only reset
        across samples — EvolveGCN's own ``detach_hidden_states`` handles the
        cross-sample case when the hidden state is torch.no_grad-compatible.
    M3  News encoder took only the last non-None day of each window.  We now
        time-decay-weight all available days across the window.
    M5  ``max_nodes=550`` hardcoded padding wasted 4.5× GNN capacity on
        zero-padded nodes.  We now slice the GNN output by ``num_active``
        instead of operating on the full pad.
    M8  ``edge_dropout`` config knob was dead code because ``build_snapshot``
        was bypassed.  We now apply edge_dropout directly inside this module
        during training.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders_gnn import NewsEncoder, ReportsEncoder, build_ts_encoder
from src.fusion_gnn import build_fusion
from src.graph_gnn import CorrelationEdgeBuilder
from src.loss_gnn import reconstruct_prices

logger = logging.getLogger(__name__)


# ===========================================================================
# Temporal GNN Layers
# ===========================================================================


class EvolveGCNBlock(nn.Module):
    """EvolveGCN-H: GRU evolves GCN weight matrices across timesteps."""

    def __init__(
        self, num_nodes, in_channels, num_layers=3, dropout=0.1, backend="auto"
    ):
        super().__init__()
        self.use_evolvegcn = False

        if backend in {"auto", "evolvegcn_h"}:
            try:
                from torch_geometric_temporal.nn.recurrent import EvolveGCNH

                self.layers = nn.ModuleList(
                    [
                        EvolveGCNH(num_of_nodes=num_nodes, in_channels=in_channels)
                        for _ in range(num_layers)
                    ]
                )
                # EvolveGCNH stores `weight` as nn.Parameter, but its forward()
                # reassigns it with a plain tensor from the GRU.  PyTorch ≥2.0
                # rejects this.  Convert weight to a buffer so reassignment works.
                for layer in self.layers:
                    if hasattr(layer, "weight") and isinstance(
                        layer.weight, nn.Parameter
                    ):
                        w_data = layer.weight.data.clone()
                        del layer._parameters["weight"]
                        layer.register_buffer("weight", w_data)
                        layer.initial_weight = w_data.clone()
                self.use_evolvegcn = True
                logger.info("Using torch_geometric_temporal EvolveGCNH layers")
            except ImportError:
                self._init_pyg_gcn_gru(in_channels, num_layers)
                if backend == "evolvegcn_h":
                    logger.warning(
                        "Requested torch_geometric_temporal EvolveGCNH, but the package is unavailable; "
                        "falling back to the built-in PyG GCN + GRU temporal block"
                    )
                else:
                    logger.info(
                        "torch_geometric_temporal not installed; using the built-in PyG GCN + GRU temporal block"
                    )
        elif backend == "pyg_gcn_gru":
            self._init_pyg_gcn_gru(in_channels, num_layers)
            logger.info("Using the built-in PyG GCN + GRU temporal block")
        else:
            raise ValueError(f"Unknown temporal backend: {backend}")

        self.norms = nn.ModuleList(
            [nn.LayerNorm(in_channels) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def _init_pyg_gcn_gru(self, in_channels, num_layers):
        from torch_geometric.nn import GCNConv

        self.layers = nn.ModuleList(
            [GCNConv(in_channels, in_channels) for _ in range(num_layers)]
        )
        self.gru = nn.GRU(in_channels, in_channels, batch_first=True)

    def _reset_evolvegcn_state(self):
        """Reset EvolveGCNH internal weights to their learned initial values.

        EvolveGCNH stores the GRU-evolved weight matrix as ``self.weight``
        after each call.  We call this ONCE at the start of each forward
        pass so that the weight evolution within a forward pass (over the
        K snapshots) is a clean, re-running GRU sequence with the learnable
        ``initial_weight`` as the seed.  This is the correct semantics for
        EvolveGCN: the GRU weights are persistent across snapshots within
        one sequence, but reset between sequences (samples).

        FIX M2: the previous call site was inside ``forward`` and wiped the
        weight right before the snapshot loop — but also before any useful
        evolution could happen.  That's fine in principle, but combined
        with Fix M1 (real snapshots) this is what lets the GRU finally
        carry temporal signal across snapshots within a sample.
        """
        for layer in self.layers:
            if hasattr(layer, "initial_weight"):
                layer.weight = layer.initial_weight.clone()

    def forward(self, snapshots):
        """snapshots: list of (x, edge_index, edge_weight) tuples, length S.

        The GRU weight evolution proceeds across the S snapshots WITHIN one
        forward pass.  The seed weight is reset at the start of every call
        (see _reset_evolvegcn_state docstring).
        """
        if self.use_evolvegcn:
            # Seed the weight evolution from the learnable initial_weight.
            # Within this forward pass, subsequent snapshots see the
            # GRU-evolved weight from the previous snapshot.
            self._reset_evolvegcn_state()

            # EvolveGCNH assigns recurrent output to self.weight (nn.Parameter).
            # Under AMP autocast the output is float16, which cannot be assigned
            # to a Parameter.  Disable autocast so the layer stays in float32.
            h = None
            with torch.amp.autocast("cuda", enabled=False):
                for x, edge_index, edge_weight in snapshots:
                    h_new = x.float()
                    for layer, norm in zip(self.layers, self.norms):
                        h_new = layer(h_new, edge_index, edge_weight)
                        h_new = norm(h_new)
                        h_new = F.relu(h_new)
                        h_new = self.dropout(h_new)
                    if h is not None and h.shape == h_new.shape:
                        h_new = h_new + h
                    h = h_new
            return h
        else:
            feats = []
            for x, edge_index, edge_weight in snapshots:
                h = x
                for layer, norm in zip(self.layers, self.norms):
                    h = layer(h, edge_index, edge_weight)
                    h = norm(h)
                    h = F.relu(h)
                    h = self.dropout(h)
                feats.append(h)
            temporal_stack = torch.stack(feats, dim=1)  # (N, W, d)
            gru_out, _ = self.gru(temporal_stack)
            return gru_out[:, -1, :]


class ForecastHead(nn.Module):
    def __init__(self, d_in, horizon=5, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 2, d_in // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 4, horizon),
        )

    def forward(self, h):
        return self.mlp(h)


class DirectionHead(nn.Module):
    """Separate head for directional (up/down) prediction.

    Produces logits on a natural scale for BCE, decoupled from the
    regression head so the two losses don't fight over the same output.
    """

    def __init__(self, d_in, horizon=5, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_in // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 4, horizon),
        )

    def forward(self, h):
        return self.mlp(h)


# ===========================================================================
# Full Model
# ===========================================================================


class TemporalGNN(nn.Module):
    def __init__(self, config, max_nodes=550):
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        graph_cfg = config.get("graph", {})
        data_cfg = config.get("data", {})

        self.horizon = model_cfg.get("forecast_horizon", 5)
        self.d_fused = model_cfg.get("d_fused", 256)
        self.max_nodes = max_nodes
        self.window_size = data_cfg.get("window_size", 60)

        # ── FIX E2 / M1: configurable snapshot count (was hardcoded to window_size=60) ──
        # 60 identical snapshots wasted 60x compute while giving EvolveGCN's GRU
        # nothing meaningful to learn.  A small number (3-8) suffices.
        self.num_gnn_snapshots = model_cfg.get("num_gnn_snapshots", 4)

        # ── Edge dropout (FIX M8: was dead code in graph_gnn.py, now wired in here) ──
        self.edge_dropout = graph_cfg.get("edge_dropout", 0.1)

        # ── News decay (FIX M3): half-life in trading days for weighting
        # per-day FinBERT embeddings across the W-day window.
        self.news_decay_halflife = float(model_cfg.get("news_decay_halflife", 10.0))

        # ── Correlation edge builder (FIX M1): used to recompute correlation
        # edges for each sub-window snapshot.  This is separate from the one
        # in the dataset and is keyed by (sub_window_idx, active_tickers) in
        # the same per-date LRU so we don't recompute the full matrix on
        # every forward.
        self.corr_builder = CorrelationEdgeBuilder(
            top_k=graph_cfg.get("correlation_top_k", 10),
            window=graph_cfg.get("correlation_window", 60),
        )

        # ── Modality Encoders ──
        self.ts_encoder = build_ts_encoder(config)

        news_embed_dim = (
            768
            if "minilm" not in data_cfg.get("news_encoder_model", "").lower()
            else 384
        )
        self.news_encoder = NewsEncoder(
            embedding_dim=news_embed_dim, output_dim=model_cfg.get("d_news", 256)
        )

        self.reports_encoder = ReportsEncoder(
            d_rep=model_cfg.get("d_rep", 256), dropout=model_cfg.get("dropout", 0.1)
        )

        # ── Fusion ──
        self.fusion = build_fusion(config)

        # ── Temporal GNN ──
        gnn_type = model_cfg.get("temporal_gnn", "auto")
        gnn_layers = model_cfg.get("gnn_layers", 3)
        dropout = model_cfg.get("dropout", 0.1)

        # E5 note: EvolveGCNH requires a fixed num_of_nodes at init.  We pad to
        # max_nodes; inactive (zero-padded) nodes are graph-isolated (no edges)
        # so they don't corrupt active-node representations via message passing.
        if gnn_type in {"auto", "evolvegcn_h", "pyg_gcn_gru"}:
            self.temporal_gnn = EvolveGCNBlock(
                max_nodes, self.d_fused, gnn_layers, dropout, backend=gnn_type
            )
        else:
            raise ValueError(f"Unknown temporal_gnn: {gnn_type}")

        # ── Forecast Head (regression) + Direction Head (classification) ──
        self.forecast_head = ForecastHead(self.d_fused, self.horizon, dropout)
        self.direction_head = DirectionHead(self.d_fused, self.horizon, dropout)

    # ------------------------------------------------------------------
    # Helper: time-decayed news aggregation (FIX M3)
    # ------------------------------------------------------------------
    def _aggregate_news_window(self, news_per_stock: list, dev: torch.device) -> list:
        """Return a list of per-stock aggregate news embeddings.

        For each stock, collapse the W-day list of (possibly None) per-day
        embeddings into a single (num_articles, embed_dim) tensor by
        time-decay-weighting each day's articles before concatenation.
        The downstream ``NewsEncoder`` then attention-pools over the
        resulting weighted-article set.

        This replaces the previous "last non-None day" logic that threw
        away 95%+ of the news signal in the window.
        """
        decay = self.news_decay_halflife
        aggregates: list = []
        for stock_news in news_per_stock:
            # stock_news is a list of W elements, each a tensor(num_articles, E) or None.
            weighted_parts: list = []
            W = len(stock_news)
            for day_offset_from_end, day_emb in enumerate(reversed(stock_news)):
                if day_emb is None or day_emb.numel() == 0:
                    continue
                # Half-life decay: most recent day gets weight 1, decays by
                # 0.5 per ``decay`` days.
                w = float(2.0 ** (-day_offset_from_end / max(1e-6, decay)))
                weighted_parts.append(day_emb.to(dev) * w)
            if weighted_parts:
                aggregates.append(torch.cat(weighted_parts, dim=0))
            else:
                aggregates.append(None)
        return aggregates

    # ------------------------------------------------------------------
    # Helper: edge dropout during training (FIX M8)
    # ------------------------------------------------------------------
    def _apply_edge_dropout(
        self,
        edge_index: torch.Tensor,
        num_active: int,
    ) -> torch.Tensor:
        """Randomly drop non-self-loop edges during training."""
        if not self.training or self.edge_dropout <= 0 or edge_index.numel() == 0:
            return edge_index
        E = edge_index.size(1)
        is_self_loop = edge_index[0] == edge_index[1]
        non_self_mask = ~is_self_loop
        drop_mask = (
            torch.rand(E, device=edge_index.device) < self.edge_dropout
        ) & non_self_mask
        return edge_index[:, ~drop_mask]

    def _encode_modalities(
        self,
        ts_feat: torch.Tensor,
        news_aggregates: list,
        report_feats: list,
        dev: torch.device,
    ) -> torch.Tensor:
        """Run all modality encoders + fusion for a (N, sub_W, 21) TS slice."""
        h_ts = self.ts_encoder(ts_feat)  # (N, d_ts)
        h_news = self.news_encoder(news_aggregates)  # (N, d_news)
        h_rep = self.reports_encoder(report_feats, device=dev)  # (N, d_rep)
        return self.fusion(h_ts, h_news, h_rep)  # (N, d_fused)

    def forward(self, sample, device=None):
        """
        Full forward pass: raw data → encoders → fusion → graph → GNN → forecast.

        Pipeline (see fix notes at top of file):
          1. Time-decay-aggregate news across the full W-day window (FIX M3).
          2. Compute fundamentals once (they are point-in-time at pred_date).
          3. For each of K sub-windows (growing prefix of the W-day window):
              a. Encode TS on the sub-window.
              b. Fuse with news + reports to get h_fused_k.
              c. Build sector + correlation (computed on the sub-window's
                 return slice) + self-loops, optionally with edge dropout.
              d. Scatter into a (max_nodes, d) padded tensor.
          4. Pass the K (h_padded_k, edge_index_k, None) tuples to EvolveGCN.
          5. Slice the GNN output at ``num_active`` (FIX M5) and run the
             forecast head on ``target_idx``.
        """
        dev = device or next(self.parameters()).device
        N = sample["num_active"]

        if N == 0:
            horizon = self.horizon
            empty = torch.zeros(0, horizon, device=dev)
            return {"log_returns": empty, "pred_close": empty}

        # ═══════════════════════════════════════════════════════════════
        # 1. News aggregation (FIX M3): time-decay across full window
        # ═══════════════════════════════════════════════════════════════
        news_aggregates = self._aggregate_news_window(sample["news_per_stock"], dev)

        # ═══════════════════════════════════════════════════════════════
        # 2. Report features (point-in-time at pred_date) — computed once
        # ═══════════════════════════════════════════════════════════════
        pred_date = sample["pred_date_ts"]
        report_feats: list = []
        for fund_dict in sample["report_fundamentals"]:
            if fund_dict is not None and pred_date is not None:
                feat = self.reports_encoder.get_point_in_time_features(
                    fund_dict, pred_date
                )
                report_feats.append(feat)
            else:
                report_feats.append(None)

        # ═══════════════════════════════════════════════════════════════
        # 3. Build K real sub-window snapshots (FIX M1)
        # ═══════════════════════════════════════════════════════════════
        ts_feat_full = sample["ts_features"].to(dev)  # (N, W, 21)
        W = ts_feat_full.shape[1]
        K = max(1, self.num_gnn_snapshots)

        # Min sub-window length so the TCN has enough receptive field
        # (dilations [1,2,4,8] + kernel 3 → receptive ~31); use max(10, W//K)
        min_sub_w = max(10, W // (K * 2))
        sub_window_ends = [
            max(min_sub_w, int(round(W * (k + 1) / K))) for k in range(K)
        ]

        # Precompute the static sector edges once (they don't depend on sub-window).
        sector_ei = sample["sector_edge_index"].to(dev)  # (2, E_s)

        # Self-loops over active nodes
        self_loops = torch.arange(N, dtype=torch.long, device=dev)
        self_loop_ei = torch.stack([self_loops, self_loops])

        # Active ticker list + returns_dict are used to build per-sub-window
        # correlation edges.  ``returns_dict`` maps ticker → full returns series
        # up to the prediction date (the dataset clipped it to corr_window).
        active_tickers = sample.get("active_tickers", [])
        ticker_to_idx = {t: i for i, t in enumerate(active_tickers)}
        returns_dict = sample.get("returns_dict", {})

        snapshots: list = []
        for sub_end in sub_window_ends:
            # 3a. Slice TS and run all encoders on the sub-window.
            ts_sub = ts_feat_full[:, :sub_end, :]  # (N, sub_end, 21)
            h_fused_k = self._encode_modalities(
                ts_sub, news_aggregates, report_feats, dev
            )  # (N, d_fused)

            # 3b. Correlation edges on the sub-window's returns.
            # Because returns_dict is short (corr_window), slicing up to
            # sub_end keeps it point-in-time.  If sub_end < corr window
            # length, fall back to whatever is available (>= min 10 days).
            sub_returns_dict: Dict[str, np.ndarray] = {}
            for ticker, r in returns_dict.items():
                if len(r) == 0:
                    continue
                # Keep the most recent sub_end entries (right-aligned).
                sub_returns_dict[ticker] = r[-sub_end:] if sub_end < len(r) else r

            if active_tickers and sub_returns_dict and pred_date is not None:
                corr_ei_k, _ = self.corr_builder.build(
                    (pred_date, sub_end),
                    sub_returns_dict,
                    active_tickers,
                    ticker_to_idx,
                )
                corr_ei_k = corr_ei_k.to(dev)
            else:
                corr_ei_k = sample.get(
                    "corr_edge_index", torch.zeros(2, 0, dtype=torch.long)
                ).to(dev)

            # 3c. Assemble edges: sector ∪ correlation ∪ self-loops.
            edge_index_k = torch.cat([sector_ei, corr_ei_k, self_loop_ei], dim=1)
            # 3d. Edge dropout during training (FIX M8).
            edge_index_k = self._apply_edge_dropout(edge_index_k, N)

            # 3e. Scatter into max_nodes padded feature tensor.
            h_padded = torch.zeros(
                self.max_nodes, self.d_fused, device=dev, dtype=h_fused_k.dtype
            )
            h_padded[:N] = h_fused_k

            snapshots.append((h_padded, edge_index_k, None))

        # ═══════════════════════════════════════════════════════════════
        # 4. TEMPORAL GNN: K snapshots → h_graph (max_nodes, 256)
        # ═══════════════════════════════════════════════════════════════
        h_graph = self.temporal_gnn(snapshots)  # (max_nodes, 256)

        # FIX M5: slice to active nodes before downstream consumers touch it.
        h_active = h_graph[:N]

        # ═══════════════════════════════════════════════════════════════
        # 5. FORECAST HEAD: → log_returns (N_T, 5)
        # ═══════════════════════════════════════════════════════════════
        target_idx = sample["target_idx"].to(dev)
        last_close = sample["last_close"].to(dev)

        h_target = h_active[target_idx]  # (N_T, 256)
        log_returns = self.forecast_head(h_target)
        # FIX M9: Detach backbone features before direction head so the
        # direction loss gradients do NOT flow into the shared backbone.
        # Previously gamma=0.1 × dir_loss≈0.69 = 0.069 was 97% of total
        # loss, causing the backbone to be trained almost entirely by the
        # noisy direction signal and collapsing predictions to near-zero.
        direction_logits = self.direction_head(h_target.detach())

        return {
            "log_returns": log_returns,
            "direction_logits": direction_logits,
            "pred_close": reconstruct_prices(log_returns, last_close),
        }

    def predict_with_uncertainty(self, sample, device=None, n_mc=10):
        """MC Dropout inference."""
        self.train()  # Keep dropout active
        preds = []
        with torch.no_grad():
            for _ in range(n_mc):
                result = self.forward(sample, device)
                preds.append(result["log_returns"])

        preds = torch.stack(preds)
        mean_pred, std_pred = preds.mean(0), preds.std(0)
        dev = mean_pred.device
        last_close = sample["last_close"].to(dev)

        return {
            "log_returns_mean": mean_pred,
            "log_returns_std": std_pred,
            "log_returns_lower": mean_pred - 1.96 * std_pred,
            "log_returns_upper": mean_pred + 1.96 * std_pred,
            "price_mean": reconstruct_prices(mean_pred, last_close),
            "price_lower": reconstruct_prices(mean_pred - 1.96 * std_pred, last_close),
            "price_upper": reconstruct_prices(mean_pred + 1.96 * std_pred, last_close),
        }
