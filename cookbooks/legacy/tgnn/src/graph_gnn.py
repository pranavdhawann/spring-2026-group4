"""
graph_gnn.py — Graph construction for the Temporal GNN.

Uses sp500stock_data_description.csv as sector source.
Filters out invalid sectors ("Client Error", "N/A").

Edge types:
    - Sector edges: connect stocks in the same sector
    - Correlation edges: top-K by rolling return correlation
    - Self-loops: always present, never dropped
"""

import logging
import os
from typing import (  # Tuple used in CorrelationEdgeBuilder type hints
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

INVALID_SECTORS = {"Client Error", "N/A", "", "nan", "None"}


# ===========================================================================
# Sector Edge Builder
# ===========================================================================


class SectorEdgeBuilder:
    """Build sector edges from sp500stock_data_description.csv sectors."""

    def __init__(self, sectors_df: pd.DataFrame):
        """
        Args:
            sectors_df: DataFrame with columns ['ticker', 'sector']
                        ticker is UPPERCASE.
        """
        self.sectors_df = sectors_df.copy()
        self._cache = {}

    def build(
        self,
        active_tickers: List[str],
        ticker_to_idx: Dict[str, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build sector edges for active tickers."""
        cache_key = tuple(sorted(active_tickers))
        if cache_key in self._cache:
            return self._cache[cache_key]

        active_set = set(active_tickers)

        # Filter to active tickers with valid sectors
        sectors = self.sectors_df[
            (self.sectors_df["ticker"].isin(active_set))
            & (~self.sectors_df["sector"].isin(INVALID_SECTORS))
        ]

        sector_groups = sectors.groupby("sector")["ticker"].apply(list).to_dict()

        src_list, dst_list = [], []

        for sector, tickers in sector_groups.items():
            indices = [ticker_to_idx[t] for t in tickers if t in ticker_to_idx]
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    src_list.extend([indices[i], indices[j]])
                    dst_list.extend([indices[j], indices[i]])

        if len(src_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            n_edges = edge_index.size(1)
            edge_attr = torch.zeros(n_edges, 4)
            edge_attr[:, 0] = 1.0  # same_sector
            edge_attr[:, 2] = 1.0  # is_sector_edge

        self._cache[cache_key] = (edge_index, edge_attr)
        return edge_index, edge_attr

    def clear_cache(self):
        self._cache.clear()


# ===========================================================================
# Correlation Edge Builder
# ===========================================================================


class CorrelationEdgeBuilder:
    """Top-K correlation edges with proper per-date caching.

    FIX (critical): The previous caching logic had two fatal bugs:

      1. **In-memory reuse ignored shuffled training order.**  The cache
         compared ``(date - self._last_computed_date).days`` to decide
         whether to reuse edges.  But the DataLoader shuffles dates, so
         after computing edges for 2024-10-30, a date like 2001-02-23
         (centuries earlier) would satisfy ``abs(days) > recompute_days``
         — yet the code tested an *unsigned* difference, and more
         importantly, the reuse check assumed dates arrive in ascending
         order.  In practice, the very first recompute sets
         ``_last_computed_date`` to a late training-split date, and
         almost every subsequent shuffled date is *before* it, yielding
         a negative ``timedelta.days`` that never triggers recompute.
         Result: 176 533 of 204 500 log lines were "Reusing in-memory
         correlation edges from 2024-10-30 for <random date>".

      2. **Disk cache keyed on date alone, not on the returns window.**
         Two calls with the same date but different ``returns_dict``
         (e.g. different active-ticker sets) would silently load stale
         edges from disk.

    The fix:
      - Remove the fragile in-memory single-entry cache entirely.
      - Replace disk caching with a lightweight in-memory LRU so
        identical (date, active-ticker-set) pairs within the same
        epoch are served instantly, but different dates always get
        freshly computed edges.
      - Each LRU key is (date, frozenset(active_tickers)).
    """

    def __init__(self, top_k=10, window=60, recompute_days=None, cache_dir=None):
        self.top_k = top_k
        self.window = window
        # recompute_days and cache_dir are accepted for config-compat but
        # no longer used — every unique (date, ticker-set) is computed fresh.
        self._lru: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._lru_max = 256  # keep at most 256 entries (~val/test size)
        logger.info(
            "CorrelationEdgeBuilder | top_k=%d | window=%d | lru_max=%d",
            top_k,
            window,
            self._lru_max,
        )

    # ------------------------------------------------------------------
    def _compute_correlation_matrix(self, returns_dict, active_tickers):
        """Compute pairwise correlation matrix across active tickers.

        FIX (sub-window bug): Previously required every ticker to have at
        least ``self.window`` return observations.  For sub-window snapshots
        (e.g. sub_end=15 with window=60) this caused ALL tickers to be
        marked invalid → 0 correlation edges for 3 of 4 snapshots.

        Fix:
          - Accept any ticker with >= min_required returns (at least 10 or
            window//4, whichever is smaller).
          - Build the correlation matrix using however many returns are
            available (up to self.window), dynamically sizing the matrix.
        """
        N = len(active_tickers)
        # Determine minimum required entries and actual window to use.
        # min_required: at least 10 (need some history) but never more than
        # self.window (otherwise we'd still require the full window).
        min_required = max(10, self.window // 4)
        min_required = min(min_required, self.window)  # can't require more than window

        # Gather available return lengths so we can size the matrix.
        available_lengths = []
        for ticker in active_tickers:
            if ticker in returns_dict:
                available_lengths.append(len(returns_dict[ticker]))
            else:
                available_lengths.append(0)

        # Actual window = min(self.window, max available length among valid tickers)
        valid_lengths = [l for l in available_lengths if l >= min_required]
        if not valid_lengths:
            # No ticker has enough history — return zero matrix.
            return np.zeros((N, N))
        actual_window = min(self.window, max(valid_lengths))
        actual_window = max(actual_window, min_required)

        returns_matrix = np.zeros((N, actual_window))
        valid_mask = np.ones(N, dtype=bool)

        for i, ticker in enumerate(active_tickers):
            avail = available_lengths[i]
            if avail >= min_required:
                r = returns_dict[ticker][-actual_window:]
                returns_matrix[i, : len(r)] = r
            else:
                valid_mask[i] = False

        with np.errstate(invalid="ignore"):
            corr_matrix = np.corrcoef(returns_matrix)
        # np.corrcoef returns a scalar when N=1; force to 2D
        corr_matrix = np.atleast_2d(np.nan_to_num(corr_matrix, nan=0.0))
        corr_matrix[~valid_mask, :] = 0.0
        corr_matrix[:, ~valid_mask] = 0.0
        return corr_matrix

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_cache_key(date) -> str:
        """Format a cache key for logging.  Accepts pd.Timestamp or a tuple
        (pd.Timestamp, sub_window_end) used by the model-side sub-window
        correlation builder (FIX M1).
        """
        if isinstance(date, tuple):
            ts, suffix = date[0], date[1:]
            ts_str = ts.date().isoformat() if hasattr(ts, "date") else str(ts)
            return f"{ts_str}@{suffix}"
        return date.date().isoformat() if hasattr(date, "date") else str(date)

    def build(
        self, date, returns_dict, active_tickers, ticker_to_idx, force_recompute=False
    ):
        # --- LRU lookup (keyed on date + ticker set) ---
        ticker_key = tuple(sorted(active_tickers))
        cache_key = (date, ticker_key)
        if not force_recompute and cache_key in self._lru:
            return self._lru[cache_key]

        # --- Compute fresh correlation edges for this date ---
        corr_matrix = self._compute_correlation_matrix(returns_dict, active_tickers)
        N = len(active_tickers)
        src_list, dst_list, corr_list = [], [], []

        for i in range(N):
            corr_row = corr_matrix[i].copy()
            corr_row[
                i
            ] = 0.0  # FIX: was -np.inf, which abs() turns to +inf and always lands in top-K
            top_k_indices = np.argsort(np.abs(corr_row))[-self.top_k :]
            for j in top_k_indices:
                if i != j and np.abs(corr_row[j]) > 1e-6:
                    src_list.append(i)
                    dst_list.append(j)
                    corr_list.append(corr_row[j])

        if len(src_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            n_edges = len(src_list)
            edge_attr = torch.zeros(n_edges, 4)
            edge_attr[:, 1] = torch.tensor(corr_list, dtype=torch.float)
            edge_attr[:, 3] = 1.0

        # --- Store in LRU (evict oldest if over capacity) ---
        if len(self._lru) >= self._lru_max:
            # evict first-inserted key (roughly FIFO)
            oldest = next(iter(self._lru))
            del self._lru[oldest]
        self._lru[cache_key] = (edge_index, edge_attr)

        logger.debug(
            "Computed correlation edges for %s | active_tickers=%d | edges=%d",
            self._fmt_cache_key(date),
            len(active_tickers),
            edge_index.size(1),
        )
        return edge_index, edge_attr

    def clear_cache(self):
        """Clear the LRU edge cache (e.g. between seed runs)."""
        self._lru.clear()


# ===========================================================================
# Graph Snapshot Builder
# ===========================================================================


class GraphBuilder:
    """Builds complete graph snapshots combining all edge types."""

    def __init__(self, config: dict, sectors_df: pd.DataFrame):
        graph_cfg = config["graph"]
        self.sector_builder = SectorEdgeBuilder(sectors_df)
        self.corr_builder = CorrelationEdgeBuilder(
            top_k=graph_cfg.get("correlation_top_k", 10),
            window=graph_cfg.get("correlation_window", 60),
        )
        self.edge_dropout = graph_cfg.get("edge_dropout", 0.1)
        self.protect_self_loops = graph_cfg.get("protect_self_loops", True)

    def build_snapshot(
        self,
        date,
        active_tickers,
        ticker_to_idx,
        node_features,
        returns_dict,
        training=False,
    ):
        N = len(active_tickers)

        sector_ei, sector_ea = self.sector_builder.build(active_tickers, ticker_to_idx)
        corr_ei, corr_ea = self.corr_builder.build(
            date, returns_dict, active_tickers, ticker_to_idx
        )

        self_loops = torch.arange(N, dtype=torch.long)
        self_loop_ei = torch.stack([self_loops, self_loops], dim=0)
        self_loop_ea = torch.zeros(N, 4)

        edge_index = torch.cat([sector_ei, corr_ei, self_loop_ei], dim=1)
        edge_attr = torch.cat([sector_ea, corr_ea, self_loop_ea], dim=0)

        if training and self.edge_dropout > 0:
            edge_index, edge_attr = self._apply_edge_dropout(edge_index, edge_attr, N)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
        data.ticker_ids = torch.tensor(
            [ticker_to_idx[t] for t in active_tickers], dtype=torch.long
        )
        data.num_nodes = N
        return data

    def _apply_edge_dropout(self, edge_index, edge_attr, num_nodes):
        E = edge_index.size(1)
        is_self_loop = edge_index[0] == edge_index[1]
        keep_mask = torch.ones(E, dtype=torch.bool)
        non_self_mask = ~is_self_loop
        drop_mask = (torch.rand(E) < self.edge_dropout) & non_self_mask
        keep_mask[drop_mask] = False

        kept_edge_index = edge_index[:, keep_mask]
        for node_id in range(num_nodes):
            node_non_self = (
                (kept_edge_index[0] == node_id) | (kept_edge_index[1] == node_id)
            ) & (kept_edge_index[0] != kept_edge_index[1])
            if not node_non_self.any():
                node_edges = (
                    (edge_index[0] == node_id) | (edge_index[1] == node_id)
                ) & non_self_mask
                keep_mask[node_edges] = True

        return edge_index[:, keep_mask], edge_attr[keep_mask]
