"""
dataset_gnn.py — TemporalGraphDataset for the FinMultiTime dataset.

ARCHITECTURE NOTE:
    The dataset produces RAW inputs for each modality. The model's forward()
    is responsible for running the modality encoders (TS, News, Reports),
    fusion, and graph construction. The dataset does NOT pre-fuse features.

    Each sample contains:
        - ts_features: (N, W, 21) raw TS features per stock per day
        - news_embeddings: list of N dicts, each mapping day_offset → tensor
        - report_features: list of N optional tensors (20-dim fundamentals)
        - edge_index / edge_attr: graph structure per snapshot
        - targets, last_close, etc.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.graph_gnn import GraphBuilder
from src.technical_indicators_gnn import add_technical_indicators
from src.utils_gnn import (
    assign_news_to_trading_day,
    build_master_calendar,
    build_trading_calendar,
    clip_log_returns,
    compute_log_returns,
    expanding_zscore,
    resolve_data_path,
    safe_log1p,
    temporal_train_val_test_split,
)

logger = logging.getLogger(__name__)

INVALID_SECTORS = {"Client Error", "N/A", "", "nan", "None"}


# ===========================================================================
# Raw Data Loaders (unchanged)
# ===========================================================================


def load_time_series(
    data_dir: str, ts_subdir: str = "sp500_time_series"
) -> Dict[str, pd.DataFrame]:
    """Load all time series CSVs. Returns UPPER ticker → DataFrame."""
    _, ts_dir = resolve_data_path(
        data_dir,
        ts_subdir,
        "sp500_time_series",
        kind="directory",
        aliases=["time_series"],
    )
    ticker_dfs = {}

    if not os.path.exists(ts_dir):
        logger.warning(f"Time series directory not found: {ts_dir}")
        return ticker_dfs

    total_non_positive_prices = 0
    total_negative_volumes = 0
    tickers_sanitized = 0

    for fname in sorted(os.listdir(ts_dir)):
        if not fname.endswith(".csv"):
            continue
        ticker = fname.replace(".csv", "").upper()
        fpath = os.path.join(ts_dir, fname)
        try:
            df = pd.read_csv(fpath)
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
            col_map = {}
            for c in df.columns:
                cl = c.strip().lower().replace(" ", "_")
                mapping = {
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
                if cl in mapping:
                    col_map[c] = mapping[cl]
            df = df.rename(columns=col_map)
            required = ["date", "open", "high", "low", "close", "volume"]
            if any(c not in df.columns for c in required):
                continue
            # Keep only date + OHLCV — drop dividends, stock_splits, and any other extras.
            df = df[required].copy()
            df = df.sort_values("date").reset_index(drop=True).dropna(subset=["close"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            sanitized = False
            for c in ["open", "high", "low", "close"]:
                invalid_mask = df[c].notna() & (df[c] <= 0)
                invalid_count = int(invalid_mask.sum())
                if invalid_count:
                    df.loc[invalid_mask, c] = np.nan
                    total_non_positive_prices += invalid_count
                    sanitized = True

            negative_volume_mask = df["volume"].notna() & (df["volume"] < 0)
            negative_volume_count = int(negative_volume_mask.sum())
            if negative_volume_count:
                df.loc[negative_volume_mask, "volume"] = 0
                total_negative_volumes += negative_volume_count
                sanitized = True

            df = df.dropna(subset=["close"]).reset_index(drop=True)
            if sanitized:
                tickers_sanitized += 1
            ticker_dfs[ticker] = df
        except Exception as e:
            logger.warning(f"Failed to load {fpath}: {e}")

    if tickers_sanitized:
        logger.info(
            "Sanitized OHLCV anomalies for %d tickers | non_positive_prices=%d | negative_volumes=%d",
            tickers_sanitized,
            total_non_positive_prices,
            total_negative_volumes,
        )
    logger.info(f"Loaded time series for {len(ticker_dfs)} tickers")
    return ticker_dfs


def load_sectors(
    data_dir: str, filename: str = "sp500stock_data_description.csv"
) -> pd.DataFrame:
    """Load sector mapping. Returns DataFrame with [ticker (UPPER), sector]."""
    _, fpath = resolve_data_path(
        data_dir,
        filename,
        "sp500stock_data_description.csv",
        kind="file",
    )
    if not os.path.exists(fpath):
        return pd.DataFrame(columns=["ticker", "sector"])
    df = pd.read_csv(fpath)
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("stock_name", "ticker", "symbol"):
            col_map[c] = "stock_name"
        elif cl == "sector":
            col_map[c] = "sector"
    df = df.rename(columns=col_map)
    if "stock_name" not in df.columns or "sector" not in df.columns:
        return pd.DataFrame(columns=["ticker", "sector"])
    df = df[~df["sector"].isin(INVALID_SECTORS)].dropna(subset=["sector"])
    df["ticker"] = df["stock_name"].str.strip().str.upper()
    df["sector"] = df["sector"].str.strip()
    result = df[["ticker", "sector"]].drop_duplicates(subset=["ticker"])
    logger.info(f"Loaded sectors for {len(result)} tickers")
    return result


def load_news_embeddings(
    cache_dir: str, tickers: List[str]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load pre-computed news embeddings. Returns ticker → {date_str → tensor}."""
    news_cache = {}
    if not os.path.exists(cache_dir):
        logger.warning(f"News embedding cache directory not found: {cache_dir}")
        return news_cache
    for ticker in tickers:
        cache_file = os.path.join(cache_dir, f"{ticker}_news_embeddings.pt")
        if os.path.exists(cache_file):
            try:
                news_cache[ticker] = torch.load(
                    cache_file, map_location="cpu", weights_only=False
                )
            except Exception:
                logger.warning(
                    f"Failed to load news cache for {ticker} from {cache_file}"
                )
    logger.info(f"Loaded news embeddings for {len(news_cache)}/{len(tickers)} tickers")
    return news_cache


def load_fundamentals(
    data_dir: str,
    table_subdir: str = "sp500_table",
    tickers: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load SEC XBRL fundamentals. Returns UPPER ticker → {filing_date → {metric → value}}."""
    _, table_dir = resolve_data_path(
        data_dir,
        table_subdir,
        "sp500_table",
        kind="directory",
        aliases=["table"],
    )
    all_fundamentals = {}
    if not os.path.exists(table_dir):
        return all_fundamentals

    for dirname in sorted(os.listdir(table_dir)):
        ticker = dirname.upper()
        if tickers is not None and ticker not in tickers:
            continue
        ticker_dir = os.path.join(table_dir, dirname)
        if not os.path.isdir(ticker_dir):
            continue
        all_snapshots = {}
        for json_file in sorted(os.listdir(ticker_dir)):
            if not json_file.endswith(".json"):
                continue
            fpath = os.path.join(ticker_dir, json_file)
            stmt = json_file.replace(".json", "").replace("condensed_consolidated_", "")
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "filings" in data:
                    for filing in data["filings"]:
                        _parse_xbrl_filing(filing, stmt, all_snapshots)
            except Exception:
                pass
        if all_snapshots:
            all_fundamentals[ticker] = all_snapshots
    logger.info(f"Loaded fundamentals for {len(all_fundamentals)} tickers")
    return all_fundamentals


def _parse_xbrl_filing(filing, stmt_prefix, result):
    """Parse a single SEC XBRL filing, keyed by filing_date for point-in-time safety."""
    if not isinstance(filing, dict):
        return
    filing_date = filing.get("filing_date", "")
    if not filing_date:
        return
    gaap = filing.get("facts", {}).get("us-gaap", {})
    if not isinstance(gaap, dict):
        return
    if filing_date not in result:
        result[filing_date] = {}
    for metric_name, metric_data in gaap.items():
        if not isinstance(metric_data, dict):
            continue
        units = metric_data.get("units", {})
        if not isinstance(units, dict):
            continue
        obs_list = units.get("USD", [])
        if not obs_list:
            for uv in units.values():
                if isinstance(uv, list) and uv:
                    obs_list = uv
                    break
        if not obs_list:
            continue
        latest_end, latest_val = "", None
        for obs in obs_list:
            if not isinstance(obs, dict):
                continue
            end, val = obs.get("end", ""), obs.get("val")
            if (
                end
                and val is not None
                and isinstance(val, (int, float))
                and end > latest_end
            ):
                latest_end, latest_val = end, val
        if latest_val is not None:
            result[filing_date][f"{stmt_prefix}_{metric_name}"] = float(latest_val)


# ===========================================================================
# Feature Builder
# ===========================================================================


class FeatureBuilder:
    """
    Builds 21-dim per-stock per-day features from raw OHLCV + technical indicators.
    6 base features + 15 technical indicators.
    """

    NUM_FEATURES = 21

    def __init__(self, clip_sigma=5.0, zscore_min_days=60, volatility_window=20):
        self.clip_sigma = clip_sigma
        self.zscore_min_days = zscore_min_days
        self.volatility_window = volatility_window
        self.global_stats = {}

    def compute_global_stats(self, ticker_dfs, train_dates):
        all_lr, all_lv, all_raw_v = [], [], []
        for ticker, df in ticker_dfs.items():
            train_df = df[df["date"].isin(train_dates)]
            if len(train_df) < 2:
                continue
            lr = compute_log_returns(train_df["close"])
            all_lr.extend(lr.dropna().tolist())
            volume = (
                pd.to_numeric(train_df["volume"], errors="coerce")
                .fillna(0)
                .clip(lower=0)
            )
            all_lv.extend(safe_log1p(volume).tolist())
            all_raw_v.extend(volume.tolist())
        if all_lr:
            self.global_stats = {
                "log_return": {"mean": np.mean(all_lr), "std": np.std(all_lr) + 1e-8},
                "log_volume": {"mean": np.mean(all_lv), "std": np.std(all_lv) + 1e-8},
                "raw_volume": {
                    "mean": np.mean(all_raw_v),
                    "std": np.std(all_raw_v) + 1e-8,
                },
            }
            logger.info(
                "Computed feature normalization stats from %d tickers across %d train dates",
                len(ticker_dfs),
                len(train_dates),
            )
        else:
            logger.warning(
                "Feature normalization stats were empty; feature scaling will rely on defaults"
            )

    def build_features(self, df, master_calendar):
        df_indexed = df.set_index("date")
        aligned = df_indexed.reindex(master_calendar)
        for c in ["open", "high", "low", "close"]:
            price = pd.to_numeric(aligned[c], errors="coerce")
            aligned[c] = price.where(price > 0)
        is_trading = aligned["close"].notna().astype(float)

        # Identify first valid trading day
        if not is_trading.any():
            return pd.DataFrame(), pd.Series(dtype=float)
        first_valid = is_trading.idxmax()

        # Forward-fill AFTER first valid day only
        last_close = aligned["close"].ffill()
        aligned["open"] = aligned["open"].fillna(last_close)
        aligned["high"] = aligned["high"].fillna(last_close)
        aligned["low"] = aligned["low"].fillna(last_close)
        aligned["close"] = last_close
        aligned["volume"] = (
            pd.to_numeric(aligned["volume"], errors="coerce").fillna(0).clip(lower=0)
        )

        close, volume = aligned["close"], aligned["volume"]
        high, low, opn = aligned["high"], aligned["low"], aligned["open"]

        log_return = clip_log_returns(compute_log_returns(close), sigma=self.clip_sigma)
        gs_lv = self.global_stats.get("log_volume", {})
        log_volume_z = expanding_zscore(
            safe_log1p(volume),
            min_periods=self.zscore_min_days,
            global_mean=gs_lv.get("mean"),
            global_std=gs_lv.get("std"),
        )
        volatility = log_return.rolling(self.volatility_window).std()
        intraday_range = (high - low) / (close + 1e-8)
        gs_rv = self.global_stats.get("raw_volume", {})
        volume_zscore = expanding_zscore(
            volume,
            min_periods=self.zscore_min_days,
            global_mean=gs_rv.get("mean"),
            global_std=gs_rv.get("std"),
        )

        tech = add_technical_indicators(
            close=close, high=high, low=low, open_=opn, volume=volume
        )

        features = pd.DataFrame(
            {
                "log_return": log_return,
                "log_volume": log_volume_z,
                "volatility": volatility,
                "intraday_range": intraday_range,
                "volume_zscore": volume_zscore,
                "is_trading": is_trading,
                "rsi_14": tech["rsi_14"],
                "macd": tech["macd"],
                "macd_signal": tech["macd_signal"],
                "macd_hist": tech["macd_hist"],
                "bb_upper": tech["bb_upper"],
                "bb_lower": tech["bb_lower"],
                "atr_14": tech["atr_14"],
                "obv_zscore": tech["obv_zscore"],
                "stoch_k": tech["stoch_k"],
                "stoch_d": tech["stoch_d"],
                "adx_14": tech["adx_14"],
                "cci_20": tech["cci_20"],
                "willr_14": tech["willr_14"],
                "roc_10": tech["roc_10"],
                "mfi_14": tech["mfi_14"],
            },
            index=master_calendar,
        )

        # Mask out dates before stock's first valid trading day
        features.loc[: first_valid - pd.Timedelta(days=1)] = 0.0
        is_trading.loc[: first_valid - pd.Timedelta(days=1)] = 0.0
        features = features.fillna(0)

        return features, is_trading


# ===========================================================================
# Temporal Graph Dataset
# ===========================================================================


class TemporalGraphDataset(Dataset):
    """
    Each sample is a temporal window of W days for ~500 stocks.

    Returns RAW modality inputs — the model runs the encoders.

    __getitem__ returns dict with:
        ts_features:       (N_active, W, 21) raw time series features
        news_embeddings:   list of N_active, each is list of W (may be None per day)
        report_features:   list of N_active Optional[Tensor(20,)] fundamentals
        edge_index_sector: (2, E_s) sector edges
        edge_attr_sector:  (E_s, 4)
        targets:           (N_T, 5) log-return targets
        target_close:      (N_T, 5) future close prices
        last_close:        (N_T,) close at prediction day
        target_idx:        (N_T,) indices into N_active
        active_tickers:    list of ticker strings
        pred_date:         date string
        num_active:        int
        returns_dict:      dict for correlation edge building (ticker → recent returns)
    """

    def __init__(
        self,
        config,
        dates,
        ticker_dfs,
        sectors_df,
        feature_builder,
        news_cache=None,
        fundamentals=None,
        master_calendar=None,
        max_nodes=550,
        mode="train",
    ):
        super().__init__()
        self.config = config
        self.dates = dates
        self.sectors_df = sectors_df
        self.news_cache = news_cache or {}
        self.fundamentals = fundamentals or {}
        self.max_nodes = max_nodes
        self.mode = mode

        data_cfg = config["data"]
        self.window_size = data_cfg.get("window_size", 60)
        self.horizon = data_cfg.get("horizon", 5)
        self.min_history = data_cfg.get("min_history", 252)
        self.num_features = FeatureBuilder.NUM_FEATURES

        self.master_calendar = (
            master_calendar
            if master_calendar is not None
            else build_master_calendar(ticker_dfs)
        )
        self.date_to_idx = {d: i for i, d in enumerate(self.master_calendar)}

        # Pre-compute features for all tickers
        self.ticker_features = {}
        self.ticker_is_trading = {}
        self.ticker_close_prices = {}
        self.ticker_log_returns = {}

        logger.info(f"Pre-computing features for {len(ticker_dfs)} tickers...")
        for ticker, df in ticker_dfs.items():
            features, is_trading = feature_builder.build_features(
                df, self.master_calendar
            )
            if len(features) > 0:
                self.ticker_features[ticker] = features.values  # numpy (T, 21)
                self.ticker_is_trading[ticker] = is_trading.values  # numpy (T,)
                close = df.set_index("date")["close"].reindex(self.master_calendar)
                close = pd.to_numeric(close, errors="coerce")
                close = close.where(close > 0).ffill()
                self.ticker_close_prices[ticker] = close.values  # numpy (T,)
                lr = compute_log_returns(close)
                self.ticker_log_returns[ticker] = lr.values  # numpy (T,)

        # FIX E13: Log per-feature statistics so we can diagnose bad features
        # (NaN, Inf, or degenerate variance) before they silently break training.
        # Only log once, on the train split to avoid noisy repeats.
        if self.mode == "train" and self.ticker_features:
            self._log_feature_stats()

        # Build graph builder for sector edges
        self.graph_builder = GraphBuilder(config, sectors_df)

        self._build_valid_indices()
        logger.info(
            f"TemporalGraphDataset ({self.mode}): {len(self)} samples, "
            f"{len(self.ticker_features)} tickers, {self.num_features} features"
        )

    def _log_feature_stats(self):
        """FIX E13: Per-feature summary statistics for post-normalization diagnostics."""
        feature_names = [
            "log_return",
            "log_volume",
            "volatility",
            "intraday_range",
            "volume_zscore",
            "is_trading",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_lower",
            "atr_14",
            "obv_zscore",
            "stoch_k",
            "stoch_d",
            "adx_14",
            "cci_20",
            "willr_14",
            "roc_10",
            "mfi_14",
        ]
        stacked = np.concatenate(
            list(self.ticker_features.values()), axis=0
        )  # (T*tickers, 21)
        logger.info("=" * 70)
        logger.info("POST-NORMALIZATION FEATURE STATS (train)")
        logger.info("=" * 70)
        logger.info(
            "  %-18s %10s %10s %10s %10s %10s",
            "feature",
            "mean",
            "std",
            "min",
            "max",
            "nan_rate",
        )
        for i, name in enumerate(feature_names[: stacked.shape[1]]):
            col = stacked[:, i]
            finite = np.isfinite(col)
            if finite.sum() == 0:
                logger.warning("  %-18s  ALL NON-FINITE — feature is broken", name)
                continue
            vals = col[finite]
            nan_rate = 1.0 - finite.mean()
            logger.info(
                "  %-18s %10.4f %10.4f %10.4f %10.4f %9.2f%%",
                name,
                float(vals.mean()),
                float(vals.std()),
                float(vals.min()),
                float(vals.max()),
                100.0 * nan_rate,
            )
            if not np.isfinite(vals).all():
                logger.warning("  %s contains non-finite values after masking", name)
            if vals.std() < 1e-8:
                logger.warning(
                    "  %s has near-zero variance (std=%.2e) — degenerate",
                    name,
                    vals.std(),
                )
        logger.info("=" * 70)

    def _build_valid_indices(self):
        self.valid_indices = []
        for date in self.dates:
            idx = self.date_to_idx.get(date)
            if (
                idx is None
                or idx < self.window_size
                or idx + self.horizon >= len(self.master_calendar)
            ):
                continue
            self.valid_indices.append(idx)
        logger.info(
            "Dataset %s valid windows: %d/%d candidate dates",
            self.mode,
            len(self.valid_indices),
            len(self.dates),
        )

    def _get_active_tickers(self, master_idx):
        active = []
        for ticker, is_trd in self.ticker_is_trading.items():
            # Count actual trading days up to master_idx
            n_trading = is_trd[: master_idx + 1].sum()
            if n_trading >= self.min_history:
                active.append(ticker)
        return sorted(active)[: self.max_nodes]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        master_idx = self.valid_indices[idx]
        pred_date = self.master_calendar[master_idx]
        active_tickers = self._get_active_tickers(master_idx)
        N = len(active_tickers)

        if N == 0:
            return self._empty_sample()

        ticker_to_local = {t: i for i, t in enumerate(active_tickers)}
        W = self.window_size
        window_start = master_idx - W

        # ── TS features: (N, W, 21) ──
        ts_features = np.zeros((N, W, self.num_features), dtype=np.float32)
        for i, ticker in enumerate(active_tickers):
            feat = self.ticker_features.get(ticker)
            if feat is not None:
                ts_features[i] = feat[window_start:master_idx]  # (W, 21)

        # ── News embeddings: list of N, each is dict {day_offset → tensor} ──
        # We pass the dates so the model can look up per-day embeddings
        window_dates = [
            str(self.master_calendar[t].date()) for t in range(window_start, master_idx)
        ]
        news_per_stock = []
        for ticker in active_tickers:
            ticker_news = self.news_cache.get(ticker, {})
            # Collect embeddings for each day in the window
            day_embeddings = []
            for date_str in window_dates:
                emb = ticker_news.get(
                    date_str
                )  # tensor(num_articles, embed_dim) or None
                day_embeddings.append(emb)
            news_per_stock.append(day_embeddings)

        # ── Report features: list of N Optional[Tensor(20,)] ──
        # Uses the most recent filing before pred_date (point-in-time safe)
        # Computed at training time by the model using ReportsEncoder.get_point_in_time_features
        report_features_raw = []
        for ticker in active_tickers:
            report_features_raw.append(self.fundamentals.get(ticker))

        # ── Graph edges (sector edges are static per active set) ──
        sector_ei, sector_ea = self.graph_builder.sector_builder.build(
            active_tickers, ticker_to_local
        )

        # ── FIX E3: Build correlation edges and include them in the sample ──
        # Previously, returns_dict was computed but correlation edges were never
        # built here — they were dead code.  Now we build them so the model can
        # merge them with sector edges for a richer graph.
        returns_dict = {}
        corr_window = self.config["graph"].get("correlation_window", 60)
        for ticker in active_tickers:
            lr = self.ticker_log_returns.get(ticker)
            if lr is not None:
                # Get returns up to but not including pred_date
                r = lr[max(0, master_idx - corr_window) : master_idx]
                r = np.nan_to_num(r, nan=0.0)
                returns_dict[ticker] = r

        corr_ei, corr_ea = self.graph_builder.corr_builder.build(
            pred_date,
            returns_dict,
            active_tickers,
            ticker_to_local,
        )

        # ── Targets ──
        target_lr, target_close_list, target_idx, target_tickers, last_close_list = (
            [],
            [],
            [],
            [],
            [],
        )
        for i, ticker in enumerate(active_tickers):
            lr = self.ticker_log_returns.get(ticker)
            close = self.ticker_close_prices.get(ticker)
            if lr is None or close is None:
                continue
            future_returns = []
            valid = True
            for k in range(1, self.horizon + 1):
                fi = master_idx + k
                if fi >= len(lr) or np.isnan(lr[fi]):
                    valid = False
                    break
                future_returns.append(lr[fi])
            if valid and len(future_returns) == self.horizon:
                target_lr.append(future_returns)
                target_close_list.append(
                    [close[master_idx + k] for k in range(1, self.horizon + 1)]
                )
                last_close_list.append(close[master_idx])
                target_idx.append(i)
                target_tickers.append(ticker)

        if not target_lr:
            return self._empty_sample()

        return {
            "ts_features": torch.tensor(ts_features, dtype=torch.float),  # (N, W, 21)
            "news_per_stock": news_per_stock,  # list of N, each is list of W (tensor or None)
            "report_fundamentals": report_features_raw,  # list of N (dict or None)
            "sector_edge_index": sector_ei,  # (2, E_s)
            "sector_edge_attr": sector_ea,  # (E_s, 4)
            "corr_edge_index": corr_ei,  # (2, E_c)  ← FIX E3
            "corr_edge_attr": corr_ea,  # (E_c, 4)  ← FIX E3
            "returns_dict": returns_dict,  # kept for compatibility
            "targets": torch.tensor(target_lr, dtype=torch.float),
            "target_close": torch.tensor(target_close_list, dtype=torch.float),
            "last_close": torch.tensor(last_close_list, dtype=torch.float),
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
            "target_tickers": target_tickers,
            "active_tickers": active_tickers,
            "pred_date": str(pred_date.date()),
            "pred_date_ts": pred_date,
            "num_active": N,
        }

    def _empty_sample(self):
        return {
            "ts_features": torch.zeros(0, self.window_size, self.num_features),
            "news_per_stock": [],
            "report_fundamentals": [],
            "sector_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "sector_edge_attr": torch.zeros(0, 4),
            "corr_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "corr_edge_attr": torch.zeros(0, 4),
            "returns_dict": {},
            "targets": torch.zeros(0, self.horizon),
            "target_close": torch.zeros(0, self.horizon),
            "last_close": torch.zeros(0),
            "target_idx": torch.zeros(0, dtype=torch.long),
            "target_tickers": [],
            "active_tickers": [],
            "pred_date": "",
            "pred_date_ts": None,
            "num_active": 0,
        }


def temporal_collate_fn(batch):
    valid = [b for b in batch if b["num_active"] > 0]
    if not valid:
        return batch[0]
    return valid[0] if len(valid) == 1 else valid


def build_dataloaders(config):
    data_cfg, split_cfg = config["data"], config["split"]
    data_dir = data_cfg["data_dir"]
    logger.info("Building dataloaders from %s", os.path.abspath(data_dir))

    ticker_dfs = load_time_series(
        data_dir, data_cfg.get("time_series_dir", "sp500_time_series")
    )
    sectors_df = load_sectors(
        data_dir, data_cfg.get("description_file", "sp500stock_data_description.csv")
    )
    if not ticker_dfs:
        raise ValueError("No time series data found")

    max_tickers = data_cfg.get("max_tickers")
    total_available = len(ticker_dfs)
    if max_tickers is not None and max_tickers < len(ticker_dfs):
        # Sort by history length (descending) so we keep the most data-rich tickers
        sorted_tickers = sorted(
            ticker_dfs, key=lambda t: len(ticker_dfs[t]), reverse=True
        )
        kept = sorted_tickers[:max_tickers]
        dropped = sorted_tickers[max_tickers:]
        ticker_dfs = {t: ticker_dfs[t] for t in kept}
        logger.info(
            "Ticker selection | kept=%d / %d available (max_tickers=%d) | "
            "min_history_kept=%d days | max_history_kept=%d days",
            len(kept),
            total_available,
            max_tickers,
            min(len(ticker_dfs[t]) for t in kept),
            max(len(ticker_dfs[t]) for t in kept),
        )
        logger.info("Selected tickers: %s", ", ".join(sorted(kept)))
        if dropped:
            logger.debug(
                "Dropped tickers (%d): %s",
                len(dropped),
                ", ".join(sorted(dropped[:20])) + ("..." if len(dropped) > 20 else ""),
            )

    fundamentals = load_fundamentals(
        data_dir, data_cfg.get("table_dir", "sp500_table"), list(ticker_dfs.keys())
    )
    master_calendar = build_master_calendar(ticker_dfs)
    train_dates, val_dates, test_dates = temporal_train_val_test_split(
        master_calendar,
        split_cfg.get("val_days", 45),
        split_cfg.get("test_days", 45),
        split_cfg.get("purge_days", 5),
    )

    feature_builder = FeatureBuilder(
        data_cfg.get("log_return_clip_sigma", 5.0), data_cfg.get("zscore_min_days", 60)
    )
    feature_builder.compute_global_stats(ticker_dfs, train_dates)
    _, news_cache_dir = resolve_data_path(
        data_dir,
        data_cfg.get("news_cache_dir"),
        os.path.join("cache", "news_embeddings"),
        kind="directory",
    )
    news_cache = load_news_embeddings(news_cache_dir, list(ticker_dfs.keys()))
    # ── Comprehensive pre-training data summary ──
    logger.info("=" * 70)
    logger.info("DATA SUMMARY")
    logger.info("=" * 70)
    logger.info(
        "Tickers: %d selected (of %d available) | Master calendar: %d trading days",
        len(ticker_dfs),
        total_available,
        len(master_calendar),
    )
    if len(master_calendar) >= 2:
        logger.info(
            "Date range: %s → %s",
            master_calendar[0].strftime("%Y-%m-%d"),
            master_calendar[-1].strftime("%Y-%m-%d"),
        )
    logger.info(
        "Split | train=%d days (%s → %s) | val=%d days (%s → %s) | test=%d days (%s → %s)",
        len(train_dates),
        train_dates[0].strftime("%Y-%m-%d") if len(train_dates) else "?",
        train_dates[-1].strftime("%Y-%m-%d") if len(train_dates) else "?",
        len(val_dates),
        val_dates[0].strftime("%Y-%m-%d") if len(val_dates) else "?",
        val_dates[-1].strftime("%Y-%m-%d") if len(val_dates) else "?",
        len(test_dates),
        test_dates[0].strftime("%Y-%m-%d") if len(test_dates) else "?",
        test_dates[-1].strftime("%Y-%m-%d") if len(test_dates) else "?",
    )
    # Sector breakdown for selected tickers
    selected_set = set(ticker_dfs.keys())
    matched_sectors = sectors_df[sectors_df["ticker"].isin(selected_set)]
    if len(matched_sectors) > 0:
        sector_counts = matched_sectors["sector"].value_counts()
        logger.info(
            "Sector breakdown (%d tickers with sector info):", len(matched_sectors)
        )
        for sector, count in sector_counts.items():
            logger.info("  %-30s  %3d tickers", sector, count)
        no_sector = selected_set - set(matched_sectors["ticker"])
        if no_sector:
            logger.info("  %-30s  %3d tickers", "(no sector info)", len(no_sector))
    logger.info(
        "Auxiliary data | fundamentals=%d tickers | news_cache=%d tickers | news_cache_dir=%s",
        len(fundamentals),
        len(news_cache),
        os.path.abspath(news_cache_dir),
    )
    logger.info("=" * 70)

    common = dict(
        config=config,
        ticker_dfs=ticker_dfs,
        sectors_df=sectors_df,
        feature_builder=feature_builder,
        news_cache=news_cache,
        fundamentals=fundamentals,
        master_calendar=master_calendar,
        max_nodes=550,
    )

    train_ds = TemporalGraphDataset(dates=train_dates, mode="train", **common)
    val_ds = TemporalGraphDataset(dates=val_dates, mode="val", **common)
    test_ds = TemporalGraphDataset(dates=test_dates, mode="test", **common)

    kw = dict(batch_size=1, collate_fn=temporal_collate_fn, num_workers=0)
    logger.info(
        "Dataloader summary | train=%d | val=%d | test=%d | batch_size=%d | num_workers=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        kw["batch_size"],
        kw["num_workers"],
    )
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        {
            "num_tickers": len(ticker_dfs),
            "max_nodes": 550,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": len(test_ds),
            "sectors_df": sectors_df,
            "fundamentals": fundamentals,
            "num_features": FeatureBuilder.NUM_FEATURES,
        },
    )
