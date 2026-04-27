"""Pooled cross-asset windowed dataset. Split each asset chronologically, then pool."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset

from .dataset import TARGET_IDX
from .features import FEATURE_COLS


@dataclass
class TargetScalerStats:
    center: float
    scale: float


@dataclass
class AssetWindows:
    ticker: str
    ticker_id: int
    X: np.ndarray  # [N, C] float32, already chronologically sliced for this split
    close: np.ndarray  # [N] float32, aligned raw close for each feature row


class PooledWindowDataset(Dataset):
    """Concatenates windows across assets. Target = r_{t+1..t+H} * scale."""

    def __init__(self, assets: List[AssetWindows], lookback: int, horizon: int, target_scale: float):
        self.L = lookback
        self.H = horizon
        self.scale = target_scale
        self.arrays = []
        self.closes = []
        self.ticker_ids = []
        self.offsets = []  # (array_idx, start_row)
        for i, a in enumerate(assets):
            n = len(a.X) - lookback - horizon + 1
            if n <= 0:
                continue
            self.arrays.append(a.X.astype(np.float32))
            self.closes.append(a.close.astype(np.float32))
            self.ticker_ids.append(a.ticker_id)
            for s in range(n):
                self.offsets.append((i, s))
        if not self.offsets:
            raise ValueError("No usable windows across any asset for this split.")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        i, s = self.offsets[idx]
        arr = self.arrays[i]
        close = self.closes[i]
        x = arr[s : s + self.L]
        y = arr[s + self.L : s + self.L + self.H, TARGET_IDX] * self.scale
        ticker_id = self.ticker_ids[i]
        anchor_close = np.float32(close[s + self.L - 1])
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)), ticker_id, torch.tensor(anchor_close)


def build_splits(
    assets, train_frac: float, val_frac: float
) -> Tuple[
    List[AssetWindows],
    List[AssetWindows],
    List[AssetWindows],
    Dict[str, int],
    Dict[str, TargetScalerStats],
]:
    """assets: list of (ticker, features_df[, close]). Returns per-split lists of AssetWindows."""
    train_a, val_a, test_a = [], [], []
    ticker_to_id = {asset[0]: i for i, asset in enumerate(assets)}
    target_scalers: Dict[str, TargetScalerStats] = {}

    for asset in assets:
        if len(asset) == 3:
            ticker, feats, close = asset
        else:
            ticker, feats = asset
            close = np.ones(len(feats), dtype=np.float32)
        x = feats[FEATURE_COLS].values.astype(np.float32)
        close = np.asarray(close, dtype=np.float32)
        if len(close) != len(x):
            raise ValueError(f"Close alignment mismatch for {ticker}: close={len(close)} features={len(x)}")
        n = len(x)
        i_tr = int(n * train_frac)
        i_va = int(n * (train_frac + val_frac))
        if i_tr <= 0:
            continue

        scaler = RobustScaler()
        tr_target = x[:i_tr, TARGET_IDX].reshape(-1, 1)
        scaler.fit(tr_target)
        center = float(scaler.center_[0]) if hasattr(scaler, "center_") else float(np.median(tr_target))
        raw_scale = float(scaler.scale_[0]) if hasattr(scaler, "scale_") else float(np.percentile(tr_target, 75) - np.percentile(tr_target, 25))
        scale = raw_scale if abs(raw_scale) > 1e-8 else 1.0
        target_scalers[ticker] = TargetScalerStats(center=center, scale=scale)

        x_scaled = x.copy()
        x_scaled[:, TARGET_IDX] = ((x_scaled[:, TARGET_IDX] - center) / scale).astype(np.float32)
        tid = ticker_to_id[ticker]

        if i_tr > 0:
            train_a.append(AssetWindows(ticker=ticker, ticker_id=tid, X=x_scaled[:i_tr], close=close[:i_tr]))
        if i_va > i_tr:
            val_a.append(AssetWindows(ticker=ticker, ticker_id=tid, X=x_scaled[i_tr:i_va], close=close[i_tr:i_va]))
        if n > i_va:
            test_a.append(AssetWindows(ticker=ticker, ticker_id=tid, X=x_scaled[i_va:], close=close[i_va:]))
    return train_a, val_a, test_a, ticker_to_id, target_scalers
