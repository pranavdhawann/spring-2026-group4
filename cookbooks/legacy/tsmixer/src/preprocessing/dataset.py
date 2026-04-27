"""Windowed dataset for TSMixer. Targets = future log returns (scaled)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import FEATURE_COLS, build_features

TARGET_IDX = FEATURE_COLS.index("log_return")


class WindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        lookback: int,
        horizon: int,
        target_scale: float = 1.0,
    ):
        self.X = features.astype(np.float32)
        self.L = lookback
        self.H = horizon
        self.scale = target_scale
        self.n = len(self.X) - lookback - horizon + 1
        if self.n <= 0:
            raise ValueError(
                f"Not enough rows: {len(self.X)} for L={lookback} H={horizon}"
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[i : i + self.L]
        y = self.X[i + self.L : i + self.L + self.H, TARGET_IDX] * self.scale
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


def load_asset(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    feats = build_features(df)

    raw = df.copy()
    raw["Date"] = pd.to_datetime(raw["Date"], utc=True).dt.tz_convert(None)
    raw = raw.sort_values("Date").set_index("Date")
    close = raw["Close"].replace(0, np.nan).reindex(feats.index)
    valid = close.notna()
    feats = feats.loc[valid]
    close = close.loc[valid]
    return feats, close.values.astype(np.float32)


def load_all(data_dir: Path) -> List[Tuple[str, pd.DataFrame, np.ndarray]]:
    out = []
    for p in sorted(Path(data_dir).glob("*.csv")):
        feats, close = load_asset(p)
        if len(feats) >= 300:
            out.append((p.stem, feats, close))
    return out


def split_frames(feats: pd.DataFrame, train_frac: float, val_frac: float):
    n = len(feats)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return feats.iloc[:i_train], feats.iloc[i_train:i_val], feats.iloc[i_val:]
