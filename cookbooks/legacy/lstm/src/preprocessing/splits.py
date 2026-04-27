"""Chronological splits, train-only scaling, and sliding-window construction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import TARGET_COL


@dataclass
class Splits:
    X_train: np.ndarray; y_train: np.ndarray
    X_val: np.ndarray;   y_val: np.ndarray
    X_test: np.ndarray;  y_test: np.ndarray
    feat_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_names: list[str]


def make_windows(
    features: np.ndarray, target: np.ndarray, lookback: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each valid t, X = features[t-lookback:t], y = target[t:t+horizon]."""
    n = len(features)
    last = n - horizon
    if last - lookback <= 0:
        raise ValueError("not enough rows for the requested lookback/horizon")
    xs = np.stack([features[i - lookback : i] for i in range(lookback, last)])
    ys = np.stack([target[i : i + horizon] for i in range(lookback, last)])
    return xs.astype(np.float32), ys.astype(np.float32)


def prepare_splits(
    feat_df: pd.DataFrame,
    lookback: int,
    horizon: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Splits:
    n = len(feat_df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))

    train_df = feat_df.iloc[:i_train]
    val_df   = feat_df.iloc[i_train:i_val]
    test_df  = feat_df.iloc[i_val:]

    feat_scaler = StandardScaler().fit(train_df.values)
    target_scaler = StandardScaler().fit(train_df[[TARGET_COL]].values)

    def _scale(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feats = feat_scaler.transform(df.values)
        tgt = target_scaler.transform(df[[TARGET_COL]].values).ravel()
        return feats, tgt

    tr_f, tr_t = _scale(train_df)
    va_f, va_t = _scale(val_df)
    te_f, te_t = _scale(test_df)

    X_tr, y_tr = make_windows(tr_f, tr_t, lookback, horizon)
    X_va, y_va = make_windows(va_f, va_t, lookback, horizon)
    X_te, y_te = make_windows(te_f, te_t, lookback, horizon)

    return Splits(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        feat_scaler, target_scaler, list(feat_df.columns),
    )
