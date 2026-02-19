
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.dataLoader.dataLoaderBaseline import getTrainTestDataLoader
from src.utils import read_json_file, read_yaml, set_seed



TS_COLS = ["open", "high", "low", "close", "volume", "dividends", "stock splits"]

TABLE_COLS = [
    "us-gaap_Assets",
    "us-gaap_AssetsCurrent",
    "us-gaap_Liabilities",
    "us-gaap_LiabilitiesCurrent",
    "us-gaap_StockholdersEquity",
    "us-gaap_NetCashProvidedByUsedInOperatingActivities",
    "us-gaap_RetainedEarningsAccumulatedDeficit",
]
def _flatten_articles(sample: Dict) -> List[str]:
    texts = []
    for day_articles in sample.get("articles") or []:
        if not isinstance(day_articles, list):
            continue
        for a in day_articles:
            if isinstance(a, str) and a.strip():
                texts.append(a.strip())
    return texts


def _text_encoder_placeholder(texts: List[str], dim: int = 64) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not texts:
        return vec
    lengths = [len(t.split()) for t in texts[:100]]
    vec[0] = len(texts)
    vec[1] = np.mean(lengths) if lengths else 0
    vec[2] = np.std(lengths) if len(lengths) > 1 else 0
    return vec


def _time_series_features(ts_list: List[Dict]) -> np.ndarray:
    if not ts_list:
        return np.zeros(len(TS_COLS) * 3, dtype=np.float32)

    values = {col: [] for col in TS_COLS}
    for day in ts_list:
        for col in TS_COLS:
            v = day.get(col) if isinstance(day, dict) else None
            if v is not None and isinstance(v, (int, float)):
                values[col].append(float(v))

    feats = []
    for col in TS_COLS:
        arr = np.array(values[col]) if values[col] else np.array([0.0])
        feats.extend([np.min(arr), np.mean(arr), np.max(arr)])
    return np.array(feats, dtype=np.float32)


def _table_features(table_list: List[Dict]) -> np.ndarray:
    feats = []
    if not table_list:
        return np.zeros(len(TABLE_COLS), dtype=np.float32)
    last = table_list[-1]
    for col in TABLE_COLS:
        v = last.get(col)
        if v is None:
            feats.append(0.0)
        elif isinstance(v, (int, float)):
            feats.append(float(v))
        else:
            feats.append(0.0)
    return np.array(feats, dtype=np.float32)


def _sample_to_vector(
    sample: Dict,
    text_encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
    text_dim: int = 64,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    if text_encoder is not None:
        texts = _flatten_articles(sample)
        text_feats = text_encoder(texts)
    else:
        text_feats = _text_encoder_placeholder(
            _flatten_articles(sample), dim=text_dim
        )

    ts_feats = _time_series_features(sample.get("time_series") or [])
    table_feats = _table_features(sample.get("table_data") or [])

    X = np.concatenate([text_feats, ts_feats, table_feats]).astype(np.float32)
    target = sample.get("target") or []
    y = np.array(
        [float(t) if t is not None else np.nan for t in target],
        dtype=np.float32,
    )
    ts_list = sample.get("time_series") or []
    input_price = np.nan
    if ts_list:
        last_day = ts_list[-1]
        if isinstance(last_day, dict):
            input_price = last_day.get("close")
    if input_price is None or (isinstance(input_price, float) and np.isnan(input_price)):
        input_price = np.nan

    meta = {
        "dates": sample.get("dates") or [],
        "input_price": float(input_price) if input_price is not None else np.nan,
        "target": y.tolist(),
        "ticker_text": sample.get("ticker_text", ""),
        "ticker_id": sample.get("ticker_id", -1),
    }
    return X, y, meta

FEATURE_BATCH_SIZE = 5000


def build_tabnet_features(
    config: Dict,
    text_encoder: Optional[Callable[[List[str]], np.ndarray]] = None,
    text_dim: int = 64,
    batch_size: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict], List[Dict]]:
    set_seed()
    if batch_size is None:
        batch_size = config.get("feature_batch_size", FEATURE_BATCH_SIZE)
    data_config = {
        "data_path": config["data_path"],
        "ticker2idx": config["ticker2idx"],
        "test_train_split": config.get("test_train_split", 0.2),
        "random_seed": config.get("random_seed", 42),
    }
    train_dl, test_dl = getTrainTestDataLoader(data_config)

    def _collect(dataloader, desc):
        X_list, y_list, meta_list = [], [], []
        n = len(dataloader)
        for start in tqdm(range(0, n, batch_size), desc=desc):
            end = min(start + batch_size, n)
            batch_x, batch_y, batch_meta = [], [], []
            for idx in range(start, end):
                sample = dataloader[idx]
                X, y, meta = _sample_to_vector(
                    sample, text_encoder=text_encoder, text_dim=text_dim
                )
                if np.isnan(y).all():
                    continue
                batch_x.append(X)
                batch_y.append(y)
                batch_meta.append(meta)
            if batch_x:
                X_list.append(np.stack(batch_x, axis=0))
                y_list.append(np.stack(batch_y, axis=0))
                meta_list.extend(batch_meta)

        if not X_list:
            return [], [], []
        return (
            np.concatenate(X_list, axis=0),
            np.concatenate(y_list, axis=0),
            meta_list,
        )

    X_train, y_train, meta_train = _collect(train_dl, "Train features")
    X_test, y_test, meta_test = _collect(test_dl, "Test features")

    if len(meta_train) == 0 or len(meta_test) == 0:
        raise ValueError("No valid samples after feature building (all targets NaN?).")

    return X_train, y_train, X_test, y_test, meta_train, meta_test

if __name__ == "__main__":
    import os
    from src.utils import read_json_file, read_yaml

    config = read_yaml("config/config.yaml")
    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    data_config = {
        "data_path": config["BASELINE_DATA_PATH"],
        "ticker2idx": ticker2idx,
        "test_train_split": 0.2,
        "random_seed": 42,
    }
    X_train, y_train, X_test, y_test, meta_train, meta_test = build_tabnet_features(
        data_config
    )
    print("X_train", X_train.shape, "y_train", y_train.shape)
    print("X_test", X_test.shape, "y_test", y_test.shape)
    print("Sample meta:", meta_train[0])
