import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml


def set_seed(seed=42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    print(f"Random seed set to {seed}")


def read_yaml(path):
    assert os.path.exists(path), f"{path} does not exist"

    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def read_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    except FileNotFoundError:
        print(f"Error: File not found -> {file_path}")

    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file -> {file_path}")

    except Exception as e:
        print(f"Unexpected error: {e}")


def load_stock_csv(
    ticker: str, stock_data_dir: Path, start_year: Optional[int] = None
) -> pd.DataFrame:
    csv_file = stock_data_dir / f"{ticker.lower()}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"Stock data not found: {csv_file}")

    df = pd.read_csv(csv_file)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df = df.dropna(subset=["Date"]).sort_values("Date")

    if start_year:
        df = df[df["Date"].dt.year >= start_year]

    return df
