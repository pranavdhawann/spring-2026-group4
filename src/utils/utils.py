import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def read_yaml(path):
    assert os.path.exists(path), f"{path} does not exist"

    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def working_directory_to_src():
    current_path = Path().resolve()
    project_root = None
    for parent in current_path.parents:
        if parent.name == "spring-2026-group4":
            project_root = parent
            break

    if project_root is None:
        for parent in current_path.parents:
            if (parent / "src").exists():
                project_root = parent
                break
        else:
            print("Could not find project root. Staying in current directory.")
            return 0
    src_path = project_root / "src"
    if not src_path.exists():
        print(f"Warning: {src_path} does not exist. Creating it.")
        src_path.mkdir(parents=True)

    os.chdir(str(src_path))
    print("Path set to:", os.getcwd())

    return 1


def read_jsonl(path):
    assert os.path.exists(path), f"{path} does not exist"
    data = []
    with open(path, "r") as file:
        for line in file:
            data.append(json.loads(line))

    return data


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


def remove_outliers(data, factor=1.5):
    if not data:
        return []

    data_arr = np.array(data)
    Q1 = np.percentile(data_arr, 25)
    Q3 = np.percentile(data_arr, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    filtered_data = data_arr[(data_arr >= lower_bound) & (data_arr <= upper_bound)]
    print(f"25% : {Q1} 75%: {Q3}")

    return filtered_data.tolist()


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


def filter_timeseries_by_date(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    date_col: str = "Date",
) -> pd.DataFrame:
    df_filtered = df.copy()

    if start_date:
        df_filtered = df_filtered[df_filtered[date_col] >= start_date]

    if end_date:
        df_filtered = df_filtered[df_filtered[date_col] <= end_date]

    return df_filtered
