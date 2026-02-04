import json
import os
from pathlib import Path

import numpy as np
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
