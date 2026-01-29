import os
from pathlib import Path

import yaml


def read_yaml(path):
    assert os.path.exists(path), f"{path} does not exist"

    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def working_directory_to_src(
    parent_level=0,
):  # 0 is the 1st parent of the directory you are currently working
    project_root = Path().resolve().parents[0]
    os.chdir(str(project_root))
    print("Path set to :", os.getcwd())

    return 1
