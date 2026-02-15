"""

Script to train FinBert model

"""
import os
import pickle

from src.dataLoader import getTrainTestDataLoader

# from src.models import finBertForecastingBaseline
from src.utils import read_json_file, read_yaml, set_seed


def train(train_config):
    set_seed()
    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/finbertForecasting",
    }

    config.update(train_config)

    yaml_config = read_yaml(config["yaml_config_path"])
    config.update(yaml_config)
    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    data_config = {
        "data_path": config["BASELINE_DATA_PATH"],  # path to dataset folder
        "ticker2idx": ticker2idx,  # Ticker to Id dictionary
        "test_train_split": 0.2,
    }

    config.update(data_config)

    # load or process dataloader from pkl file
    if not os.path.exists(os.path.join(config["experiment_path"], "dataloaders.pkl")):
        train_dataloader, test_dataloader = getTrainTestDataLoader(config)
        with open(
            os.path.join(config["experiment_path"], "dataloaders.pkl"), "wb"
        ) as f:
            pickle.dump({"train": train_dataloader, "test": test_dataloader}, f)
        print("Dataloaders saved!")
    else:
        with open(
            os.path.join(config["experiment_path"], "dataloaders.pkl"), "rb"
        ) as f:
            dataloaders = pickle.load(f)

        train_dataloader = dataloaders["train"]
        test_dataloader = dataloaders["test"]

        print("Dataloaders loaded successfully!")


if __name__ == "__main__":
    train_config = {}
    train(train_config)
