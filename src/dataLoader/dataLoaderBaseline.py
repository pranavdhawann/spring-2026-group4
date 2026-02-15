import json
import os
import random

from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import read_jsonl


class BaselineDataLoader(Dataset):
    def __init__(self, config=None):
        self.config = {"train_or_test": "train", "data": None}  # 'train', 'test
        self.config.update(config)
        self._print_data_format()

    def _print_data_format(self):
        print(
            f"========================= DATA SUMMARY [{self.config['train_or_test']}] ========================="
        )
        print(f"     Number of data points : {self.__len__()}")
        print(
            f"     Data loader __getitem__ returns dictionary with these key values : {self.__getitem__(0).keys()}"
        )

    def __len__(self):
        return len(self.config["data"])

    def __getitem__(self, idx):
        jsonl_path, line_idx, ticker_text, ticker_id = self.config["data"][idx]

        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if i == line_idx:
                    out = json.loads(line)
                    break

        out["ticker_text"] = ticker_text
        out["ticker_id"] = ticker_id
        return out


def _add_ticker_feature(data, ticker_text, ticker_id):
    for data_ in data:
        data_["ticker_text"] = ticker_text
        data_["ticker_id"] = ticker_id

    return data


def _remove_data_with_no_news(data):
    filtered = []

    for data_ in data:
        articles = data_.get("articles")

        if isinstance(articles, list) and any(
            isinstance(a, list) and len(a) > 0 for a in articles
        ):
            filtered.append(data_)

    return filtered


def _test_train_split(data, config):
    random.shuffle(data)
    split_idx = int(len(data) * (1 - config["test_train_split"]))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    return train_data, test_data


def _load_valid_data_points(config):
    data = []

    for ticker, idx in tqdm(config["ticker2idx"].items(), desc="Loading data files"):
        jsonl_path = os.path.join(config["data_path"], f"{ticker.lower()}.jsonl")
        if not os.path.exists(jsonl_path):
            continue
        data_ = read_jsonl(jsonl_path)
        for element in range(len(data_)):
            data.append((jsonl_path, element, ticker, idx))

    print(f" Total Data Points : {len(data)}")
    train_set, test_set = _test_train_split(data, config)
    print(f"    Train split : {len(train_set)}")
    print(f"    Test split : {len(test_set)}")
    return train_set, test_set


def getTrainTestDataLoader(dataConfig):
    config = {
        "data_path": None,  # path to dataset folder
        "ticker2idx": None,  # Ticker to Id dictionary
        "test_train_split": 0.2,
        "random_seed": 42,
        "train_or_test": "train",  # 'train', 'test
    }
    if not os.path.exists(dataConfig["data_path"]):
        raise FileNotFoundError(f"file not found {dataConfig['data_path']}")
    config.update(dataConfig)
    random.seed(config["random_seed"])
    train_set, test_set = _load_valid_data_points(config)

    # train data loader
    train_dataloader = BaselineDataLoader({"train_or_test": "train", "data": train_set})

    # test data loader
    test_dataloader = BaselineDataLoader({"train_or_test": "test", "data": test_set})

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    from src.utils import read_json_file, read_yaml

    print(os.getcwd())
    config = read_yaml("./config/config.yaml")
    ticker2idx = read_json_file(
        os.path.join(config["BASELINE_DATA_PATH"], config["TICKER2IDX"])
    )
    data_config = {
        "data_path": config["BASELINE_DATA_PATH"],  # path to dataset folder
        "ticker2idx": ticker2idx,  # Ticker to Id dictionary
        "test_train_split": 0.2,
        "random_seed": 42,
    }
    train_dataloader, test_dataloader = getTrainTestDataLoader(data_config)
    n = train_dataloader.__len__()
    for idx in tqdm(range(n)):
        train_dataloader.__getitem__(idx)
