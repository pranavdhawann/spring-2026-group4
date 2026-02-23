import time

import numpy as np
from transformers import AutoTokenizer

from .preProcessMultiModalFinBert import preprocessFinbertMMBaseline
from .preProcessMultiModalTcn import preprocessTCNMMBaseline


class MultiModalPreProcessing(object):
    def __init__(self, collator_cfg):
        self.config = {
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": False,
            "strip_whitespace": True,
            "language": "en",
            "tokenizer_path": "ProsusAI/finbert",
            "padding": False,
            "truncation": True,
            "max_length": 512,
            "local_files_only": True,
            "return_tensors": None,
            "news_stride": 512 // 2,
        }
        self.config.update(collator_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer_path"],
            local_files_only=self.config["local_files_only"],
        )

    def __call__(self, batch):
        return self._preprocess(batch)

    def _preprocess(self, batch, verbose=False):
        st_ = time.time()
        input_size = max(len(b["dates"]) for b in batch)
        X, y = [], []  # (B,) (B,) where B is batch_size
        for b in batch:  # to process each elements in a batch
            dates_ = b["dates"]  # (1,Input_window_size)
            articles_ = b["articles"]
            time_series_ = b["time_series"]
            # table_data_ = b["table_data"]
            sector_ = b["sector"]
            target_ = b["target"]
            ticker_text_ = b["ticker_text"]
            ticker_id_ = b["ticker_id"]

            pre_processed_articles = preprocessFinbertMMBaseline(
                articles_, dates_, self.tokenizer, self.config
            )


            pre_processed_time_series = preprocessTCNMMBaseline(
                time_series_, dates_, self.config, verbose=False
            )

            X_ = {
                "tokenized_news_": pre_processed_articles[
                    0
                ],  # (Input_window_size, article_len)
                "attention_mask_news_": pre_processed_articles[
                    1
                ],  # (Input_window_size, article_len)
                "time_series_features_": pre_processed_time_series[0],
                "ticker_text_": ticker_text_,
                "ticker_id_": ticker_id_,
                "sector_": sector_,
            }  # concat all the data

            # pre process targets
            closes = []
            for day in range(input_size):
                ts_close = (
                    b["time_series"][day]["close"] if b["time_series"][day] else np.nan
                )
                closes.append(ts_close)
            closes = self._replace_none_with_avg_np(closes)
            mean, std, closes = self._standardize_list_np(closes)
            targets = self._replace_none_with_avg_np(target_)
            _, _, targets = self._standardize_list_np(targets, mean, std)

            X.append(X_)
            y.append(targets)

        if verbose:
            print(" Time to pre process : ", time.time() - st_)
        return X, y

    @staticmethod
    def _replace_none_with_avg_np(arr):
        arr = np.array(arr, dtype=np.float32)
        mask = np.isnan(arr)
        avg = np.mean(arr[~mask]) if np.any(~mask) else 0.0
        arr[mask] = avg
        return arr

    @staticmethod
    def _standardize_list_np(arr, mean=None, std=None):
        arr = np.array(arr, dtype=np.float32)
        if mean is None:
            mean = arr.mean()
            std = arr.std()
            std = std if std != 0 else 1.0
        arr = (arr - mean) / std
        return mean, std, arr


if __name__ == "__main__":
    import os
    import pickle

    from torch.utils.data import DataLoader

    from src.dataLoader import getTrainTestDataLoader
    from src.utils import read_json_file, read_yaml

    config = {
        "yaml_config_path": "config/config.yaml",
        "experiment_path": "experiments/baseline/multiModal",
        "load_pre_trained": False,
        "batch_size": 64,
        "max_length": 512,
        "tokenizer_path": "ProsusAI/finbert",
        "local_files_only": True,
        "sample_fraction": 1,
        "rand_seed": 42,
        "verbose": False,
        "max_window_size": 14,
    }
    if not os.path.exists(config["experiment_path"]):
        os.makedirs(config["experiment_path"])
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

    collator = MultiModalPreProcessing(config)

    train_loader = DataLoader(
        train_dataloader,
        collate_fn=collator,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    for X, y in train_loader:
        print("batch_size : ", len(X))
        print("each data has : ", X[0].keys())
        break