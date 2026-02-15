import time

import numpy as np
from transformers import AutoTokenizer

from src.utils.tokenizer import tokenize_sentences


class FinBertCollator:
    def __init__(self, collator_cfg):
        self.config = {
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": False,
            "strip_whitespace": True,
            "language": "en",
            "tokenizer_path": "ProsusAI/finbert",
            "padding": "max_length",
            "truncation": True,
            "max_length": 512,
            "local_files_only": True,
        }
        self.config.update(collator_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer_path"],
            local_files_only=self.config["local_files_only"],
        )

    def __call__(self, batch):
        return self.preprocess(batch, False)

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

    def preprocess(self, batch, verbose=False):
        """
        Fast preprocessing for FinBERT Baseline
        """
        st_ = time.time()
        input_size = max(len(b["dates"]) for b in batch)
        X, y = [], []
        for b in batch:
            news_texts = []
            closes = []

            for i in range(input_size):
                day_article = b["articles"][i][0] if b["articles"][i] else ""
                news_texts.append(day_article)

                ts_close = (
                    b["time_series"][i]["close"] if b["time_series"][i] else np.nan
                )
                closes.append(ts_close)
            _, inputs = tokenize_sentences(
                news_texts, self.tokenizer, config=self.config, verbose=False
            )

            closes = self._replace_none_with_avg_np(closes)
            mean, std, closes = self._standardize_list_np(closes)

            targets = self._replace_none_with_avg_np(b["target"])
            _, _, targets = self._standardize_list_np(targets, mean, std)

            X_ = [inputs["input_ids"], inputs["attention_mask"], closes[-1], mean, std]
            X.append(X_)
            y.append(targets)
        if verbose:
            print(" Time to pre process : ", time.time() - st_)
        return X, y
