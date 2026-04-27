import time

import numpy as np
import torch
from transformers import AutoTokenizer

from src.utils.tokenizer import tokenize_sentences


class ChronosCollator:
    def __init__(self, collator_cfg):
        self.config = {
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": False,
            "strip_whitespace": True,
            "language": "en",
            "tokenizer_path": "/home/ubuntu/capstone/ProsusAI/finbert",
            "padding": "max_length",
            "truncation": True,
            "max_length": 512,
            "return_tensors": "pt",
            "local_files_only": False,
        }
        self.config.update(collator_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["tokenizer_path"],
            local_files_only=self.config["local_files_only"],
            revision="main",
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

    def preprocess(self, batch, verbose=False):
        """
        Fast preprocessing for ChronosFinBert with log returns.
        Fix: target log returns are anchored using the last log PRICE,
        not the last log return, so inputs and targets are on the same scale.
        """
        st_ = time.time()
        input_size = max(len(b["dates"]) for b in batch)
        X, y = [], []
        for b in batch:
            T = len(b["dates"])
            news_texts = []

            for i in range(T):
                day_article = b["articles"][i][0] if b["articles"][i] else ""
                news_texts.append(day_article)

            ts_closes_raw = b.get("ts_closes", [])
            closes = [float(v) if v else np.nan for v in ts_closes_raw]
            if not closes:
                closes = [np.nan] * self.config.get("HISTORY_WINDOW_SIZE", 60)

            tokenize_config = {
                k: v for k, v in self.config.items() if k != "local_files_only"
            }
            _, inputs = tokenize_sentences(
                news_texts, self.tokenizer, config=tokenize_config, verbose=False
            )

            # Pad input_ids and attention_mask to input_size windows with zeros
            ids = inputs["input_ids"]  # (T, L)
            mask = inputs["attention_mask"]  # (T, L)
            if T < input_size:
                L = ids.shape[1]
                pad = input_size - T
                ids = torch.cat(
                    [ids, torch.zeros(pad, L, dtype=ids.dtype)], dim=0
                )  # (input_size, L)
                mask = torch.cat(
                    [mask, torch.zeros(pad, L, dtype=mask.dtype)], dim=0
                )  # (input_size, L)

            # Log returns instead of StandardScaler
            closes = self._replace_none_with_avg_np(closes)
            log_prices = np.log(
                np.clip(closes, 1e-8, None)
            )  # keep log prices for target anchor
            log_returns = np.diff(log_prices)
            log_returns = np.concatenate([[0.0], log_returns]).astype(np.float32)
            mean = float(np.mean(log_returns))
            std = float(np.std(log_returns) + 1e-8)
            closes = (log_returns - mean) / std

            # Targets as simple returns: (P_t - P_{t-1}) / P_{t-1}
            # Input is still log returns, output is simple returns (no log on targets)
            targets_raw = self._replace_none_with_avg_np(b["target"][:5])
            last_price = np.exp(log_prices[-1])  # recover last actual price
            all_prices = np.concatenate([[last_price], targets_raw])
            targets = np.diff(all_prices) / all_prices[:-1]  # simple returns
            targets = targets.astype(np.float32)

            X_ = [ids, mask, closes, mean, std]
            X.append(X_)
            y.append(targets)
        if verbose:
            print(" Time to pre process : ", time.time() - st_)
        return X, y
