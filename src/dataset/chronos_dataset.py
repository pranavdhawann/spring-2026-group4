"""
Custom PyTorch Dataset for Chronos-2 fine-tuning.

Wraps the preprocessed dictionary lists (from data_preprocessing_chronos.py)
into a HuggingFace-compatible dataset yielding `input_ids`, `attention_mask`, and `labels`.
"""

import math
from typing import List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from chronos import ChronosTokenizer


class ChronosFineTuningDataset(Dataset):
    """
    A mapping-style PyTorch dataset for Chronos-2 fine-tuning.

    Given a list of ticker time series (loaded via `load_all_tickers`),
    it extracts sliding windows of `context_length + prediction_length` 
    and tokenizes them using the `ChronosTokenizer`.
    """

    def __init__(
        self,
        data_list: List[Dict],
        tokenizer: ChronosTokenizer,
        context_length: int = 512,
        prediction_length: int = 64,
        stride: int = 1,
        mode: str = "training",
    ):
        """
        Args:
            data_list: List of dictionaries e.g. [{"ticker": "AAPL", "values": np.array([...])}]
            tokenizer: An initialized ChronosTokenizer.
            context_length: Number of historical steps to use.
            prediction_length: Number of future steps to predict.
            stride: The window sliding stride.
            mode: "training" or "validation". If validation, we might want to sample 
                  less aggressively or take the very last window.
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.mode = mode

        self.samples = []
        self._build_windows(data_list)

    def _build_windows(self, data_list: List[Dict]):
        """
        Process the raw data list into fixed-size windows.
        """
        window_size = self.context_length + self.prediction_length

        for ticker_data in data_list:
            values = ticker_data["values"]
            n = len(values)

            if n < window_size:
                continue

            if self.mode == "training":
                # Extract sliding windows
                for i in range(0, n - window_size + 1, self.stride):
                    window = values[i : i + window_size]
                    self.samples.append(window)
            else:
                # For validation/testing, typically take non-overlapping windows 
                # or just the last available window to evaluate zero-shot forecasting.
                # Here we'll take non-overlapping chunks from the end.
                for i in range(n - window_size, -1, -window_size):
                    window = values[i : i + window_size]
                    self.samples.append(window)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.samples[idx]
        
        # Split into context and target
        context = window[: self.context_length]
        target = window[self.context_length :]

        # Convert to tensors
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0) # [1, context_len]
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)   # [1, target_len]

        # Tokenize context
        # Returns: input_ids [1, num_tokens], attention_mask [1, num_tokens], scale [1, 1]
        input_ids, attention_mask, scale = self.tokenizer.context_input_transform(context_tensor)

        # Tokenize target
        # Returns: labels [1, num_tokens], labels_mask [1, num_tokens]
        labels, labels_mask = self.tokenizer.label_input_transform(target_tensor, scale)
        
        # HF expects ignored labels to be -100
        labels[labels_mask == 0] = -100

        return {
            "input_ids": input_ids.squeeze(0),         # [num_tokens]
            "attention_mask": attention_mask.squeeze(0), # [num_tokens]
            "labels": labels.squeeze(0)                  # [num_tokens]
        }
