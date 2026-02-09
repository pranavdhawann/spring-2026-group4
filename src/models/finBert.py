"""
src/models/finbert.py

Configurable FinBERT model wrapper for sentiment inference and training.
Supports:
- Hugging Face or local model loading
- Tokenized tensor inputs
- Sliding window inference for long sequences
- Configurable aggregation strategies
"""

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class FinBERTModel(nn.Module):
    """
    Generic FinBERT wrapper supporting training and inference
    with sliding window aggregation for long sequences.
    """

    def __init__(self, config: Dict):
        super(FinBERTModel, self).__init__()

        # -------- Defaults --------
        self.config = {
            "model_name_or_path": "ProsusAI/finbert",
            "num_labels": 3,
            "max_tokens": 512,
            "window_size": 512,
            "stride": 256,
            "aggregation": "mean",  # mean | median
            "device": "cuda",
            "local_files_only": False,
        }

        # Override defaults
        self.config.update(config)

        self.device = self.config["device"]

        # -------- Load model config --------
        hf_config = AutoConfig.from_pretrained(
            self.config["model_name_or_path"],
            num_labels=self.config["num_labels"],
            local_files_only=self.config["local_files_only"],
        )

        # -------- Load model --------
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config["model_name_or_path"],
            config=hf_config,
            local_files_only=self.config["local_files_only"],
        )

        self.model.to(self.device)

        # Trainable by default (explicit for clarity)
        for p in self.model.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    def _sliding_windows(self, tensor: torch.Tensor):
        """
        Generate sliding windows over sequence dimension.
        """
        window_size = self.config["window_size"]
        stride = self.config["stride"]
        seq_len = tensor.size(1)

        windows = []
        for start in range(0, seq_len, stride):
            end = start + window_size
            window = tensor[:, start:end]

            if window.size(1) < window_size:
                # Pad last window
                pad_len = window_size - window.size(1)
                pad = torch.zeros(
                    (tensor.size(0), pad_len),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                window = torch.cat([window, pad], dim=1)

            windows.append(window)

            if end >= seq_len:
                break

        return windows

    # ------------------------------------------------------------------
    def _aggregate(self, logits: torch.Tensor):
        """
        Aggregate logits across windows.
        Shape: (num_windows, batch, num_labels)
        """
        method = self.config["aggregation"].lower()

        if method == "mean":
            return logits.mean(dim=0)
        elif method == "median":
            return logits.median(dim=0).values
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

    # ------------------------------------------------------------------
    def forward(self, inputs: Dict[str, torch.Tensor], **kwargs):
        """
        Forward pass.
        Accepts tokenized tensors:
        - input_ids
        - attention_mask
        - optional token_type_ids
        """

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        token_type_ids = inputs.get("token_type_ids")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        batch_size, seq_len = input_ids.shape

        # -------- Short sequence --------
        if seq_len <= self.config["max_tokens"]:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                **kwargs,
            )
            return outputs

        # -------- Long sequence: sliding window --------
        input_windows = self._sliding_windows(input_ids)
        mask_windows = self._sliding_windows(attention_mask)
        type_windows = (
            self._sliding_windows(token_type_ids)
            if token_type_ids is not None
            else None
        )

        logits_per_window = []

        for i in range(len(input_windows)):
            outputs = self.model(
                input_ids=input_windows[i],
                attention_mask=mask_windows[i],
                token_type_ids=type_windows[i] if type_windows else None,
                **kwargs,
            )
            logits_per_window.append(outputs.logits.unsqueeze(0))

        # Shape: (num_windows, batch, num_labels)
        stacked_logits = torch.cat(logits_per_window, dim=0)

        aggregated_logits = self._aggregate(stacked_logits)

        return {
            "logits": aggregated_logits,
            "window_logits": stacked_logits,
        }


# ======================================================================
# Example Driver Code
# ======================================================================
if __name__ == "__main__":
    import torch

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = {
        "model_name_or_path": "ProsusAI/finbert",
        "aggregation": "mean",
        "device": device,
    }

    model = FinBERTModel(config)
    model.eval()

    n_sentences = 10
    seq_len = 1024  # longer than max_tokens
    vocab_size = 30522

    inputs = {
        "input_ids": torch.randint(0, vocab_size, (n_sentences, seq_len)),
        "attention_mask": torch.ones(n_sentences, seq_len),
    }
    print(inputs["input_ids"].shape)

    with torch.no_grad():
        outputs = model(inputs)

    print("Aggregated logits:", outputs["logits"])
    print("Per-window logits shape:", outputs["window_logits"].shape)
