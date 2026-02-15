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
from transformers import AutoModel, AutoTokenizer

from src.utils import set_seed


class FinBertForecastingBL(nn.Module):
    """
    FinBERT encoder + LSTM decoder for time-series forecasting
    """

    def __init__(self, config: Dict):
        set_seed()
        super(FinBertForecastingBL, self).__init__()

        # Default config
        self.config = {
            "finbert_name_or_path": "ProsusAI/finbert",
            "device": torch.device("cpu"),
            "local_files_only": False,
            "bert_hidden_dim": 768,  # must match finbert hidden size
            "lstm_hidden_dim": 256,
            "lstm_num_layers": 1,
            "lstm_dropout": 0.1,
            "forecast_duration": 7,
        }

        self.config.update(config)
        self.device = self.config["device"]

        # finBert Encoder
        self.finbert = AutoModel.from_pretrained(
            self.config["finbert_name_or_path"],
            local_files_only=self.config["local_files_only"],
        )

        # lstm decoder
        self.lstm = nn.LSTM(
            input_size=self.config["bert_hidden_dim"],
            hidden_size=self.config["lstm_hidden_dim"],
            num_layers=self.config["lstm_num_layers"],
            dropout=self.config["lstm_dropout"]
            if self.config["lstm_num_layers"] > 1
            else 0.0,
            batch_first=True,
        )

        # forecast head
        self.regressor = nn.Sequential(
            nn.Linear(
                self.config["lstm_hidden_dim"], self.config["lstm_hidden_dim"] // 2
            ),
            nn.ReLU(),
            nn.Linear(
                self.config["lstm_hidden_dim"] // 2, self.config["forecast_duration"]
            ),
        )

        self.to(self.device)

    def forward(self, inputs):
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        token_type_ids = inputs.get("token_type_ids")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)

        outputs = self.finbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state

        lstm_out, (h_n, c_n) = self.lstm(sequence_output)

        final_hidden = h_n[-1]

        forecast = self.regressor(final_hidden)

        return forecast


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = {
        "finbert_name_or_path": "ProsusAI/finbert",
        "device": device,
        "forecast_duration": 7,
        "lstm_hidden_dim": 256,
    }
    model = FinBertForecastingBL(config)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config["finbert_name_or_path"])
    texts = [
        "Apple stock rose 3% after strong earnings report.",
        "Market volatility increases amid inflation concerns.",
        "Tesla shares drop following delivery miss.",
        "Federal Reserve signals potential rate cuts.",
        "Oil prices surge due to supply constraints.",
    ]
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(inputs)

    print("Input batch size:", len(texts))
    print("Forecast output shape:", outputs.shape)
    print("Forecast output:")
    print(outputs)
