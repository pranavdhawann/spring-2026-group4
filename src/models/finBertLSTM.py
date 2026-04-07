from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from src.utils import set_seed


class FinBertForecastingBL(nn.Module):
    """
    FinBERT + Encoder LSTM (time) + Feature-conditioned Decoder LSTM
    """

    def __init__(self, config: Dict):
        super(FinBertForecastingBL, self).__init__()
        set_seed()

        self.config = {
            "finbert_name_or_path": "ProsusAI/finbert",
            "device": torch.device("cpu"),
            "local_files_only": True,
            "bert_hidden_dim": 768,
            "lstm_hidden_dim": 256,
            "lstm_num_layers": 1,
            "FORECAST_HORIZON": 7,
        }

        self.config.update(config)
        self.device = self.config["device"]

        self.finbert = AutoModel.from_pretrained(
            self.config["finbert_name_or_path"],
            local_files_only=self.config["local_files_only"],
        )

        self.encoder_lstm = nn.LSTM(
            input_size=self.config["bert_hidden_dim"],
            hidden_size=self.config["lstm_hidden_dim"],
            num_layers=self.config["lstm_num_layers"],
            batch_first=True,
        )

        # (hidden + 3 floats) → hidden_dim
        self.feature_projection = nn.Linear(
            self.config["lstm_hidden_dim"] + 3, self.config["lstm_hidden_dim"]
        )

        self.decoder_lstm = nn.LSTM(
            input_size=1,
            hidden_size=self.config["lstm_hidden_dim"],
            num_layers=self.config["lstm_num_layers"],
            batch_first=True,
        )

        self.regressor = nn.Linear(self.config["lstm_hidden_dim"], 1)

        self.to(self.device)

    def forward(self, inputs):
        """
        inputs:
        {
            "input_ids": (B, W, L)
            "attention_mask": (B, W, L)
            "extra_features": (B, 3)
        }
        """

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        extra_features = inputs["extra_features"].to(self.device)  # (B, 2)

        closes = inputs["closes"][:, -1].to(self.device)  # (B,)
        closes = closes.unsqueeze(1)  # (B, 1)
        extra_features = torch.cat([closes, extra_features], dim=1)

        B, W, L = input_ids.shape

        input_ids = input_ids.view(B * W, L)
        attention_mask = attention_mask.view(B * W, L)

        outputs = self.finbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings.view(B, W, -1)  # (B, W, 768)

        _, (h_n, c_n) = self.encoder_lstm(cls_embeddings)

        encoder_hidden = h_n[-1]  # (B, hidden_dim)

        conditioned_hidden = torch.cat(
            [encoder_hidden, extra_features], dim=1
        )  # (B, hidden_dim + 3)

        projected_hidden = torch.tanh(
            self.feature_projection(conditioned_hidden)
        )  # (B, hidden_dim)

        decoder_hidden = (
            projected_hidden.unsqueeze(0),  # h_0
            c_n,  # reuse encoder cell
        )

        forecast_steps = self.config["FORECAST_HORIZON"]

        decoder_input = torch.zeros(B, 1, 1, device=self.device)
        outputs = []

        for _ in range(forecast_steps):
            out, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)

            pred = self.regressor(out)  # (B,1,1)
            outputs.append(pred)
            decoder_input = pred  # autoregressive

        outputs = torch.cat(outputs, dim=1)  # (B, T, 1)
        return outputs.squeeze(-1)  # (B, T)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    config = {
        "finbert_name_or_path": "ProsusAI/finbert",
        "device": device,
        "FORECAST_HORIZON": 7,
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
