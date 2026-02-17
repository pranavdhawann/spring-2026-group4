from typing import Dict

import torch
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModel, AutoTokenizer

from src.utils import set_seed


class FinBertForecastingBL(nn.Module):
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
            "teacher_forcing_ratio": 0.5,
            "scheduled_sampling": "linear",  # Options: "linear", "exponential", "constant", None
        }

        self.config.update(config)
        self.device = self.config["device"]

        self.teacher_forcing_ratio = self.config.get("teacher_forcing_ratio", 0.5)
        self.scheduled_sampling = self.config.get("scheduled_sampling", "linear")
        self.current_epoch = 0
        self.total_epochs = self.config.get("num_epochs", 100)

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

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_current_teacher_forcing_ratio(self) -> float:
        if self.scheduled_sampling is None:
            return 0.0

        if self.scheduled_sampling == "constant":
            return self.teacher_forcing_ratio

        elif self.scheduled_sampling == "linear":
            progress = min(1.0, self.current_epoch / max(1, self.total_epochs - 1))
            return max(0.0, self.teacher_forcing_ratio * (1 - progress))

        elif self.scheduled_sampling == "exponential":
            decay_rate = 0.9
            return self.teacher_forcing_ratio * (decay_rate**self.current_epoch)

        elif self.scheduled_sampling == "inverse_sigmoid":
            k = 0.3  # Steepness param
            progress = min(1.0, self.current_epoch / max(1, self.total_epochs - 1))
            return self.teacher_forcing_ratio * (
                1 - 1 / (1 + torch.exp(torch.tensor(-k * (progress - 0.5))))
            )

        else:
            return self.teacher_forcing_ratio

    def forward(self, inputs, targets=None, force_teacher_forcing=None):
        """
        Forward pass with optional teacher forcing

        Args:
            inputs: Dictionary containing:
                - "input_ids": (B, W, L)
                - "attention_mask": (B, W, L)
                - "extra_features": (B, 3)
                - "closes": (B, N) historical closes
            targets: Optional ground truth future values (B, T) for teacher forcing
            force_teacher_forcing: If True, force teacher forcing; if False, force no teacher forcing;
                                  if None, use scheduled sampling

        Returns:
            outputs: (B, T) predicted values
        """
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        extra_features = inputs["extra_features"].to(self.device)  # (B, 2)

        closes = inputs["closes"][:, -1].to(self.device)  # (B,)
        closes = closes.unsqueeze(1)  # (B, 1)
        extra_features = torch.cat([closes, extra_features], dim=1)  # (B, 3)

        B, W, L = input_ids.shape

        input_ids = input_ids.view(B * W, L)
        attention_mask = attention_mask.view(B * W, L)

        with autocast(device_type="cuda"):
            outputs = self.finbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        cls_embeddings = cls_embeddings.view(B, W, -1)  # (B, W, 768)

        _, (h_n, c_n) = self.encoder_lstm(cls_embeddings)
        encoder_hidden = h_n[-1]  # (B, hidden_dim)

        # Condition on features
        conditioned_hidden = torch.cat(
            [encoder_hidden, extra_features], dim=1
        )  # (B, hidden_dim + 3)
        projected_hidden = torch.tanh(
            self.feature_projection(conditioned_hidden)
        )  # (B, hidden_dim)

        # Initialize decoder
        decoder_hidden = (
            projected_hidden.unsqueeze(0),  # h_0 (num_layers, B, hidden_dim)
            c_n,  # reuse encoder cell
        )

        forecast_steps = self.config["FORECAST_HORIZON"]

        use_teacher_forcing = False
        if force_teacher_forcing is not None:
            use_teacher_forcing = force_teacher_forcing
        elif targets is not None and self.training:
            teacher_forcing_ratio = self.get_current_teacher_forcing_ratio()
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

        decoder_input = torch.zeros(B, 1, 1, device=self.device)
        outputs = []

        for step in range(forecast_steps):
            out, decoder_hidden = self.decoder_lstm(decoder_input, decoder_hidden)
            pred = self.regressor(out)  # (B, 1, 1)
            outputs.append(pred)

            if use_teacher_forcing and targets is not None:
                decoder_input = targets[:, step : step + 1].unsqueeze(-1)  # (B, 1, 1)
            else:
                decoder_input = pred

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
        "teacher_forcing_ratio": 0.7,
        "scheduled_sampling": "linear",
        "num_epochs": 100,
    }

    model = FinBertForecastingBL(config)
    tokenizer = AutoTokenizer.from_pretrained(config["finbert_name_or_path"])

    print("\n" + "=" * 50)
    print("DEMO: Teacher Forcing in Action")
    print("=" * 50)

    texts = [
        "Apple stock rose 3% after strong earnings report.",
        "Market volatility increases amid inflation concerns.",
        "Tesla shares drop following delivery miss.",
    ] * 3  # Create 9 samples

    batch_size = len(texts)
    forecast_horizon = config["FORECAST_HORIZON"]

    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    # Create dummy input dictionary
    dummy_inputs = {
        "input_ids": inputs["input_ids"].unsqueeze(1),  # Add window dimension
        "attention_mask": inputs["attention_mask"].unsqueeze(1),
        "extra_features": torch.randn(batch_size, 2),  # Dummy extra features
        "closes": torch.randn(batch_size, 30),  # Dummy historical closes
    }

    dummy_targets = torch.randn(batch_size, forecast_horizon)

    print(
        f"\n1. Training mode with teacher forcing (ratio={config['teacher_forcing_ratio']}):"
    )
    model.train()
    model.set_epoch(0)  # Early epoch - high teacher forcing
    with torch.no_grad():
        train_outputs = model(dummy_inputs, targets=dummy_targets)
    print(f"   Output shape: {train_outputs.shape}")
    print(f"   Sample predictions: {train_outputs[0, :3].detach().cpu().numpy()}")

    print("\n2. Training mode later epoch (reduced teacher forcing):")
    model.set_epoch(80)  # Late epoch - low teacher forcing
    with torch.no_grad():
        train_outputs_late = model(dummy_inputs, targets=dummy_targets)
    print(f"   Teacher forcing ratio: {model.get_current_teacher_forcing_ratio():.3f}")
    print(f"   Sample predictions: {train_outputs_late[0, :3].detach().cpu().numpy()}")

    print("\n3. Evaluation mode (no teacher forcing):")
    model.eval()
    with torch.no_grad():
        eval_outputs = model(dummy_inputs, force_teacher_forcing=False)
    print(f"   Sample predictions: {eval_outputs[0, :3].detach().cpu().numpy()}")

    print("\n. Forced teacher forcing (for testing):")
    with torch.no_grad():
        forced_outputs = model(
            dummy_inputs, targets=dummy_targets, force_teacher_forcing=True
        )
    print(f"   Sample predictions: {forced_outputs[0, :3].detach().cpu().numpy()}")

    # Demo scheduled sampling strategies
    print("\n" + "=" * 50)
    print("SCHEDULED SAMPLING STRATEGIES")
    print("=" * 50)

    strategies = ["constant", "linear", "exponential", "inverse_sigmoid", None]
    epochs = list(range(0, 101, 10))

    for strategy in strategies:
        print(f"\nStrategy: {strategy}")
        model.config["scheduled_sampling"] = strategy
        ratios = []
        for epoch in epochs:
            model.set_epoch(epoch)
            ratio = model.get_current_teacher_forcing_ratio()
            ratios.append(ratio)
        print(f"   Teacher forcing ratios over epochs: {[f'{r:.3f}' for r in ratios]}")
