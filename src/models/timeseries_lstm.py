"""Generic PyTorch LSTM model for time series forecasting."""
import torch
import torch.nn as nn


class TimeSeriesModel(nn.Module):
    """
    Configurable LSTM-based model for time series prediction.
    All architecture parameters are passed via a config dictionary.

    Args:
        - input_size (int): Number of input features per timestep. Default: 5
        - hidden_size (int): LSTM hidden dimension. Default: 64
        - num_layers (int): Number of stacked LSTM layers. Default: 2
        - dropout (float): Dropout between LSTM layers. Default: 0.1
        - output_size (int): Number of output predictions. Default: 1
        - bidirectional (bool): Use bidirectional LSTM. Default: False
    """

    DEFAULTS = {
        "input_size": 5,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "output_size": 1,
        "bidirectional": False,
    }

    def __init__(self, config: dict):
        super().__init__()

        self.config = {**self.DEFAULTS, **config}

        self.input_size = self.config["input_size"]
        self.hidden_size = self.config["hidden_size"]
        self.num_layers = self.config["num_layers"]
        self.dropout = self.config["dropout"]
        self.output_size = self.config["output_size"]
        self.bidirectional = self.config["bidirectional"]
        self.num_directions = 2 if self.bidirectional else 1

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Output projection
        fc_input_size = self.hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, self.output_size)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        lstm_out, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            h_forward = h_n[-2]  
            h_backward = h_n[-1]  
            hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            hidden = h_n[-1] 

        out = self.fc(hidden)
        return out


if __name__ == "__main__":
    config = {
        "input_size": 16,
        "hidden_size": 64,
        "num_layers": 2,
        "output_size": 1,
    }

    model = TimeSeriesModel(config)
    x = torch.randn(8, 20, 16) 
    y = model(x)

    print(f"Config: {model.config}")
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")

    config_bidir = {
        "input_size": 5,
        "hidden_size": 32,
        "num_layers": 3,
        "dropout": 0.2,
        "output_size": 1,
        "bidirectional": True,
    }

    model_bidir = TimeSeriesModel(config_bidir)
    x_ohlcv = torch.randn(4, 30, 5)  
    y_bidir = model_bidir(x_ohlcv)

    print(f"\nBidirectional model:")
    print(f"Input shape:  {x_ohlcv.shape}")
    print(f"Output shape: {y_bidir.shape}")
    print(f"Parameters:   {sum(p.numel() for p in model_bidir.parameters()):,}")

    minimal_config = {"input_size": 10}
    model_minimal = TimeSeriesModel(minimal_config)
    x_min = torch.randn(2, 15, 10)
    y_min = model_minimal(x_min)

    print(f"\nMinimal config (defaults applied):")
    print(f"Config: {model_minimal.config}")
    print(f"Output shape: {y_min.shape}")
