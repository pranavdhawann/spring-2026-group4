"""Enhanced PyTorch LSTM model for time series forecasting."""
import torch
import torch.nn as nn


class TimeSeriesModel(nn.Module):
    """
    Enhanced LSTM-based model for time series prediction.
    
    Improvements:
    - Optional attention mechanism
    - Layer normalization
    - Residual connections (optional)
    - Better initialization
    
    Args:
        - input_size (int): Number of input features per timestep. Default: 5
        - hidden_size (int): LSTM hidden dimension. Default: 64
        - num_layers (int): Number of stacked LSTM layers. Default: 2
        - dropout (float): Dropout between LSTM layers. Default: 0.1
        - output_size (int): Number of output predictions. Default: 1
        - bidirectional (bool): Use bidirectional LSTM. Default: False
        - use_attention (bool): Add attention mechanism. Default: False
        - use_layer_norm (bool): Add layer normalization. Default: True
    """

    DEFAULTS = {
        "input_size": 5,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
        "output_size": 1,
        "bidirectional": False,
        "use_attention": False,
        "use_layer_norm": True,
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
        self.use_attention = self.config["use_attention"]
        self.use_layer_norm = self.config["use_layer_norm"]
        self.num_directions = 2 if self.bidirectional else 1

        self.input_proj = nn.Linear(self.input_size, self.hidden_size)
        
        if self.use_layer_norm:
            self.input_norm = nn.LayerNorm(self.hidden_size)

        self.lstm = nn.LSTM(
            input_size=self.hidden_size, 
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        if self.use_attention:
            attn_input_size = self.hidden_size * self.num_directions
            self.attention = nn.Sequential(
                nn.Linear(attn_input_size, attn_input_size // 2),
                nn.Tanh(),
                nn.Linear(attn_input_size // 2, 1)
            )

        fc_input_size = self.hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, fc_input_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(fc_input_size // 2, self.output_size)
        )

        self._init_weights()

    def _init_weights(self):
        """Better weight initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)
        
        if self.use_layer_norm:
            x = self.input_norm(x)

        # LSTM encoding
        lstm_out, (h_n, _) = self.lstm(x)

        if self.use_attention:
            attn_weights = self.attention(lstm_out) 
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)  
        else:
            if self.bidirectional:
                h_forward = h_n[-2]  
                h_backward = h_n[-1]  
                context = torch.cat([h_forward, h_backward], dim=1)
            else:
                context = h_n[-1]

        out = self.fc(context)
        return out


class SimpleLSTM(nn.Module):
    """
    Simplified LSTM baseline - same as original for comparison.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.input_size = config.get("input_size", 5)
        self.hidden_size = config.get("hidden_size", 64)
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.1)
        self.bidirectional = config.get("bidirectional", False)
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        fc_input_size = self.hidden_size * self.num_directions
        self.fc = nn.Linear(fc_input_size, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        lstm_out, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        return self.fc(hidden)


class LSTMForecaster(nn.Module):
    """
    Config-driven LSTM for multi-step time-series forecasting.

    Extends the project's LSTM models with a configurable fully connected head
    built dynamically from the ``fc_hidden_sizes`` list.  This allows
    experimenting with different FC depths without changing code.

    Args (via config dict, missing keys fall back to DEFAULTS):
        input_size (int): Number of input features per timestep. Default: 16
        hidden_size (int): LSTM hidden dimension. Default: 128
        num_layers (int): Number of stacked LSTM layers. Default: 2
        dropout (float): Dropout rate. Default: 0.2
        output_size (int): Number of output predictions. Default: 7
        bidirectional (bool): Use bidirectional LSTM. Default: False
        use_attention (bool): Add attention mechanism. Default: False
        use_layer_norm (bool): Add layer normalization. Default: True
        fc_hidden_sizes (list[int]): Sizes for FC head hidden layers. Default: [64, 32]
    """

    DEFAULTS = {
        "input_size": 16,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "output_size": 7,
        "bidirectional": False,
        "use_attention": False,
        "use_layer_norm": True,
        "fc_hidden_sizes": [64, 32],
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
        self.use_attention = self.config["use_attention"]
        self.use_layer_norm = self.config["use_layer_norm"]
        self.fc_hidden_sizes = self.config["fc_hidden_sizes"]
        self.num_directions = 2 if self.bidirectional else 1

        # Input projection
        self.input_proj = nn.Linear(self.input_size, self.hidden_size)

        if self.use_layer_norm:
            self.input_norm = nn.LayerNorm(self.hidden_size)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Optional attention
        if self.use_attention:
            attn_input_size = self.hidden_size * self.num_directions
            self.attention = nn.Sequential(
                nn.Linear(attn_input_size, attn_input_size // 2),
                nn.Tanh(),
                nn.Linear(attn_input_size // 2, 1),
            )

        # Configurable FC head
        fc_input_size = self.hidden_size * self.num_directions
        fc_layers = []
        prev_size = fc_input_size
        for fc_size in self.fc_hidden_sizes:
            fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev_size = fc_size
        fc_layers.append(nn.Linear(prev_size, self.output_size))
        self.fc = nn.Sequential(*fc_layers)

        self._init_weights()

    def _init_weights(self):
        """Weight initialization: orthogonal for LSTM, Xavier for others."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)

        if self.use_layer_norm:
            x = self.input_norm(x)

        lstm_out, (h_n, _) = self.lstm(x)

        if self.use_attention:
            attn_weights = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = (lstm_out * attn_weights).sum(dim=1)
        else:
            if self.bidirectional:
                h_forward = h_n[-2]
                h_backward = h_n[-1]
                context = torch.cat([h_forward, h_backward], dim=1)
            else:
                context = h_n[-1]

        out = self.fc(context)
        return out

    def get_model_summary(self) -> str:
        """Return a string summary of model architecture and parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        lines = [
            "LSTMForecaster Summary",
            f"  Input size:           {self.input_size}",
            f"  Hidden size:          {self.hidden_size}",
            f"  Num LSTM layers:      {self.num_layers}",
            f"  Bidirectional:        {self.bidirectional}",
            f"  Attention:            {self.use_attention}",
            f"  Layer norm:           {self.use_layer_norm}",
            f"  FC hidden sizes:      {self.fc_hidden_sizes}",
            f"  Output size:          {self.output_size}",
            f"  Dropout:              {self.dropout}",
            f"  Total parameters:     {total_params:,}",
            f"  Trainable parameters: {trainable_params:,}",
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    config = {
        "input_size": 14,
        "hidden_size": 128,
        "num_layers": 3,
        "use_attention": True,
    }

    model = TimeSeriesModel(config)
    x = torch.randn(32, 20, 14)
    y = model(x)

    print(f"Enhanced Model:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Parameters:   {sum(p.numel() for p in model.parameters()):,}")

    simple = SimpleLSTM(config)
    y_simple = simple(x)
    print(f"\nSimple Model:")
    print(f"  Parameters:   {sum(p.numel() for p in simple.parameters()):,}")

    # LSTMForecaster demo
    forecaster_config = {
        "input_size": 16,
        "hidden_size": 128,
        "num_layers": 2,
        "fc_hidden_sizes": [64, 32],
        "output_size": 7,
    }

    forecaster = LSTMForecaster(forecaster_config)
    x_fc = torch.randn(32, 14, 16)  # batch=32, seq_len=14, features=16
    y_fc = forecaster(x_fc)

    print(f"\nLSTMForecaster:")
    print(f"  Input shape:  {x_fc.shape}")
    print(f"  Output shape: {y_fc.shape}")
    print(forecaster.get_model_summary())