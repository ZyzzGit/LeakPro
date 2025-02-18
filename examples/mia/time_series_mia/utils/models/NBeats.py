import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    """
    A single block in the N-BEATS architecture.
    """
    def __init__(self, input_dim, lookback, horizon, hidden_dim, num_layers):
        """
        Args:
        - input_dim: Number of features (variables).
        - lookback: Number of past time steps (input sequence length).
        - horizon: Number of future time steps to predict.
        - hidden_dim: Number of hidden units per layer.
        - num_layers: Number of fully connected layers per block.
        """
        super(NBeatsBlock, self).__init__()
        self.input_dim = input_dim
        self.lookback = lookback
        self.horizon = horizon

        layers = [nn.Linear(lookback * input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.fc_layers = nn.Sequential(*layers)
        self.backcast = nn.Linear(hidden_dim, lookback * input_dim)  # Residual reconstruction
        self.forecast = nn.Linear(hidden_dim, horizon * input_dim)  # Future prediction

    def forward(self, x):
        """
        Args:
        - x: Input shape (batch_size, lookback, input_dim)
        Returns:
        - backcast: Residual reconstruction (batch_size, lookback, input_dim)
        - forecast: Future prediction (batch_size, horizon, input_dim)
        """
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # Flatten to (batch_size, lookback * input_dim)
        x = self.fc_layers(x)
        backcast = self.backcast(x).reshape(batch_size, self.lookback, self.input_dim)
        forecast = self.forecast(x).reshape(batch_size, self.horizon, self.input_dim)
        return backcast, forecast

class NBeats(nn.Module):
    """
    N-BEATS model with multiple blocks.
    """
    def __init__(self, input_dim, lookback, horizon, hidden_dim=128, num_layers=4, num_blocks=3):
        """
        Args:
        - input_dim: Number of features (variables).
        - lookback: Number of past time steps (input sequence length).
        - horizon: Number of future time steps to predict.
        - hidden_dim: Number of hidden units per layer.
        - num_layers: Number of layers per block.
        - num_blocks: Number of stacked blocks.
        """
        super(NBeats, self).__init__()
        self.init_params = {"input_dim": input_dim,
                            "lookback": lookback,
                            "horizon": horizon,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "num_blocks": num_blocks}
        self.blocks = nn.ModuleList([NBeatsBlock(input_dim, lookback, horizon, hidden_dim, num_layers) for _ in range(num_blocks)])

    def forward(self, x):
        """
        Args:
        - x: Input shape (batch_size, lookback, input_dim)
        Returns:
        - Final forecast (batch_size, horizon, input_dim)
        """
        residual = x
        forecast = torch.zeros(x.shape[0], x.shape[2], self.blocks[0].forecast.out_features // x.shape[2], device=x.device).permute(0, 2, 1)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast  # Remove learned part
            forecast = forecast + block_forecast  # Aggregate forecast

        return forecast