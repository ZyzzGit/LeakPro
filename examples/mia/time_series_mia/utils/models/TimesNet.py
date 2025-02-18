import torch
import torch.nn as nn

class TimeSeriesBlock(nn.Module):
    """
    A single Time-Series Block (TSBlock) using 2D convolutions.
    Assumes input shape: (batch_size, lookback, input_dim)
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(TimeSeriesBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=hidden_dim,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0)
        )
        # Change: out_channels remains input_dim so that after conv2, 
        # the shape is (batch, input_dim, lookback, input_dim)
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=input_dim,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0)
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (batch, lookback, input_dim)
        x = x.unsqueeze(1)  # -> (batch, 1, lookback, input_dim)
        x = self.conv1(x)   # -> (batch, hidden_dim, lookback, input_dim)
        x = self.activation(x)
        x = self.conv2(x)   # -> (batch, input_dim, lookback, input_dim)
        # Instead of squeezing (which would only remove dims of size 1),
        # take the mean over the channel dimension to reduce it.
        x = x.mean(dim=1)   # -> (batch, lookback, input_dim)
        return x

class TimesNet(nn.Module):
    """
    TimesNet: A convolution-based model for long-term time series forecasting.
    Expects input shape: (batch, lookback, input_dim)
    Outputs: (batch, horizon, input_dim)
    """
    def __init__(self, input_dim, lookback, horizon, hidden_dim=64, num_blocks=2):
        super(TimesNet, self).__init__()
        self.init_params = {
            "input_dim": input_dim,
            "lookback": lookback,
            "horizon": horizon,
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks
        }
        self.blocks = nn.ModuleList(
            [TimeSeriesBlock(input_dim, hidden_dim) for _ in range(num_blocks)]
        )
        # Apply a fully connected layer along the lookback (time) dimension.
        self.fc = nn.Linear(lookback, horizon)

    def forward(self, x):
        # x: (batch, lookback, input_dim)
        for block in self.blocks:
            x = block(x)
        # Permute so that the FC layer is applied along the time axis.
        x = x.permute(0, 2, 1)   # -> (batch, input_dim, lookback)
        x = self.fc(x)           # -> (batch, input_dim, horizon)
        x = x.permute(0, 2, 1)   # -> (batch, horizon, input_dim)
        return x
