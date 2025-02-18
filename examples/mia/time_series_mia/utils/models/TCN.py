import torch
import torch.nn as nn
import torch.nn.functional as F

class Chomp1d(nn.Module):
    """Ensures the convolution output has the correct size by trimming padding."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    """A single residual block in the Temporal Convolutional Network."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation  # Ensures causality
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # Removes extra padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return out + self.residual(x)  # Residual connection

class TCN(nn.Module):
    """Full Temporal Convolutional Network with multiple residual blocks."""
    def __init__(self, input_dim, horizon, num_channels=None, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        if num_channels is None:
            num_channels = [2,2]
        self.init_params = {"input_dim": input_dim,
                            "horizon": horizon,
                            "num_channels": num_channels,
                            "kernel_size": kernel_size,
                            "dropout": dropout}

        self.input_dim = input_dim
        self.horizon = horizon

        layers = []
        num_layers = len(num_channels)

        for i in range(num_layers):
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = 2 ** i  # Exponential growth in dilation
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], input_dim * horizon)

    def forward(self, x):
        x = x.movedim(-1, -2)
        x = self.network(x)  # Pass through TCN layers
        x = x[:, :, -1]  # Take the last time step
        x = self.fc(x)
        x = x.view(-1, self.horizon, self.input_dim)
        return x  # Fully connected output

