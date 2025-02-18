import torch
import torch.nn as nn

class DLinear(nn.Module):
    """
    DLinear: Decomposition-Linear Model for Multivariate Time Series Forecasting.
    Works with input of shape (batch_size, seq_len, input_dim).
    """
    def __init__(self, input_dim, lookback, horizon, individual=False):
        """
        Args:
        - lookback: Number of past time steps (lookback window).
        - horizon: Number of future time steps to predict.
        - input_dim: Number of features (variables).
        - individual: Whether to use separate linear layers per feature.
        """
        super(DLinear, self).__init__()
        self.init_params = {"input_dim": input_dim,
                            "lookback": lookback,
                            "horizon": horizon,
                            "individual": individual}
        self.lookback = lookback
        self.horizon = horizon
        self.input_dim = input_dim
        self.individual = individual

        if self.individual:
            self.linear_trend = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(input_dim)])
            self.linear_seasonal = nn.ModuleList([nn.Linear(lookback, horizon) for _ in range(input_dim)])
        else:
            self.linear_trend = nn.Linear(lookback, horizon)
            self.linear_seasonal = nn.Linear(lookback, horizon)

    def forward(self, x):
        """
        Args:
        - x: Input time series of shape (batch_size, seq_len, input_dim)
        Returns:
        - Forecast of shape (batch_size, horizon, input_dim)
        """
        mean = torch.mean(x, dim=1, keepdim=True)  # Compute mean for trend
        trend_component = x - mean  # Detrend the time series
        seasonal_component = x - trend_component  # Isolate seasonal pattern

        if self.individual:
            trend_out = torch.stack(
                [self.linear_trend[i](trend_component[:, :, i]) for i in range(self.input_dim)], dim=-1
            )
            seasonal_out = torch.stack(
                [self.linear_seasonal[i](seasonal_component[:, :, i]) for i in range(self.input_dim)], dim=-1
            )
        else:
            trend_out = self.linear_trend(trend_component.permute(0, 2, 1)).permute(0, 2, 1)
            seasonal_out = self.linear_seasonal(seasonal_component.permute(0, 2, 1)).permute(0, 2, 1)

        return trend_out + seasonal_out  # Sum of trend and seasonal components
