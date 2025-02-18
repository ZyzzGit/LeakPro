import torch
import torch.nn as nn

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Used in TFT for feature transformation.
    """
    def __init__(self, input_dim, hidden_dim):
        super(GatedResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Add a residual projection if input_dim != hidden_dim
        self.residual_layer = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x):
        """
        Args:
            x: Input of shape (batch_size, seq_len, input_dim)
        Returns:
            Transformed output of shape (batch_size, seq_len, hidden_dim)
        """
        # Project residual if needed
        residual = self.residual_layer(x) if self.residual_layer is not None else x
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        gate = self.sigmoid(self.gate(out))
        out = gate * out + (1 - gate) * residual
        return self.layer_norm(out)

class TemporalSelfAttention(nn.Module):
    """
    Multi-Head Temporal Self-Attention Mechanism in TFT.
    """
    def __init__(self, hidden_dim, num_heads):
        super(TemporalSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Input of shape (batch_size, seq_len, hidden_dim)
        Returns:
            Attention output of shape (batch_size, seq_len, hidden_dim)
        """
        attn_output, _ = self.attention(x, x, x)
        return self.layer_norm(attn_output + x)

class TFT(nn.Module):
    """
    Temporal Fusion Transformer (TFT) for time series forecasting.
    """
    def __init__(self, input_dim, lookback, horizon, hidden_dim=128, num_heads=4):
        """
        Args:
            input_dim: Number of features (variables).
            lookback: Number of past time steps (input sequence length).
            horizon: Number of future time steps to predict.
            hidden_dim: Hidden dimension for embeddings.
            num_heads: Number of attention heads.
        """
        super(TFT, self).__init__()
        self.init_params = {"input_dim": input_dim,
                            "lookback": lookback,
                            "horizon": horizon,
                            "hidden_dim": hidden_dim,
                            "num_heads": num_heads}

        self.lookback = lookback
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # Gated Residual Network for Feature Transformation
        self.feature_transform = GatedResidualNetwork(input_dim, hidden_dim)

        # Temporal Self-Attention Block
        self.temporal_attention = TemporalSelfAttention(hidden_dim, num_heads)

        # Fully Connected Output Layer
        self.fc = nn.Linear(hidden_dim, horizon * input_dim)

    def forward(self, x):
        """
        Args:
            x: Input shape (batch_size, lookback, input_dim)
        Returns:
            Forecast output (batch_size, horizon, input_dim)
        """
        batch_size = x.shape[0]

        # Feature Transformation
        x = self.feature_transform(x)

        # Temporal Self-Attention
        x = self.temporal_attention(x)

        # Global context aggregation and forecast generation
        x = x.mean(dim=1)  # (batch_size, hidden_dim)
        x = self.fc(x)     # (batch_size, horizon * input_dim)
        return x.view(batch_size, self.horizon, -1)  # (batch_size, horizon, input_dim)
