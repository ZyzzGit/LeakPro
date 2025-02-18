import torch
import torch.nn as nn

class LSTM(nn.Module):
    """LSTM for multi-variate forecasting"""
    
    def __init__(self, input_dim, horizon, hidden_dim = 64, num_layers = 1):
        super().__init__()
        self.init_params = {"input_dim": input_dim,
                            "horizon": horizon,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers}
        self.input_dim = input_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, self.hidden_dim, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, input_dim * horizon)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x) # h_n shape: (num_layers, batch_size, hidden_size) 
        linear_out = self.linear(h_n[0])   
        return linear_out.view(-1, self.horizon, self.input_dim)   # reshape to (batch_size, horizon, num_variables) 
