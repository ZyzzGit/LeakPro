import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(
            self, 
            input_size, 
            hidden_size=128, 
            num_layers=2, 
            bidirectional=False
        ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(num_layers * self.hidden_size * (2 if bidirectional else 1), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x) # h_n shape: (num_layers, batch_size, hidden_size) 
        h_n = h_n.movedim(1, 0) # h_n shape: (batch_size, num_layers, hidden_size)
        h_n = h_n.flatten(start_dim = 1) # h_n shape: (batch_size, num_layers * hidden_size)
        fc_out = self.fc(h_n)
        return self.sigmoid(fc_out) # use sigmoid to model membership probability
