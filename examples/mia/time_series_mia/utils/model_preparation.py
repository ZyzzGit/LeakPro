import os
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from torch import optim, cuda, no_grad, save


class SimpleLSTM(nn.Module):
    """Single layer LSTM for multi-variate forecasting"""
    
    def __init__(self, input_size, horizon):
        super().__init__()
        self.init_params = {"input_size": input_size,
                            "horizon": horizon}
        self.input_size = input_size
        self.horizon = horizon
        self.hidden_size = 64

        self.lstm = nn.LSTM(input_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, input_size * horizon)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x) # h_n shape: (num_layers, batch_size, hidden_size) 
        linear_out = self.linear(h_n[0])   
        return linear_out.view(-1, self.horizon, self.input_size)   # reshape to (batch_size, horizon, num_variables) 

def evaluate(model, loader, criterion, device):
    model.eval()
    loss = 0
    with no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data)
            loss += criterion(pred, target).item()
        loss /= len(loader)
    return loss

def create_trained_model_and_metadata(model, train_loader, test_loader, epochs):
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    train_losses, test_losses = [], []
    
    for _ in tqdm(range(epochs), desc="Training Progress"):
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(data)
            
            loss = criterion(pred, target)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        test_loss = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)

    # Move the model back to the CPU
    model.to("cpu")
    if not os.path.exists("target"):
        os.makedirs("target")
    with open("target/target_model.pkl", "wb") as f:
        save(model.state_dict(), f)

    # Create metadata and store it
    meta_data = {}
    meta_data["train_indices"] = train_loader.dataset.indices
    meta_data["test_indices"] = test_loader.dataset.indices
    meta_data["num_train"] = len(meta_data["train_indices"])
    
    # Write init params
    meta_data["init_params"] = {}
    for key, value in model.init_params.items():
        meta_data["init_params"][key] = value
    
    # read out optimizer parameters
    meta_data["optimizer"] = {}
    meta_data["optimizer"]["name"] = optimizer.__class__.__name__.lower()
    meta_data["optimizer"]["lr"] = optimizer.param_groups[0].get("lr", 0)
    meta_data["optimizer"]["weight_decay"] = optimizer.param_groups[0].get("weight_decay", 0)
    meta_data["optimizer"]["momentum"] = optimizer.param_groups[0].get("momentum", 0)
    meta_data["optimizer"]["dampening"] = optimizer.param_groups[0].get("dampening", 0)
    meta_data["optimizer"]["nesterov"] = optimizer.param_groups[0].get("nesterov", False)

    # read out criterion parameters
    meta_data["loss"] = {}
    meta_data["loss"]["name"] = criterion.__class__.__name__.lower()

    meta_data["batch_size"] = train_loader.batch_size
    meta_data["epochs"] = epochs
    meta_data["train_loss"] = train_loss
    meta_data["test_loss"] = test_loss
    meta_data["dataset"] = "ECG"
    
    with open("target/model_metadata.pkl", "wb") as f:
        pickle.dump(meta_data, f)
    
    return train_losses, test_losses
