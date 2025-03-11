import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch import nn, optim, cuda, no_grad, save

def predict(model, loader, device, original_scale=False):
    model.eval()
    model.to(device)
    all_targets = []
    all_preds = []
    with no_grad():
        for data, target in loader:                
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            if original_scale:
                pred_2D = pred.reshape(-1, pred.shape[-1])
                target_2D = target.reshape(-1, target.shape[-1])

                pred_descaled = loader.dataset.dataset.scaler.inverse_transform(pred_2D)
                target_descaled = loader.dataset.dataset.scaler.inverse_transform(target_2D)

                pred = pred_descaled.reshape(pred.shape)
                target = target_descaled.reshape(target.shape)
            all_preds.append(pred)
            all_targets.append(target)
    return np.concatenate(all_targets), np.concatenate(all_preds)

def evaluate(model, loader, criterion, device, original_scale=False):
    model.eval()
    model.to(device)
    loss = 0
    with no_grad():
        for data, target in loader:                
            data, target = data.to(device), target.to(device)
            pred = model(data)
            if original_scale:
                pred_2D = pred.detach().cpu().numpy().reshape(-1, pred.shape[-1])
                target_2D = target.detach().cpu().numpy().reshape(-1, target.shape[-1])

                pred_descaled = loader.dataset.dataset.scaler.inverse_transform(pred_2D)
                target_descaled = loader.dataset.dataset.scaler.inverse_transform(target_2D)

                pred = torch.tensor(pred_descaled, device=device).reshape(pred.shape)
                target = torch.tensor(target_descaled, device=device).reshape(target.shape)

            loss += criterion(pred, target).item()
        loss /= len(loader)
    return loss

def create_trained_model_and_metadata(model, train_loader, test_loader, epochs, optimizer_name):
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    criterion = nn.MSELoss()
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters())
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters())
    else:
        raise NotImplementedError()
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
