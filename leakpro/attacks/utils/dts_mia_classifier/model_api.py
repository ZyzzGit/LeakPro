import torch
import torch.nn as nn
import numpy as np

import torch
import numpy as np

from torch import nn, optim, cuda
from tqdm import tqdm
from typing import Dict, Any

from leakpro.attacks.utils.dts_mia_classifier.models.lstm_classifier import LSTMClassifier
from leakpro.attacks.utils.dts_mia_classifier.models.inception_time import InceptionTime, DEFAULT_MAX_KERNEL_SIZE

from leakpro.utils.logger import logger


class MIClassifier():

    def __init__(self, seq_len: int, num_input_variables: int, model: str, model_kwargs: Dict[str, Any] = None):
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")
        model_kwargs = model_kwargs or {}

        if model.lower() == "lstm":
            self.model_class = LSTMClassifier
        elif model.lower() == "inceptiontimes": # TODO: Rename IT after new runs
            self.model_class = InceptionTime
        else:
            raise ValueError(f"Unknown model: {model}. Must be one of ['LSTM', 'InceptionTime'].")
        
        # Sanity check for InceptionTime
        if self.model_class == InceptionTime:
            if "fixed_kernel_sizes" in model_kwargs.keys():
                max_kernel_size = max(model_kwargs["fixed_kernel_sizes"])   # fixed_kernel_sizes overrides max_kernel_size
            else:
                max_kernel_size = model_kwargs.get("max_kernel_size", DEFAULT_MAX_KERNEL_SIZE)
            
            if max_kernel_size > seq_len: 
                logger.warning(f"InceptionTime: Maximum kernel size ({max_kernel_size}) is greater than input sequence length ({seq_len}). Consider changing max_kernel_size or specifying fixed_kernel_sizes.")

        self.model = self.model_class(
            num_input_variables,
            **model_kwargs
        )

    def fit(self, train_loader, val_loader, epochs, early_stopping_patience, verbose=0):
        self.model.to(self.device)
        self.model.train()

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters())

        best_val_loss = (-1, np.inf)  # (epoch, validation loss)
        best_state_dict = self.model.state_dict()

        for i in tqdm(range(epochs), desc="Training MI Classifier"):
            self.model.train()
            
            train_loss = 0.0
            correct = 0
            total = 0

            for data, target in train_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                optimizer.zero_grad()
                pred = self.model(data)
                loss = criterion(pred, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Accuracy
                pred_label = (pred >= 0.5).float()
                correct += (pred_label == target).sum().item()
                total += target.numel()
            
            train_loss /= len(train_loader)
            train_acc = correct / total

            val_loss, val_acc = self.evaluate(val_loader, criterion, self.device)
            
            if verbose > 0:
                logger.info(f'Epoch {i+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}')

            if val_loss < best_val_loss[1]:
                best_val_loss = (i, val_loss)
                best_state_dict = self.model.state_dict()
            elif i - best_val_loss[0] > early_stopping_patience:
                logger.info(f"Training stopped early at epoch {i+1}.")
                break
        
        # Restore best weights 
        self.model.load_state_dict(best_state_dict)
        logger.info("Best weights restored.")

    def evaluate(self, loader, criterion, device):
        self.model.eval()
        self.model.to(device)
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in loader:                
                data, target = data.to(device), target.to(device)
                pred = self.model(data)
                total_loss += criterion(pred, target).item()

                # Compute accuracy
                pred_label = (pred >= 0.5).float()
                correct += (pred_label == target).sum().item()
                total += target.numel()

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def predict(self, X_tensor, batch_size):
        device = torch.device("cuda" if cuda.is_available() else "cpu")
        self.model.eval()
        self.model.to(device)
        all_preds = []

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size].to(device)
                preds = self.model(batch).detach().cpu().numpy()
                all_preds.append(preds)

        return np.concatenate(all_preds, axis=0)
