"""Module containing the class to handle the user input for the Georgia 12-Lead ECG dataset."""

import torch
from torch import cuda, optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm

from leakpro import AbstractInputHandler

class ECGInputHandler(AbstractInputHandler):
    """Class to handle the user input for the Georgia 12-Lead ECG dataset."""

    def __init__(self, configs: dict) -> None:
        super().__init__(configs = configs)
        print(configs)

    def get_criterion(self)->None:
        """Set the MSELoss for the model."""
        return MSELoss()

    def get_optimizer(self, model:torch.nn.Module) -> None:
        """Set the optimizer for the model."""
        return optim.Adam(model.parameters())

    def train(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module = None,
        criterion: torch.nn.Module = None,
        optimizer: optim.Optimizer = None,
        epochs: int = None,
    ) -> dict:
        """Model training procedure."""

        # read hyperparams for training (the parameters for the dataloader are defined in get_dataloader):
        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        device = torch.device("cuda" if cuda.is_available() else "cpu")
        model.to(device)

        # training loop
        for epoch in range(epochs):
            train_loss = 0
            model.train()
            for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                preds = model(inputs)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

                # Accumulate performance of shadow model
                train_loss += loss.item()

        model.to("cpu")
        return {"model": model, "metrics": {"loss": train_loss, "accuracy": None}}
