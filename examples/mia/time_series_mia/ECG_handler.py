"""Module containing the class to handle the user input for the Georgia 12-Lead ECG dataset."""

import torch
import random
import numpy as np
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
    
    def sample_shadow_indices(self, shadow_population:list, data_size:int) -> np.ndarray:
        """Samples data indices from shadow population by individuals"""
        population_individuals = self.population.individual_indices
        shadow_individuals = []
        for ind in population_individuals:
            start, end = ind
            if start in shadow_population and end-1 in shadow_population:   # assumes we either have all or no samples from ind in shadow population
                shadow_individuals.append(ind)

        individual_length = self.population.num_samples_per_individual
        num_shadow_individuals = data_size // individual_length # full num. individuals needed
        num_remaining_samples = data_size % individual_length   # remaining num. samples to get data_size

        # Sample individuals and extract corresponding dataset indices
        sampled_shadow_individuals = random.sample(shadow_individuals, num_shadow_individuals + 1)
        sampled_indices = np.concatenate([np.arange(start, stop) for (start, stop) in sampled_shadow_individuals[:-1]], axis=0)

        # If required, use indices from an extra individual to get remaining samples
        if (num_remaining_samples > 0):
            start, stop = sampled_shadow_individuals[-1]
            extra_samples = np.arange(start, stop)[:num_remaining_samples]
            sampled_indices = np.concatenate(sampled_indices, extra_samples)

        np.random.shuffle(sampled_indices)  # shuffle again to get random index order
        return sampled_indices