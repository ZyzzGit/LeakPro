"""Module containing the class to handle the user input for the Georgia 12-Lead ECG or TUH-EEG dataset."""

import torch
import random
import numpy as np
from torch import cuda, optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from leakpro.schemas import TrainingOutput

from leakpro import AbstractInputHandler

class IndividualizedInputHandler(AbstractInputHandler):
    """Class to handle the user input for the Georgia 12-Lead ECG or TUH-EEG dataset."""

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
    ) -> TrainingOutput:
        """Model training procedure."""

        # read hyperparams for training (the parameters for the dataloader are defined in get_dataloader):
        if epochs is None:
            raise ValueError("epochs not found in configs")

        # prepare training
        device = torch.device("cuda" if cuda.is_available() else "cpu")
        model.to(device)

        # training loop
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            train_loss = 0
            model.train()
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                optimizer.zero_grad()
                preds = model(inputs)
                loss = criterion(preds, targets)
                loss.backward()
                optimizer.step()

                # Accumulate performance of shadow model
                train_loss += loss.item()

        model.to("cpu")

        output_dict = {"model": model, "metrics": {"loss": train_loss}}
        output = TrainingOutput(**output_dict)
        return output
    
    def sample_shadow_indices(self, shadow_population:list, data_fraction:float) -> np.ndarray:
        """Samples data indices from shadow population by individuals"""
        population_individuals = self.population.individual_indices
        shadow_individuals = []
        for ind in population_individuals:
            start, end = ind
            if start in shadow_population and end-1 in shadow_population:   # assumes we either have all or no samples from ind in shadow population
                shadow_individuals.append(ind)

        num_individuals = self.population.num_individuals
        num_shadow_individuals = int(num_individuals * data_fraction)

        # Sample individuals and extract corresponding dataset indices
        sampled_shadow_individuals = random.sample(shadow_individuals, num_shadow_individuals)
        sampled_indices = np.concatenate([np.arange(start, stop) for (start, stop) in sampled_shadow_individuals], axis=0)

        np.random.shuffle(sampled_indices)  # shuffle again to get random index order
        return sampled_indices