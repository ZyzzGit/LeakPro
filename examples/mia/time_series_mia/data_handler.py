"""Module containing the class to handle the user input for the Georgia 12-Lead ECG or TUH-EEG dataset."""

import yaml
import torch
import random
import numpy as np
from torch import cuda, optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from leakpro.schemas import TrainingOutput

from leakpro import AbstractInputHandler
from utils.model_preparation import evaluate
from utils.data_preparation import IndividualizedDataset

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

    class UserDataset(IndividualizedDataset):
        """Conforms to AbstractInputHandler.UserDataset using IndividualizedDataset."""
        def __init__(self, data, targets, *args, **kwargs):
            super().__init__(data, targets, *args, **kwargs)

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
        
        # Check for early stopping configts
        with open("train_config.yaml", 'r') as file:
            train_config = yaml.safe_load(file)
        early_stopping = train_config["train"].get("early_stopping", False)
        patience = train_config["train"].get("patience", 2)
        batch_size = train_config["train"].get("batch_size", 128)

        if early_stopping:
            val_set = dataloader.dataset.dataset.val_set
            val_loader = DataLoader(val_set, batch_size)
            best_val_loss = (-1, np.inf)  # (epoch, validation loss)
            best_state_dict = model.state_dict()

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
            
            if not early_stopping:
                continue

            # Handle early stopping
            val_loss = evaluate(model, val_loader, criterion, device)
            if val_loss < best_val_loss[1]:
                best_val_loss = (epoch, val_loss)
                best_state_dict = model.state_dict()
            elif epoch - best_val_loss[0] > patience:
                print(f"Training stopped early at epoch {epoch+1}.")
                break
        
        # Restore best weights if using early stopping
        if early_stopping:
            model.load_state_dict(best_state_dict)
            print("Best weights restored.")

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

        num_shadow_individuals_to_sample = int(len(shadow_individuals) * data_fraction)

        # Sample individuals and extract corresponding dataset indices
        sampled_shadow_individuals = random.sample(shadow_individuals, num_shadow_individuals_to_sample)
        sampled_indices = np.concatenate([np.arange(start, stop) for (start, stop) in sampled_shadow_individuals], axis=0)

        np.random.shuffle(sampled_indices)  # shuffle again to get random index order
        return sampled_indices