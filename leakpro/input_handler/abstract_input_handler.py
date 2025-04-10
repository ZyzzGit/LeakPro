"""Parent class for user inputs."""

from abc import ABC, abstractmethod

import numpy as np
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from leakpro.schemas import TrainingOutput


class AbstractInputHandler(ABC):
    """Parent class for user inputs."""

    def __init__(self, configs: dict) -> None:
        self.configs = configs

    @abstractmethod
    def get_criterion(self, criterion: _Loss) -> _Loss:
        """Get the loss function for the target model to be used in model training."""
        pass

    @abstractmethod
    def get_optimizer(self, model:Module) -> Optimizer:
        """Get the optimizer used for the target model to be used in model training."""
        pass

    @abstractmethod
    def train(
        self,
        dataloader: DataLoader,
        model: Module,
        criterion: _Loss,
        optimizer: Optimizer
    ) -> TrainingOutput:
        """Procedure to train a model on data from the population."""
        pass

    def sample_shadow_indices(
        self, 
        shadow_population:list, 
        data_fraction:float
    ) -> np.ndarray:
        """
        Procedure to sample shadow model indices.
        May be overriden in Handler implementation to perform custom sampling; 
        see for example sampling by individual in examples/mia/time_series_mia/data_handler.py
        """
        data_size = int(len(shadow_population)*data_fraction)
        data_indices = np.random.choice(shadow_population, data_size, replace=False)
        return data_indices
