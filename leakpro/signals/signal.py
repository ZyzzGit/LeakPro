"""Signal class, which is an abstract class representing any type of signal that can be obtained."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.fft import fft
from numpy.linalg import inv, norm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.signals.signal_extractor import Model
from leakpro.utils.import_helper import List, Optional, Self, Tuple

def get_signal_from_name(signal_name):
    return {
        "ModelLogits": ModelLogits(),
        "ModelRescaledLogits": ModelRescaledLogits(),
        "ModelLoss": ModelLoss(),
        "HopSkipJumpDistance": HopSkipJumpDistance(),
        "SeasonalityLoss": SeasonalityLoss(),
        "TrendLoss": TrendLoss(),
        "MSELoss": MSELoss(),
        "MASELoss": MASELoss(),
    }[signal_name]

class Signal(ABC):
    """Abstract class, representing any type of signal that can be obtained from a Model and/or a Dataset."""

    @abstractmethod
    def __call__(  # noqa: ANN204
        self: Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        extra: dict,
    ):
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            extra: Dictionary containing any additional parameter that should be passed to the signal object.

        Returns:
        -------
            The signal value.

        """
        pass

    def _is_shuffling(self:Self, dataloader:DataLoader)->bool:
        """Check if the DataLoader is shuffling the data."""
        return not isinstance(dataloader.sampler, SequentialSampler)


class ModelLogits(Signal):
    """Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the output of a model.
    """

    def __call__(
        self: Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            batch_size: Integer to determine batch size for dataloader.

        Returns:
        -------
            The signal value.

        """        # Compute the signal for each model

        # Iterate over the DataLoader (ensures we use transforms etc)
        # NOTE: Shuffle must be false to maintain indices order
        data_loader = handler.get_dataloader(indices, batch_size=batch_size)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a list to store the logits sfor the current model
            model_logits = []

            for data, _ in tqdm(data_loader, desc=f"Getting logits for model {m+1}/ {len(models)}", leave=False):
                # Get logits for each data point
                logits = model.get_logits(data)
                model_logits.extend(logits)
            model_logits = np.array(model_logits)
            # Append the logits for the current model to the results
            results.append(model_logits)

        return results

class ModelRescaledLogits(Signal):
    """Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the output of a model.
    """

    def __call__(
        self: Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            batch_size: Integer to determine batch size for dataloader.

        Returns:
        -------
            The signal value.

        """
        data_loader = handler.get_dataloader(indices, batch_size=batch_size)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a list to store the logits for the current model
            model_logits = []

            for data, labels in tqdm(data_loader, desc=f"Getting rescaled logits for model {m+1}/ {len(models)}", leave=False):
                # Get logits for each data point
                logits = model.get_rescaled_logits(data,labels)
                model_logits.extend(logits)
            model_logits = np.array(model_logits)
            # Append the logits for the current model to the results
            results.append(model_logits)

        return results

class ModelLoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the loss of a model.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            batch_size: Integer to determine batch size for dataloader.

        Returns:
        -------
            The signal value.

        """
        # Compute the signal for each model
        data_loader = handler.get_dataloader(indices, batch_size=batch_size)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a list to store the logits for the current model
            model_logits = []

            for data, labels in tqdm(data_loader, desc=f"Getting loss for model {m+1}/ {len(models)}"):
                # Get logits for each data point
                loss = model.get_loss(data,labels)
                model_logits.extend(loss)
            model_logits = np.array(model_logits)
            # Append the logits for the current model to the results
            results.append(model_logits)

        return results

class HopSkipJumpDistance(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the hop skip jump distance of a model.
    """

    def __call__(  # noqa: D102
        self:Self,
        model: Model,
        data_loader: DataLoader,
        norm: int = 2,
        y_target: Optional[int] = None,
        image_target: Optional[int] = None,
        initial_num_evals: int = 100,
        max_num_evals: int = 10000,
        stepsize_search: str = "geometric_progression",
        num_iterations: int = 100,
        gamma: float = 1.0,
        constraint: int = 2,
        batch_size: int = 128,
        epsilon_threshold: float = 1e-6,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Built-in call method.

        Args:
        ----
            model: The model to be used.
            data_loader: The data loader to load the data.
            norm: The norm to be used for distance calculation.
            y_target: The target class label (optional).
            image_target: The target image index (optional).
            initial_num_evals: The initial number of evaluations.
            max_num_evals: The maximum number of evaluations.
            stepsize_search: The step size search strategy.
            num_iterations: The number of iterations.
            gamma: The gamma value.
            constraint: The constraint value.
            batch_size: The batch size.
            epsilon_threshold: The epsilon threshold.
            verbose: Whether to print verbose output.

        Returns:
        -------
            Tuple containing the perturbed images and perturbed distance.

        """


        # Compute the signal for each model
        perturbed_imgs, perturbed_distance = model.get_hop_skip_jump_distance(
                                                    data_loader,
                                                    norm,
                                                    y_target,
                                                    image_target,
                                                    initial_num_evals,
                                                    max_num_evals,
                                                    stepsize_search,
                                                    num_iterations,
                                                    gamma,
                                                    constraint,
                                                    batch_size,
                                                    epsilon_threshold,
                                                    verbose
                                                    )

        return perturbed_imgs, perturbed_distance
    
def get_seasonality_coefficients(Y):
    Z = fft(Y, axis=2) # column-wise 1D-DFT over variables
    C = fft(Z, axis=1) # row-wise 1D-DFT over horizon
    return C

class SeasonalityLoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the seasonality loss of a time-series model output.
    We define this as the distance (L2 norm) between the true and predicted values for the seasonality component.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            batch_size: Integer to determine batch size for dataloader.

        Returns:
        -------
            The signal value.

        """
        # Compute the signal for each model
        data_loader = handler.get_dataloader(indices, batch_size=batch_size)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a matrix to store the seasonality loss for the current model
            model_seasonality_loss = []

            for data, target in tqdm(data_loader, desc=f"Getting seasonality loss for model {m+1}/ {len(models)}"):
                # Get the output seasonality and compute L2 norm wrt true seasonality
                output = model.get_logits(data)
                seasonality_pred = get_seasonality_coefficients(output)
                seasonality_true = get_seasonality_coefficients(target)
                seasonality_loss = norm(seasonality_true - seasonality_pred, axis=(1, 2))
                model_seasonality_loss.extend(seasonality_loss)

            model_seasonality_loss = np.array(model_seasonality_loss)
            results.append(model_seasonality_loss)

        return results
    
def get_trend_coefficients(Y, polynomial_degree=4):
    horizon = Y.shape[1]
    t = np.arange(horizon) / horizon
    P = np.vander(t, polynomial_degree, increasing=True)    # Vandermonde matrix of specified degree
    A = inv(P.T @ P) @ P.T @ Y                              # least squares solution to polynomial fit
    return A
    
class TrendLoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the trend loss of a time-series model output.
    We define this as the distance (L2 norm) between the true and predicted values for the trend component.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            batch_size: Integer to determine batch size for dataloader.

        Returns:
        -------
            The signal value.

        """
        # Compute the signal for each model
        data_loader = handler.get_dataloader(indices, batch_size=batch_size)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a matrix to store the trend loss for the current model
            model_trend_loss = []

            for data, target in tqdm(data_loader, desc=f"Getting trend loss for model {m+1}/ {len(models)}"):
                # Get the output trend and compute L2 norm wrt true trend
                output = model.get_logits(data)
                trend_pred = get_trend_coefficients(output)
                trend_true = get_trend_coefficients(target.numpy())
                trend_loss = norm(trend_true - trend_pred, axis=(1, 2))
                model_trend_loss.extend(trend_loss)

            model_trend_loss = np.array(model_trend_loss)
            results.append(model_trend_loss)

        return results
    

class MSELoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the per-sample MSE loss of a time-series model output.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            batch_size: Integer to determine batch size for dataloader.

        Returns:
        -------
            The signal value.

        """
        # Compute the signal for each model
        data_loader = handler.get_dataloader(indices, batch_size=batch_size)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a matrix to store the MSE loss for the current model
            model_mse_loss = []

            for data, target in tqdm(data_loader, desc=f"Getting MSE loss for model {m+1}/ {len(models)}"):
                output = model.get_logits(data)
                target = target.numpy()
                mse_loss = np.mean(np.square(output - target), axis=(1,2))
                model_mse_loss.extend(mse_loss)

            model_mse_loss = np.array(model_mse_loss)
            results.append(model_mse_loss)

        return results
    
class MASELoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the per-sample MASE loss of a time-series model output.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
        batch_size: int = 32,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.
            batch_size: Integer to determine batch size for dataloader.

        Returns:
        -------
            The signal value.

        """
        # Compute the signal for each model
        data_loader = handler.get_dataloader(indices, batch_size=batch_size)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a matrix to store the MASE loss for the current model
            model_mase_loss = []

            for data, target in tqdm(data_loader, desc=f"Getting MASE loss for model {m+1}/ {len(models)}"):
                output = model.get_logits(data)
                target = target.numpy()
                me = np.mean(np.abs(output - target), axis=(1,2))
                shifted_me = np.mean(np.abs(target[:, 1:, :] - target[:, :-1, :]), axis=(1,2))
                mase_loss = np.divide(me, shifted_me)

                model_mase_loss.extend(mase_loss)

            model_mase_loss = np.array(model_mase_loss)
            results.append(model_mase_loss)

        return results