"""Signal class, which is an abstract class representing any type of signal that can be obtained."""

from abc import ABC, abstractmethod

import os
import numpy as np
from numpy.fft import fft
from numpy.linalg import inv, norm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from ts2vec import TS2Vec
from torch import cuda

from leakpro.utils.logger import logger
from leakpro.signals.utils.TS2VecTrainer import train_ts2vec
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.signals.signal_extractor import Model
from leakpro.utils.import_helper import List, Optional, Self, Tuple
from sktime.distances import dtw_distance

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
        "TS2VecLoss": TS2VecLoss()
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
                mase_loss = np.divide(me, shifted_me + 1e-30)

                model_mase_loss.extend(mase_loss)

            model_mase_loss = np.array(model_mase_loss)
            results.append(model_mase_loss)

        return results
    
class TS2VecLoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the TS2Vec loss of a time-series model output.
    We define this as the distance (L2 norm) between the TS2Vec representations of the true and predicted values.
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
        _, _, num_variables = handler.population.y.shape

        # Compute the signal for each model
        data_loader = handler.get_dataloader(indices, batch_size=batch_size)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        # Check if representation model is available
        ts2vec_model_path = 'data/ts2vec_model.pkl'
        if not os.path.exists(ts2vec_model_path):
            logger.info("Training TS2Vec representation model")
            ts2vec_train_indices = np.concatenate([handler.train_indices, handler.test_indices])
            ts2vec_train_data = handler.population.y[ts2vec_train_indices]
            train_ts2vec(ts2vec_train_data, num_variables)

        # Load represenation model
        device = "cuda:0" if cuda.is_available() else "cpu"
        ts2vec_model = TS2Vec(
            input_dims=num_variables,
            device=device,
            batch_size=batch_size
        )
        ts2vec_model.load(ts2vec_model_path)

        results = []
        for m, model in enumerate(models):
            # Initialize a matrix to store the TS2Vec loss for the current model
            model_ts2vec_loss = []

            for data, target in tqdm(data_loader, desc=f"Getting TS2Vec loss for model {m+1}/ {len(models)}"):
                # Get the TS2Vec encodings and compute L2 norm between true and pred
                output = model.get_logits(data)
                ts2vec_pred = ts2vec_model.encode(output, encoding_window='full_series')
                ts2vec_true = ts2vec_model.encode(target.numpy(), encoding_window='full_series')
                ts2vec_loss = norm(ts2vec_true - ts2vec_pred, axis=1)
                model_ts2vec_loss.extend(ts2vec_loss)

            model_ts2vec_loss = np.array(model_ts2vec_loss)
            results.append(model_ts2vec_loss)

        return results
    
class DTWDistance(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the Dynamic Time Warping distance between a time-series model output and the target series.
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
            # Initialize a matrix to store the DTW distances for the current model
            model_dtw_distance = []

            for data, target in tqdm(data_loader, desc=f"Getting DTW distance for model {m+1}/ {len(models)}"):
                # Get the DTW distances for batch
                output = model.get_logits(data)
                batch_dtw_distances = np.array(list(map(dtw_distance, target.numpy(), output)))
                model_dtw_distance.extend(batch_dtw_distances)

            model_dtw_distance = np.array(model_dtw_distance)
            results.append(model_dtw_distance)

        return results