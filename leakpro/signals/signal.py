"""Signal class, which is an abstract class representing any type of signal that can be obtained."""

from abc import ABC, abstractmethod

import os
import numpy as np
from numpy.fft import fft
from numpy.linalg import inv, norm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm
from leakpro.utils.save_load import hash_model

from leakpro.utils.logger import logger
from leakpro.signals.utils.get_TS2Vec import get_ts2vec_model
from leakpro.signals.utils.msm import mv_msm_distance
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.signals.signal_extractor import Model
from leakpro.utils.import_helper import List, Optional, Self, Tuple
from sktime.distances import dtw_distance

def get_signal_from_name(signal_name):
    return {
        "ModelLogits": ModelLogits,
        "ModelRescaledLogits": ModelRescaledLogits,
        "ModelLoss": ModelLoss,
        "HopSkipJumpDistance": HopSkipJumpDistance,
        "SeasonalityLoss": SeasonalityLoss,
        "TrendLoss": TrendLoss,
        "MSELoss": MSELoss,
        "TS2VecLoss": TS2VecLoss,
        "SMAPELoss": SMAPELoss,
        "RescaledSMAPELoss": RescaledSMAPELoss,
        "MAELoss": MAELoss,
    }[signal_name]()

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
    
    def get_model_output(  # noqa: ANN204
        self: Self,
        model: Model,
        handler: AbstractInputHandler,
        indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        model_hash = hash_model(model.model_obj)
        output_dir = handler.configs.audit.output_dir
        os.makedirs(os.path.join(output_dir, "pred_cache"), exist_ok=True)
        output_path = os.path.join(output_dir, "pred_cache", model_hash + ".npy")
        if os.path.isfile(output_path):
            return np.load(output_path)[indices], np.array(handler.population.targets)[indices]
        else:
            data_loader = handler.get_dataloader(np.arange(handler.population_size), shuffle=False)
            assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"
            model_logits = []
            for data, _ in data_loader:
                # Get logits for each data point
                logits = model.get_logits(data)
                model_logits.extend(logits)
            model_logits = np.array(model_logits)
            np.save(output_path, model_logits)
            return model_logits[indices], np.array(handler.population.targets)[indices]


class ModelLogits(Signal):
    """Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the output of a model.
    """

    def __call__(
        self: Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.

        Returns:
        -------
            The signal value.

        """        # Compute the signal for each model

        results = []
        for model in tqdm(models, desc="Getting Model Logits"):
            model_outputs, _ = self.get_model_output(model, handler, indices)
            results.append(model_outputs)
        return np.array(results)

class ModelRescaledLogits(Signal):
    """Inherits from the Signal class, used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the output of a model.
    """

    def __call__(
        self: Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.

        Returns:
        -------
            The signal value.

        """
        data_loader = handler.get_dataloader(indices, shuffle=False)
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
    ) -> List[np.ndarray]:
        """Built-in call method.

        Args:
        ----
            models: List of models that can be queried.
            handler: The input handler object.
            indices: List of indices in population dataset that can be queried from handler.

        Returns:
        -------
            The signal value.

        """
        # Compute the signal for each model
        data_loader = handler.get_dataloader(indices, shuffle=False)
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
        
        results = []
        for model in tqdm(models, desc="Getting Seasonality loss"):
            model_outputs, model_targets = self.get_model_output(model, handler, indices)

            seasonality_pred = get_seasonality_coefficients(model_outputs)
            seasonality_true = get_seasonality_coefficients(model_targets)
            seasonality_loss = norm(seasonality_true - seasonality_pred, axis=(1, 2))
            results.append(seasonality_loss)
        return np.array(results)
    
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
        results = []
        for model in tqdm(models, desc="Getting Trend loss"):
            model_outputs, model_targets = self.get_model_output(model, handler, indices)

            trend_pred = get_trend_coefficients(model_outputs)
            trend_true = get_trend_coefficients(model_targets)
            trend_loss = norm(trend_true - trend_pred, axis=(1, 2))
            results.append(trend_loss)
        return np.array(results)

class MSELoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the per-sample MSE loss of a time-series model output.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
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

        results = []
        for model in tqdm(models, desc="Getting MSE loss"):
            model_outputs, model_targets = self.get_model_output(model, handler, indices)
            model_mse_loss = np.mean(np.square(model_outputs - model_targets), axis=(1,2))
            results.append(model_mse_loss)
        return np.array(results)
    
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
        shadow_population_indices: np.ndarray
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
        # Get represenation model
        batch_size = handler.get_dataloader(indices, shuffle=False).batch_size
        ts2vec_model = get_ts2vec_model(handler, shadow_population_indices, batch_size)

        # Get signals
        logger.info("Getting TS2Vec loss for targets")
        ts2vec_true = ts2vec_model.encode(np.array(handler.population.targets)[indices], encoding_window='full_series', batch_size=batch_size)
        logger.info("Getting TS2Vec loss for model outputs")
        results = []
        for model in tqdm(models, desc="Getting TS2Vec loss"):
            model_outputs, _ = self.get_model_output(model, handler, indices)
            ts2vec_pred = ts2vec_model.encode(model_outputs, encoding_window='full_series', batch_size=batch_size)
            ts2vec_loss = norm(ts2vec_true - ts2vec_pred, axis=1)
            results.append(ts2vec_loss)

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
        data_loader = handler.get_dataloader(indices, shuffle=False)
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
    
class MSMDistance(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the Move-Split-Merge distance between a time-series model output and the target series.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
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
        data_loader = handler.get_dataloader(indices, shuffle=False)
        assert self._is_shuffling(data_loader) is False, "DataLoader must not shuffle data to maintain order of indices"

        results = []
        for m, model in enumerate(models):
            # Initialize a matrix to store the MSM distances for the current model
            model_msm_distance = []

            for data, target in tqdm(data_loader, desc=f"Getting MSM distance for model {m+1}/ {len(models)}"):
                # Get the MSM distances for batch
                output = model.get_logits(data)
                batch_msm_distances = np.array(list(map(mv_msm_distance, target.numpy(), output)))
                model_msm_distance.extend(batch_msm_distances)

            model_msm_distance = np.array(model_msm_distance)
            results.append(model_msm_distance)

        return results

class SMAPELoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the per-sample MASE loss of a time-series model output.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
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

        results = []
        for model in tqdm(models, desc="Getting SMAPE loss"):
            model_outputs, model_targets = self.get_model_output(model, handler, indices)
            
            numerator = np.abs(model_outputs - model_targets) 
            denominator = np.abs(model_outputs) + np.abs(model_targets) + 1e-30
            fraction = numerator / denominator
            smape_loss = np.mean(fraction, axis=(1,2))

            results.append(smape_loss)
        return np.array(results)

    
class RescaledSMAPELoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the per-sample MASE loss of a time-series model output.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
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

        results = []
        for model in tqdm(models, desc="Getting Rescaled SMAPE loss"):
            model_outputs, model_targets = self.get_model_output(model, handler, indices)
            
            numerator = np.abs(model_outputs - model_targets) 
            denominator = np.abs(model_outputs) + np.abs(model_targets) + 1e-30
            fraction = numerator / denominator
            smape_loss = np.mean(fraction, axis=(1,2))
            rescaled_smape = np.log(smape_loss / (1 - smape_loss + 1e-30))

            results.append(rescaled_smape)
        return np.array(results)
    
class MAELoss(Signal):
    """Used to represent any type of signal that can be obtained from a Model and/or a Dataset.

    This particular class is used to get the per-sample MAE loss of a time-series model output.
    """

    def __call__(
        self:Self,
        models: List[Model],
        handler: AbstractInputHandler,
        indices: np.ndarray,
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
        results = []
        for model in tqdm(models, desc="Getting MAE loss"):
            model_outputs, model_targets = self.get_model_output(model, handler, indices)
            model_mae_loss = np.mean(np.abs(model_outputs - model_targets), axis=(1,2))
            results.append(model_mae_loss)
        return np.array(results)