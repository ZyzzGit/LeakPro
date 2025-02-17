"""Module for handling generators."""
import joblib
from torch.nn import Module
from torch.utils.data import DataLoader

from leakpro.attacks.utils.model_handler import ModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class GeneratorHandler(ModelHandler):
    """Base class for generator models like GANs, diffusion models, etc."""

    def __init__(self: Self, handler: AbstractInputHandler, caller: str = "generator") -> None:
        """Initialize the GeneratorHandler base class."""
        super().__init__(handler, caller)
        self._setup_generator_configs()
        logger.info(f"Initialized GeneratorHandler with caller: {caller}")


    def _setup_generator_configs(self) -> None:
        """Load generator-specific configurations (e.g., generator path, params)."""
        self.generator_path = self.handler.configs.get("generator", {}).get("module_path")
        self.generator_class = self.handler.configs.get("generator", {}).get("model_class")
        self.gen_init_params = self.handler.configs.get("generator", {}).get("init_params", {})

        if self.generator_path and self.generator_class:
            self.generator_blueprint = self._import_model_from_path(self.generator_path, self.generator_class)
        else:
            raise ValueError("Generator path and class must be specified in the config.")

    def get_generator(self) -> Module:
        """Instantiate and return a generator model."""
        return self.generator_blueprint(**self.gen_init_params)

    def train(self) -> None:
        """Abstract training method for generator models. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the train method")

    def generate_samples(self) -> None:
        """Abstract sample generation method for generator models. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the generate_samples method")

    def get_public_data(self, batch_size: int) -> DataLoader:
        """Return data loader for the public dataset."""
        # Get public dataloader
        self.public_path = self.handler.configs.get("public_data_path")
        # Load pickle file
        try:
            with open(self.public_path, "rb") as f:
                self.public_dataset = joblib.load(f)
            logger.info(f"Loaded public data from {self.public_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find the public data at {self.public_path}") from e
        return DataLoader(self.public_dataset, batch_size = batch_size, shuffle=False)