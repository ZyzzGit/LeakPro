"""Implementation of the Give me data method from "Improving Membership Inference Attacks against Classification Models"."""

import numpy as np



from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import MIAResult
from leakpro.signals.signal import TrendLoss, SeasonalityLoss, MSELoss, MASELoss
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackGimmeData(AbstractMIA):
    """Implementation of the Give me data attack."""

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the Give me data attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(handler)
        self.epsilon = 1e-6
        self.shadow_models = None
        self.shadow_model_indices = None

        logger.info("Configuring Give me data attack")
        self._configure_attack(configs)


    def _configure_attack(self:Self, configs: dict) -> None:
        """Configure the Give me data attack.

        Args:
        ----
            configs (dict): Configuration parameters for the attack.

        """
        self.num_shadow_models = configs.get("num_shadow_models", 50) 
        self.training_data_fraction = configs.get("training_data_fraction", 0.5)
        self.attack_data_fraction = configs.get("attack_data_fraction", 0.1)
        self.online = configs.get("online", True) 
        self.signals = [TrendLoss(), SeasonalityLoss(), MSELoss(), MASELoss()] # Change this to be configurable


        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "num_shadow_models": (self.num_shadow_models, 1, None),
            "training_data_fraction": (self.training_data_fraction, 0, 1),
            "attack_data_fraction": (self.attack_data_fraction, 0, 1),
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Give me data attack"
        reference_str = "Shlomit Shachor, Natalia Razinkov, Abigail Goldsteen and Ariel Farkash. \
            Improving Membership Inference Attacks against Classification Models. (2024)."
        summary_str = "The Give me data attack is a membership inference attack based on an Give me data of classifications models."
        detailed_str = "The attack is executed according to: \
            1. The shadow model training dataset is split into multiple non-overlapping subsets. \
            2. A set amount of pairs created using these subsets. \
            3. For multiple runs we randomly assign membership label to all datapoints in a pair.\
            4. For each run run multiple combinations of classification models. \
            5. i dont even know. "
        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self) -> None:
        """Prepare data needed for running the attack on the target model and dataset.

        Signals are computed on the auxiliary model(s) and dataset.
        """

        logger.info("Preparing shadow models for Give me data attack")
        # Check number of shadow models that are available

        # sample dataset to compute histogram
        logger.info("Preparing attack data for training the Give me data attack")

        # Get all available indices for attack dataset including training and test data
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)
        logger.info(f"{len(self.attack_data_indices)=}")

        # train shadow models
        logger.info(f"Check for {self.num_shadow_models} shadow models (dataset: {len(self.attack_data_indices)} points)")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = self.num_shadow_models,
            shadow_population = self.attack_data_indices,
            training_fraction = self.training_data_fraction,
            online = self.online)
        # load shadow models
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.attack_data_indices).T
        in_indices_masks = in_indices_masks.swapaxes(0, -1)


        features = [np.swapaxes(signal(self.shadow_models,
                                        self.handler,
                                        self.attack_data_indices), 0, 1)
                    for signal in self.signals]
        
        features = np.array(features)
        features = np.swapaxes(features, 0, -1)
        features = np.swapaxes(features, 0, 1)

        np.save("features", features)
        np.save("mask", in_indices_masks)



        logger.info(in_indices_masks.shape)
        logger.info(features.shape)
            


    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """

        # compute ROC, TP, TN etc
        return MIAResult(
            predicted_labels=None,
            true_labels=None,
            predictions_proba=None,
            signal_values=None,
        )


