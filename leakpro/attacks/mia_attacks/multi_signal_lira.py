"""Implementation of a multi-signal version of the LiRA attack."""

import numpy as np
from pydantic import BaseModel, Field, model_validator
from scipy.stats import multivariate_normal
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.metrics.attack_result import MIAResult
from leakpro.signals.signal import get_signal_from_name
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackMSLiRA(AbstractMIA):
    """Implementation of a multi-signal version of the LiRA attack."""

    class AttackConfig(BaseModel):
        """Configuration for the MSLiRA attack."""

        signal_names: list[str] = Field(default=["ModelRescaledLogits"], description="What signals to use.")
        individual_mia: bool = Field(default=False, description="Run individual-level MIA.")
        num_shadow_models: int = Field(default=1, ge=1, description="Number of shadow models")
        training_data_fraction: float = Field(default=0.5, ge=0.0, le=1.0, description="Part of available attack data to use for shadow models")  # noqa: E501
        online: bool = Field(default=False, description="Online vs offline attack")
        eval_batch_size: int = Field(default=32, ge=1, description="Batch size for evaluation")

        @model_validator(mode="after")
        def check_num_shadow_models_if_online(self) -> Self:
            """Check if the number of shadow models is at least 2 when online is True.

            Returns
            -------
                Config: The attack configuration.

            Raises
            ------
                ValueError: If online is True and the number of shadow models is less than 2.

            """
            if self.online and self.num_shadow_models < 2:
                raise ValueError("When online is True, num_shadow_models must be >= 2")
            return self

    def __init__(self:Self,
                 handler: MIAHandler,
                 configs: dict
                 ) -> None:
        """Initialize the MSLiRA attack.

        Args:
        ----
            handler (MIAHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        self.configs = self.AttackConfig() if configs is None else self.AttackConfig(**configs)

        # Initializes the parent metric
        super().__init__(handler)

        # Assign the configuration parameters to the object
        for key, value in self.configs.model_dump().items():
            setattr(self, key, value)

        if self.online is False and self.population_size == self.audit_size:
            raise ValueError("The audit dataset is the same size as the population dataset. \
                    There is no data left for the shadow models.")

        self.shadow_models = []
        self.signals = [get_signal_from_name(signal_name) for signal_name in self.signal_names]

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Multi-Signal Likelihood Ratio Attack"

        reference_str = "Original LiRA: Carlini N, et al. Membership Inference Attacks From First Principles"

        summary_str = "LiRA is a membership inference attack based on rescaled logits of a black-box model. \
        The multi-signal version extends LiRA to attack a model based on multiple signals extracted from the outputs."

        detailed_str = "The attack is executed according to: \
            1. A fraction of the target model dataset is sampled to be included (in-) or excluded (out-) \
            from the shadow model training dataset. \
            2. The attack signals are used to estimate multi-variate Gaussian distributions for in and out members. \
            3. The thresholds are used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."

        return {
            "title_str": title_str,
            "reference": reference_str,
            "summary": summary_str,
            "detailed": detailed_str,
        }

    def prepare_attack(self:Self)->None:
        """Prepares data to obtain metric on the target model and dataset, using signals computed on the auxiliary model/dataset.

        It selects a balanced subset of data samples from in-group and out-group members
        of the audit dataset, prepares the data for evaluation, and computes the signals
        for both shadow models and the target model.
        """

        self.attack_data_indices = self.sample_indices_from_population(include_aux_indices = not self.online,
                                                                       include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(num_models = self.num_shadow_models,
                                                                              shadow_population =  self.attack_data_indices,
                                                                              training_fraction = self.training_data_fraction,
                                                                              online = self.online)

        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

        logger.info("Create masks for all IN and OUT samples")
        self.in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"])

        if self.online:
            # Exclude all audit points that have either no IN or OUT samples
            num_shadow_models_seen_points = np.sum(self.in_indices_masks, axis=1)
            mask = (num_shadow_models_seen_points > 0) & (num_shadow_models_seen_points < self.num_shadow_models)

            # Filter the audit data
            self.audit_data_indices = self.audit_dataset["data"][mask]
            self.in_indices_masks = self.in_indices_masks[mask, :]

            # Filter IN and OUT members
            self.in_members = np.arange(np.sum(mask[self.audit_dataset["in_members"]]))
            num_out_members = np.sum(mask[self.audit_dataset["out_members"]])
            self.out_members = np.arange(len(self.in_members), len(self.in_members) + num_out_members)

            assert len(self.audit_data_indices) == len(self.in_members) + len(self.out_members)

            if len(self.audit_data_indices) == 0:
                raise ValueError("No points in the audit dataset are used for the shadow models")

        else:
            self.audit_data_indices = self.audit_dataset["data"]
            self.in_members = self.audit_dataset["in_members"]
            self.out_members = self.audit_dataset["out_members"]

        # Check offline attack for possible IN- sample(s)
        if not self.online:
            count_in_samples = np.count_nonzero(self.in_indices_masks)
            if count_in_samples > 0:
                logger.info(f"Some shadow model(s) contains {count_in_samples} IN samples in total for the model(s)")
                logger.info("This is not an offline attack!")

        # Calculate all signals for the target and shadow models
        shadow_models_signals = []
        target_model_signals = []
        for signal, signal_name in zip(self.signals, self.signal_names):
            logger.info(f"Calculating {signal_name} for all {self.num_shadow_models} shadow models")
            shadow_models_signals.append(np.swapaxes(signal(self.shadow_models,
                                                                self.handler,
                                                                self.audit_data_indices,
                                                                self.eval_batch_size), 0, 1))

            logger.info(f"Calculating {signal_name} for the target model")
            target_model_signals.append(np.swapaxes(signal([self.target_model],
                                                            self.handler,
                                                            self.audit_data_indices,
                                                            self.eval_batch_size), 0, 1).squeeze())
            
        # Stack signals to get shape (n_audit_points, n_shadow_models, n_signals)
        self.shadow_models_signals = np.stack(shadow_models_signals, axis=-1)
        self.target_model_signals = np.stack(target_model_signals, axis=-1)
    
    def safe_logpdf(x, mean, cov, eps=1e-30):
        try:
            return multivariate_normal.logpdf(x, mean, cov + eps * np.eye(len(mean)))
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix is not positive definite. Using allow_singular=True to compute logpdf.")
            return multivariate_normal.logpdf(x, mean, cov + eps * np.eye(len(mean)), allow_singular=True)

    def run_attack(self:Self) -> MIAResult:
        """Runs the attack on the target model and dataset and assess privacy risks or data leakage.

        This method evaluates how the signals extracted from the target model's output for a specific dataset
        compares to the signals of output of shadow models to determine if the dataset was part of the
        model's training data or not.

        Returns
        -------
        Result(s) of the metric. An object containing the metric results, including predictions,
        true labels, and membership scores.

        """
        n_audit_samples = self.shadow_models_signals.shape[0]
        score = np.zeros(n_audit_samples)  # List to hold the computed probability scores for each sample

        # Iterate over and extract signals for IN and OUT shadow models for each audit sample
        for i, mask in tqdm(enumerate(self.in_indices_masks),
                            total=n_audit_samples,
                            desc="Processing audit samples"):

            # Get the signals from the target and shadow models for the current sample
            target_signals = self.target_model_signals[i]
            out_signals = self.shadow_models_signals[i][~mask]
            in_signals = self.shadow_models_signals[i][mask]

            # Compute OUT statistics
            out_means = np.mean(out_signals, axis=0)
            out_covs = np.cov(out_signals, rowvar=False)
            pr_out = self.safe_logpdf(target_signals, out_means, out_covs)

            if self.online:
                # Compute IN statistics
                in_means = np.mean(in_signals, axis=0)
                in_covs = np.cov(in_signals, rowvar=False)
                pr_in = self.safe_logpdf(target_signals, in_means, in_covs)
            else:
                pr_in = 0

            score[i] = pr_in - pr_out   # Append the calculated probability density value to the score list

            if np.isnan(score[i]):
                raise ValueError("Score is NaN")

        # Split the score array into two parts based on membership: in (training) and out (non-training)
        self.in_member_signals = score[self.in_members].reshape(-1,1)  # Scores for known training data members
        self.out_member_signals = score[self.out_members].reshape(-1,1)  # Scores for non-training data members

        if self.individual_mia:
            samples_per_individual = self.handler.population.samples_per_individual
            in_num_individuals = len(self.in_member_signals) // samples_per_individual 
            out_num_individuals = len(self.out_member_signals) // samples_per_individual
            num_individuals = in_num_individuals + out_num_individuals
            logger.info(f"Running individual-level MI on {num_individuals} individuals with {samples_per_individual} samples per individual.")

            self.in_member_signals = self.in_member_signals.reshape((in_num_individuals, samples_per_individual)).mean(axis=1, keepdims=True)
            self.out_member_signals = self.out_member_signals.reshape((out_num_individuals, samples_per_individual)).mean(axis=1, keepdims=True)
            self.audit_data_indices = np.arange(num_individuals)

        # Prepare true labels array, marking 1 for training data and 0 for non-training data
        true_labels = np.concatenate(
            [np.ones(len(self.in_member_signals)), np.zeros(len(self.out_member_signals))]
        )

        # Combine all signal values for further analysis
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # Return a result object containing predictions, true labels, and the signal values for further evaluation
        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="MS-LiRA")
