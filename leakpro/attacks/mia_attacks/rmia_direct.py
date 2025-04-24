"""Implementation of the RMIA-Direct attack."""

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.reporting.mia_result import MIAResult
from leakpro.signals.signal import get_signal_from_name
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackRMIADirect(AbstractMIA):
    """Implementation of the RMIA-Direct attack."""

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the RMIA-Direct attack.

        Args:
        ----
            handler (AbstractInputHandler): The input handler object.
            configs (dict): Configuration parameters for the attack.

        """
        # Initializes the parent metric
        super().__init__(handler)
        self.shadow_models = []
        self.epsilon = 1e-6
        self.shadow_models = None
        self.shadow_model_indices = None

        logger.info("Configuring RMIA-Direct attack")
        self._configure_attack(configs)


    def _configure_attack(self:Self, configs: dict) -> None:
        """Configure the RMIA-Direct attack.

        Args:
        ----
            configs (dict): Configuration parameters for the attack.

        """
        self.num_shadow_models = configs.get("num_shadow_models", 4)
        self.gamma = configs.get("gamma", 2.0)
        self.training_data_fraction = configs.get("training_data_fraction", 0.5)
        self.attack_data_fraction = configs.get("attack_data_fraction", 0.1)
        self.online = configs.get("online", True)
        signal_name = configs.get("signal", "ModelRescaledLogits")
        self.signal = get_signal_from_name(signal_name)
        # Determine which variance estimation method to use [carlini, individual_carlini, fixed]
        self.var_calculation = configs.get("var_calculation", "carlini")

        if not self.online:
            raise AttributeError("Only online attack supported")

        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "num_shadow_models": (self.num_shadow_models, 1, None),
            "gamma": (self.gamma, 0, None),
            "training_data_fraction": (self.training_data_fraction, 0, 1),
            "attack_data_fraction": (self.attack_data_fraction, 0, 1),
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "RMIA-Direct attack"
        reference_str = "Zarifzadeh, Sajjad, Philippe Cheng-Jie Marc Liu, and Reza Shokri. \
            Low-Cost High-Power Membership Inference by Boosting Relativity. (2023)."
        summary_str = "The RMIA attack is a membership inference attack based on the output logits of a black-box model."
        detailed_str = "The attack is executed according to: \
            1. A fraction of the population is sampled to compute the likelihood LR_z of p(z|theta) to p(z) for the target model.\
            2. The ratio is used to compute the likelihood ratio LR_x of p(x|theta) to p(x) for the target model. \
            3. The ratio LL_x/LL_z is viewed as a random variable (z is random) and used to classify in-members and out-members. \
            4. The attack is evaluated on an audit dataset to determine the attack performance."
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

        # Fixed variance is used when the number of shadow models is below 32 (64, IN and OUT models)
        #       from (Membership Inference Attacks From First Principles)
        self.fix_var_threshold = 32

        logger.info("Preparing shadow models for RMIA-Direct attack")
        # Check number of shadow models that are available

        # sample dataset to compute histogram
        logger.info("Preparing attack data for training the RMIA-Direct attack")

        # Get all available indices for attack dataset, if self.online = True, include training and test data
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)

        # train shadow models
        logger.info(f"Check for {self.num_shadow_models} shadow models (dataset: {len(self.attack_data_indices)} points)")
        self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
            num_models = self.num_shadow_models,
            shadow_population = self.attack_data_indices,
            training_fraction = self.training_data_fraction,
            online = self.online)
        # load shadow models
        self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)


    def get_std(self:Self, logits: list, mask: list, is_in: bool, var_calculation: str) -> np.ndarray:
        """A function to define what method to use for calculating variance for LiRA."""

        # Fixed/Global variance calculation.
        if var_calculation == "fixed":
            return self._fixed_variance(logits, mask, is_in)

        # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        if var_calculation == "carlini":
            return self._carlini_variance(logits, mask, is_in)

        # Variance calculation as in the paper ( Membership Inference Attacks From First Principles )
        #   but check IN and OUT samples individualy
        if var_calculation == "individual_carlini":
            return self._individual_carlini(logits, mask, is_in)

        return np.array([None])

    def _fixed_variance(self:Self, logits: list, mask: list, is_in: bool) -> np.ndarray:
        if is_in and not self.online:
            return np.array([None])
        return np.std(logits[mask])

    def _carlini_variance(self:Self, logits: list, mask: list, is_in: bool) -> np.ndarray:
        if self.num_shadow_models >= self.fix_var_threshold*2:
            return np.std(logits[mask])
        if is_in:
            return self.fixed_in_std
        return self.fixed_out_std

    def _individual_carlini(self:Self, logits: list, mask: list, is_in: bool) -> np.ndarray:
        if np.count_nonzero(mask) >= self.fix_var_threshold:
            return np.std(logits[mask])
        if is_in:
            return self.fixed_in_std
        return self.fixed_out_std


    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """
        logger.info("Running RMIA online attack")

        # STEP 1: find out which audit data points can actually be audited
        # find the shadow models that are trained on what points in the audit dataset
        in_indices_mask = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"]).T
        # filter out the points that no shadow model has seen and points that all shadow models have seen
        num_shadow_models_seen_points = np.sum(in_indices_mask, axis=0)
        # make sure that the audit points are included in the shadow model training (but not all)
        mask = (num_shadow_models_seen_points > 0) & (num_shadow_models_seen_points < self.num_shadow_models)

        # STEP 2: Select datapoints that are auditable
        audit_data_indices = self.audit_dataset["data"][mask]
        # find out how many in-members survived the filtering
        in_members = np.arange(np.sum(mask[self.audit_dataset["in_members"]]))
        # find out how many out-members survived the filtering
        num_out_members = np.sum(mask[self.audit_dataset["out_members"]])
        out_members = np.arange(len(in_members), len(in_members) + num_out_members)
        
        assert len(audit_data_indices) == len(in_members) + len(out_members)

        if len(audit_data_indices) == 0:
            raise ValueError("No points in the audit dataset are used for the shadow models")
        
        in_indices_mask = in_indices_mask[:,mask]
        n_audit_points = len(audit_data_indices)

        logger.info(f"Number of points in the audit dataset that are used for online attack: {len(audit_data_indices)}")

        # STEP 3: Run the attack
        # run audit points through target and shadow models to get logits
        x_logits_target_model = np.array(self.signal([self.target_model], self.handler, audit_data_indices)).squeeze()
        x_logits_shadow_models = np.array(self.signal(self.shadow_models, self.handler, audit_data_indices))
        
        self.fixed_in_std = self.get_std(x_logits_shadow_models.flatten(), in_indices_mask.flatten(), True, "fixed")
        self.fixed_out_std = self.get_std(x_logits_shadow_models.flatten(), (~in_indices_mask).flatten(), False, "fixed")

        # Make a "random sample" to compute p(z) for points in attack dataset on the OUT shadow models for each audit point
        self.attack_data_index = self.sample_indices_from_population(include_train_indices = False,
                                                                     include_test_indices = False)
        if len(self.attack_data_index) == 0:
            raise ValueError("There are no auxilliary points to use for the attack.")
        n_attack_points = int(self.attack_data_fraction * len(self.attack_data_index))

        # subsample the attack data based on the fraction
        logger.info(f"Subsampling attack data from {len(self.attack_data_index)} points")
        self.attack_data_index = np.random.choice(
            self.attack_data_index,
            n_attack_points,
            replace=False
        )
        logger.info(f"Number of attack data points after subsampling: {len(self.attack_data_index)}")
        assert len(self.attack_data_index) == n_attack_points

        # Run sampled attack points through target and shadow models
        attack_data_in_indices_mask = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.attack_data_index).T
        z_logits_target_model = np.array(self.signal([self.target_model], self.handler, self.attack_data_index)).squeeze()
        z_logits_shadow_models = np.array(self.signal(self.shadow_models, self.handler, self.attack_data_index))

        log_likelihoods = np.ndarray((n_audit_points, n_attack_points))

        for i in tqdm(range(n_audit_points), desc="Calculating likelihoods"):
            for j in range(n_attack_points):
                x_mask = ~attack_data_in_indices_mask[:, j] & in_indices_mask[:, i]
                z_mask = attack_data_in_indices_mask[:, j] & ~in_indices_mask[:, i]

                if sum(x_mask) == 0 or sum(z_mask) == 0:
                    logger.info("error, x_mask or z_mask empty")
                    logger.info(attack_data_in_indices_mask[:, j], in_indices_mask[:, i])
                    logger.info(~attack_data_in_indices_mask[:, j], ~in_indices_mask[:, i])
                    logger.info(x_mask, z_mask)
                    log_likelihoods[i, j] = -np.inf
                    continue

                def log_pr_logit(shadow_models_logits, mask, is_in, target_logit):
                    mean = np.mean(shadow_models_logits[mask])
                    std = self.get_std(shadow_models_logits, mask, is_in, self.var_calculation)

                    return norm.logpdf(target_logit, mean, std + self.epsilon)
                
                x_ratio = log_pr_logit(x_logits_shadow_models[:, i], x_mask, True, x_logits_target_model[i]) - log_pr_logit(x_logits_shadow_models[:, i], z_mask, True, x_logits_target_model[i])
                z_ratio = log_pr_logit(z_logits_shadow_models[:, i], x_mask, True, z_logits_target_model[i]) - log_pr_logit(z_logits_shadow_models[:, i], z_mask, True, z_logits_target_model[i])
                log_likelihoods[i, j] = x_ratio + z_ratio

        # Determine score as fraction of sampled points z that exceed threshold gamma
        score = np.mean(log_likelihoods > np.log(self.gamma), axis=1)

        # pick out the in-members and out-members signals
        self.in_member_signals = score[in_members].reshape(-1,1)
        self.out_member_signals = score[out_members].reshape(-1,1)
        
        # set true labels for being in the training dataset
        true_labels = np.concatenate(
            [
                np.ones(len(self.in_member_signals)),
                np.zeros(len(self.out_member_signals)),
            ]
        )
        signal_values = np.concatenate(
            [self.in_member_signals, self.out_member_signals]
        )

        return MIAResult.from_full_scores(true_membership=true_labels,
                                    signal_values=signal_values,
                                    result_name="MS-LiRA")


