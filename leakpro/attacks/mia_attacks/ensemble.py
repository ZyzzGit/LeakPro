"""Implementation of the ensemble method from "Improving Membership Inference Attacks against Classification Models"."""

import numpy as np
from warnings import filterwarnings
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from xgboost import XGBClassifier


from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.abstract_input_handler import AbstractInputHandler
from leakpro.metrics.attack_result import MIAResult
from leakpro.signals.signal import get_signal_from_name
from leakpro.utils.import_helper import Self
from leakpro.utils.logger import logger


class AttackEnsemble(AbstractMIA):
    """Implementation of the Ensemble attack."""

    def __init__(self:Self,
                 handler: AbstractInputHandler,
                 configs: dict
                 ) -> None:
        """Initialize the Ensemble attack.

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

        logger.info("Configuring Ensemble attack")
        self._configure_attack(configs)


    def _configure_attack(self:Self, configs: dict) -> None:
        """Configure the Ensemble attack.

        Args:
        ----
            configs (dict): Configuration parameters for the attack.

        """
        self.num_instances = configs.get("num_instances", 50) # Number of instances
        self.subset_size = configs.get("subset_size", 50)
        self.num_pairs = configs.get("num_pairs", 20)
        self.num_runs = configs.get("num_runs", 5)
        self.training_data_fraction = configs.get("training_data_fraction", 0.5)
        self.audit = configs.get("audit", False)
        self.signal_names = configs.get("signals", ["TrendLoss", "SeasonalityLoss"]) # [TrendLoss(), SeasonalityLoss()]
        self.signals = [get_signal_from_name(signal_name) for signal_name in self.signal_names]
        self.online = True


        # Define the validation dictionary as: {parameter_name: (parameter, min_value, max_value)}
        validation_dict = {
            "num_instances": (self.num_instances, 1, None),
            "subset_size": (self.subset_size, 1, None),
            "num_pairs": (self.num_pairs, 1, None),
            "num_runs": (self.num_runs, 1, None),
            "training_data_fraction": (self.training_data_fraction, 0, 1),
        }

        # Validate parameters
        for param_name, (param_value, min_val, max_val) in validation_dict.items():
            self._validate_config(param_name, param_value, min_val, max_val)

    def description(self:Self) -> dict:
        """Return a description of the attack."""
        title_str = "Ensemble attack"
        reference_str = "Shlomit Shachor, Natalia Razinkov, Abigail Goldsteen and Ariel Farkash. \
            Improving Membership Inference Attacks against Classification Models. (2024)."
        summary_str = "The Ensemble attack is a membership inference attack based on an ensemble of classifications models."
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

        filterwarnings(action='ignore', category=ConvergenceWarning)
        logger.info("Preparing shadow models for Ensemble attack")
        # Check number of shadow models that are available

        # sample dataset to compute histogram
        logger.info("Preparing attack data for training the Ensemble attack")

        # Get all available indices for attack dataset including training and test data
        self.attack_data_indices = self.sample_indices_from_population(include_train_indices = self.online,
                                                                       include_test_indices = self.online)
        logger.info(f"{self.attack_data_indices=}")

        if not self.audit:
            # train shadow models
            logger.info(f"Check for {self.num_instances} shadow models (dataset: {len(self.attack_data_indices)} points)")
            self.shadow_model_indices = ShadowModelHandler().create_shadow_models(
                num_models = self.num_instances,
                shadow_population = self.attack_data_indices,
                training_fraction = self.training_data_fraction,
                online = self.online)
            # load shadow models
            self.shadow_models, _ = ShadowModelHandler().get_shadow_models(self.shadow_model_indices)

            self.in_indices_masks = ShadowModelHandler().get_in_indices_mask(self.shadow_model_indices, self.audit_dataset["data"])
            self.out_indices_masks = np.logical_not(self.in_indices_masks)
            




    def run_attack(self:Self) -> MIAResult:
        """Run the attack on the target model and dataset.

        Returns
        -------
            Result(s) of the metric.

        """
        if self.audit:
            logger.info("Running Ensemble shadow attack (audit mode)")
        else:
            logger.info("Running Ensemble shadow attack (attack mode)")
            

        ensemble_models = []
        for instance in range(self.num_instances):
            logger.info(f"Running instance number {instance+1}/{self.num_instances}")

            if self.audit:
                current_model = self.target_model
                in_indices = self.audit_dataset["in_members"]
                out_indices = self.audit_dataset["out_members"]
            else:
                # Get current shadow model
                current_model = self.shadow_models[instance]

                # Get indices which the current shadow model is trained or not trained on
                logger.info(f"{self.in_indices_masks=}")
                in_indices = self.audit_data_indices[self.in_indices_masks[:, instance]]
                out_indices = self.audit_data_indices[self.out_indices_masks[:, instance]]

            # Choose a subset of these to train the membership classifiers on
            in_indices = np.random.choice(in_indices, self.subset_size * self.num_pairs, replace=False)
            out_indices = np.random.choice(out_indices, self.subset_size * self.num_pairs, replace=False)

            # Create set of features and in/out label for each indices in subsets
            in_features = []
            out_features = []
            for signal in self.signals:
                in_features.append(np.squeeze(signal([current_model],
                                              self.handler,
                                              in_indices)))
                out_features.append(np.squeeze(signal([current_model],
                                               self.handler,
                                               out_indices)))
            in_features = np.swapaxes(np.array(in_features), 0, 1)
            out_features = np.swapaxes(np.array(out_features), 0, 1)


            pair_models = []
            for pair_i in tqdm(range(self.num_pairs),
                               total=self.num_pairs,
                               desc="Training the best membership classifier for each pair"):
                pair_subset = list(range(pair_i * self.subset_size, (pair_i + 1) * self.subset_size))

                pair_features = np.vstack((in_features[pair_subset], out_features[pair_subset]))
                pair_label = np.hstack((np.full(self.subset_size, 0), np.full(self.subset_size, 1)))

                run_models = []
                run_auc = []
                for run_i in range(self.num_runs):
                    # Randomly split the pair 50-50 into train and test data for the membership classifier
                    features_train, features_test, label_train, label_test = train_test_split(
                            pair_features, pair_label, test_size=0.5)
                    
                    # Try each combination of scaler and model, record auc score on test set
                    for scaler in [StandardScaler, MinMaxScaler, RobustScaler]:
                        models = [RandomForestClassifier(), 
                                  GradientBoostingClassifier(),
                                  LogisticRegression(),
                                  DecisionTreeClassifier(),
                                  KNeighborsClassifier(),
                                  MLPClassifier(hidden_layer_sizes=(512,100,64), max_iter=100),
                                  XGBClassifier(),
                                  SVC(kernel="poly", probability=True),
                                  SVC(kernel="rbf", probability=True),
                                  SVC(kernel="sigmoid", probability=True)]
                        
                        for model in models:
                            pipe = make_pipeline(scaler(), model)
                            pipe = pipe.fit(features_train, label_train)
                            
                            probs = pipe.predict_proba(features_test)[:, 1]

                            run_models.append(pipe)
                            run_auc.append(roc_auc_score(label_test, probs))

                # Choose model with best ROC-AUC
                best_model = run_models[0]
                best_auc = 0.0
                for i in range(len(run_models)):
                    if run_auc[i] > best_auc:
                        best_auc = run_auc[i]
                        best_model = run_models[i]
                pair_models.append(best_model)
            ensemble_models.append(pair_models)
        
        self.audit_data_indices = self.audit_dataset["data"]
        self.in_members = self.audit_dataset["in_members"]
        self.out_members = self.audit_dataset["out_members"]

        features = []
        for signal in self.signals:
            features.append(np.squeeze(signal([self.target_model],
                                              self.handler,
                                              self.audit_data_indices)))
        features = np.swapaxes(np.array(features), 0, 1)
        
        # Average membership probabilities over all instances and models
        proba = np.zeros(features.shape[0])
        for best_models in ensemble_models:
            instance_proba = np.zeros(features.shape[0])
            for model in best_models:
                instance_proba += model.predict_proba(features)[:, 1]
            proba += instance_proba / len(best_models)
        self.proba = proba / self.num_instances


        # Generate thresholds based on the range of computed scores for decision boundaries
        self.thresholds = np.linspace(np.min(self.proba), np.max(self.proba), 1000)

        # Split the score array into two parts based on membership: in (training) and out (non-training)
        self.in_member_signals = self.proba[self.in_members].reshape(-1,1)  # Scores for known training data members
        self.out_member_signals = self.proba[self.out_members].reshape(-1,1)  # Scores for non-training data members

        # Create prediction matrices by comparing each score against all thresholds
        member_preds = np.less(self.in_member_signals, self.thresholds).T  # Predictions for training data members
        non_member_preds = np.less(self.out_member_signals, self.thresholds).T  # Predictions for non-members

        # Concatenate the prediction results for a full dataset prediction
        predictions = np.concatenate([member_preds, non_member_preds], axis=1)

        # Prepare true labels array, marking 1 for training data and 0 for non-training data
        true_labels = np.concatenate(
            [np.ones(len(self.in_member_signals)), np.zeros(len(self.out_member_signals))]
        )

        # Combine all signal values for further analysis
        signal_values = np.concatenate([self.in_member_signals, self.out_member_signals])

        # compute ROC, TP, TN etc
        return MIAResult(
            predicted_labels=predictions,
            true_labels=true_labels,
            predictions_proba=None,
            signal_values=signal_values,
        )


