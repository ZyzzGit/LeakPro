"""Run optuna to find best hyperparameters."""
from collections.abc import Generator

import optuna
from torch import Tensor

from leakpro.attacks.attack_base import AbstractAttack
from leakpro.metrics.attack_result import MIAResult
from leakpro.schemas import OptunaConfig
from leakpro.utils.logger import logger
from leakpro.utils.seed import seed_everything


def optuna_optimal_hyperparameters(attack_object: AbstractAttack, optuna_config: OptunaConfig = None) -> optuna.study.Study:
    """Find optimal hyperparameters for an attack object.

    Args:
    ----
            attack_object (Union[AbstractGIA, AbstractMIA]): Attack object to find optimal hyperparameters for.
            optuna_config (OptunaConfig): configureable settings for optuna

    Returns:
    -------
            optuna.study.Study: Optuna study object containing the results of the optimization.

    """
    def objective(trial: optuna.trial.Trial) -> Tensor:
        # Suggest hyperparameters
        new_config = attack_object.suggest_parameters(trial)
        # Reset attack to apply new hyperparameters
        attack_object.reset_attack(new_config)
        seed_everything(optuna_config.seed)
        result = attack_object.run_attack()
        if isinstance(result, Generator):
            for step, intermediary_results, result_object in result:
                trial.report(intermediary_results, step)

                if trial.should_prune():
                    raise optuna.TrialPruned()
                # save results if not pruned
                if result_object is not None:
                    result_object.save(name="optuna", path="./leakpro_output/results", config=attack_object.get_configs())
                    return intermediary_results
        elif isinstance(result, MIAResult):
            # Retrieve configuration and result metric
            avg_tpr = result.avg_tpr

            trial.set_user_attr("config", new_config)
            trial.set_user_attr("avg_tpr", avg_tpr)

            # Optionally print the details for immediate feedback
            logger.info(f"Trial {trial.number} - Config: {new_config} - avg TPR (FPR < 1e-2): {avg_tpr}")

            # MIA cannot be used with pruning as we need the final result to be computed
            return result.avg_tpr  # add something reasonable to optimize toward here
        return None

    # Define the pruner and study
    pruner = optuna_config.pruner
    study = optuna.create_study(direction=optuna_config.direction, pruner=pruner)

    # Run optimization
    study.optimize(objective, n_trials=optuna_config.n_trials)

    # Display and save the results
    logger.info(f"Best hyperparameters: {study.best_params}")
    logger.info(f"Best optimized value: {study.best_value}")

    f_results_file = attack_object.attack_folder_path + "/optuna_results.txt"
    with open(f_results_file, "w") as f:
        f.write("Best hyperparameters:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")

    logger.info(f"Results saved to {f_results_file}")

    # Return the study
    return study
