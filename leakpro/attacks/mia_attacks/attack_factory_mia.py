"""Module that contains the AttackFactory class which is responsible for creating the attack objects."""


from leakpro.attacks.mia_attacks.abstract_mia import AbstractMIA
from leakpro.attacks.mia_attacks.attack_p import AttackP
from leakpro.attacks.mia_attacks.HSJ import AttackHopSkipJump
from leakpro.attacks.mia_attacks.lira import AttackLiRA
from leakpro.attacks.mia_attacks.loss_trajectory import AttackLossTrajectory
from leakpro.attacks.mia_attacks.qmia import AttackQMIA
from leakpro.attacks.mia_attacks.rmia import AttackRMIA
from leakpro.attacks.mia_attacks.rmia_direct import AttackRMIADirect
from leakpro.attacks.mia_attacks.yoqo import AttackYOQO
from leakpro.attacks.mia_attacks.ensemble import AttackEnsemble
from leakpro.attacks.utils.distillation_model_handler import DistillationModelHandler
from leakpro.attacks.utils.shadow_model_handler import ShadowModelHandler
from leakpro.input_handler.mia_handler import MIAHandler
from leakpro.utils.logger import logger


class AttackFactoryMIA:
    """Class responsible for creating the attack objects."""

    attack_classes = {
        "population": AttackP,
        "rmia": AttackRMIA,
        "qmia": AttackQMIA,
        "loss_traj":AttackLossTrajectory,
        "lira": AttackLiRA,
        "HSJ" : AttackHopSkipJump,
        "yoqo": AttackYOQO,
        "ensemble": AttackEnsemble,
        "rmia_direct": AttackRMIADirect,
    }

    # Shared variables for all attacks
    shadow_model_handler = None
    distillation_model_handler = None

    @classmethod
    def create_attack(cls, key: str, handler: MIAHandler) -> AbstractMIA:  # noqa: ANN102
        """Create the attack object.

        Args:
        ----
            key (str): The unique key of the attack.
            handler (MIAHandler): The input handler object.

        Returns:
        -------
            AttackBase: An instance of the attack object.

        Raises:
        ------
            ValueError: If the attack type is unknown.

        """

        if AttackFactoryMIA.shadow_model_handler is None:
            logger.info("Creating shadow model handler singleton")
            AttackFactoryMIA.shadow_model_handler = ShadowModelHandler(handler)
        else:
            logger.info("Shadow model handler singleton already exists, updating state")
            AttackFactoryMIA.shadow_model_handler = ShadowModelHandler(handler)

        if AttackFactoryMIA.distillation_model_handler is None:
            logger.info("Creating distillation model handler singleton")
            AttackFactoryMIA.distillation_model_handler = DistillationModelHandler(handler)
        else:
            logger.info("Distillation model handler singleton already exists, updating state")
            AttackFactoryMIA.distillation_model_handler = DistillationModelHandler(handler)

        attack_config = [ac for ac in handler.configs.audit.attack_list if ac['attack_key'] == key]
        if len(attack_config) > 1:
            raise Exception("Multiple attacks with identical key: {key}")
        else:
            attack_config = attack_config[0]
        attack_name = attack_config['attack_name']
        if attack_name in cls.attack_classes:
            attack_object = cls.attack_classes[attack_name](handler, attack_config)
            attack_object.set_effective_optuna_metadata(attack_config) # remove optuna metadata if params not will be optimized
            return attack_object
        raise ValueError(f"Unknown attack type: {attack_name} ({key})")
