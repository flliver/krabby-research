from isaaclab.utils import configclass

from parkour_tasks.crab_hexapod_task.config.crab_hex.agents.crab_hex_rl_cfg import (
    CrabHexParkourRslRlActorCfg,
    CrabHexParkourRslRlDepthEncoderCfg,
    CrabHexParkourRslRlEstimatorCfg,
    CrabHexParkourRslRlStateHistEncoderCfg,
)
from parkour_tasks.crab_hexapod_task.config.crab_hex.agents.crab_hex_rl_cfg import (
    CrabHexParkourRslRlOnPolicyRunnerCfg,
    CrabHexParkourRslRlPpoActorCriticCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlPpoAlgorithmCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_student_ppo_cfg import (
    UnitreeGo2ParkourStudentPPORunnerCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_teacher_ppo_cfg import (
    UnitreeGo2ParkourTeacherPPORunnerCfg,
)


@configclass
class CrabHexTeacherPPORunnerCfg(CrabHexParkourRslRlOnPolicyRunnerCfg, UnitreeGo2ParkourTeacherPPORunnerCfg):
    experiment_name = "crab_hex_teacher"
    policy = CrabHexParkourRslRlPpoActorCriticCfg(
        init_noise_std=0.65,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        activation="elu",
        actor=CrabHexParkourRslRlActorCfg(
            class_name="Actor",
            state_history_encoder=CrabHexParkourRslRlStateHistEncoderCfg(
                class_name="StateHistoryEncoder",
            ),
        ),
    )
    estimator = CrabHexParkourRslRlEstimatorCfg(hidden_dims=[128, 64])


@configclass
class CrabHexFlatWalkPPORunnerCfg(CrabHexTeacherPPORunnerCfg):
    experiment_name = "crab_hex_flat_walk"
    max_iterations = 20000
    clip_actions = 1.0
    algorithm = ParkourRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        dagger_update_freq=20,
        priv_reg_coef_schedual=[0.0, 0.1, 2000.0, 3000.0],
    )

    def __post_init__(self):
        self.policy.init_noise_std = 1.5


@configclass
class CrabHexStudentPPORunnerCfg(CrabHexParkourRslRlOnPolicyRunnerCfg, UnitreeGo2ParkourStudentPPORunnerCfg):
    experiment_name = "crab_hex_student"
    policy = CrabHexParkourRslRlPpoActorCriticCfg(
        init_noise_std=0.65,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        scan_encoder_dims=[128, 64, 32],
        priv_encoder_dims=[64, 20],
        activation="elu",
        actor=CrabHexParkourRslRlActorCfg(
            class_name="Actor",
            state_history_encoder=CrabHexParkourRslRlStateHistEncoderCfg(
                class_name="StateHistoryEncoder",
            ),
        ),
    )
    estimator = CrabHexParkourRslRlEstimatorCfg(hidden_dims=[128, 64])
    depth_encoder = CrabHexParkourRslRlDepthEncoderCfg(
        hidden_dims=512,
        learning_rate=1e-3,
        num_steps_per_env=24 * 5,
    )
