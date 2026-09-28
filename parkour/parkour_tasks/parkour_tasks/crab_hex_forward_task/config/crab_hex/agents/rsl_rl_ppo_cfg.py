import os

from isaaclab.utils import configclass

from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.crab_hex_rl_cfg import (
    CrabHexParkourRslRlActorCfg,
    CrabHexParkourRslRlDepthEncoderCfg,
    CrabHexParkourRslRlEstimatorCfg,
    CrabHexParkourRslRlStateHistEncoderCfg,
)
from parkour_tasks.crab_hex_forward_task.config.crab_hex.agents.crab_hex_rl_cfg import (
    CrabHexParkourRslRlOnPolicyRunnerCfg,
    CrabHexParkourRslRlPpoActorCriticCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlPpoAlgorithmCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlDistillationAlgorithmCfg,
)
from parkour_tasks.extreme_parkour_task.config.go2.agents.rsl_teacher_ppo_cfg import (
    UnitreeGo2ParkourTeacherPPORunnerCfg,
)


@configclass
class CrabHexTeacherPPORunnerCfg(CrabHexParkourRslRlOnPolicyRunnerCfg, UnitreeGo2ParkourTeacherPPORunnerCfg):
    """PPO for ``Isaac-Crab-Hex-Teacher-v0``. LR/clip/max_iters depend on ``KRABBY_HEX_TEACHER_MODE`` (see ``crab_hex_env_cfg.py``)."""
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
    algorithm = ParkourRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=16,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        dagger_update_freq=20,
        priv_reg_coef_schedual=[0.0, 0.1, 2000.0, 3000.0],
    )

    def __post_init__(self):
        from parkour_tasks.crab_hex_forward_task.config.crab_hex.crab_hex_env_cfg import (
            _crab_hex_teacher_mode,
        )

        mode = _crab_hex_teacher_mode()
        if mode in ("2a", "2b", "2c"):
            # Paradigm phase 2: same optimizer as the flat-walk runner (the lineage trained
            # these windows in the flat-walk task): LR 3e-4 adaptive, noise 1.5, clip 1.0,
            # mirror-symmetry loss (KRABBY_SYM_LOSS_COEF, default 0.5). save_interval and
            # max_iterations stay at the parent defaults, as the flat-walk runner leaves them.
            self.clip_actions = 1.0
            self.algorithm.learning_rate = 3.0e-4
            self.policy.init_noise_std = 1.5
            _apply_flat_walk_symmetry(self)
        elif mode == "bridge":
            self.clip_actions = 1.0
            self.algorithm.learning_rate = 3.0e-5
            self.save_interval = 100
            self.max_iterations = 100
        elif mode == "2b1":
            self.clip_actions = 1.0
            self.algorithm.learning_rate = 3.0e-5
            self.save_interval = 100
            self.max_iterations = 100
        elif mode == "2b2":
            self.clip_actions = 1.0
            self.algorithm.learning_rate = 1.0e-4
            self.save_interval = 100
            # Teacher-ready 2b2: resume 2b1 6198; stop early at sweet-spot (play + gates), not last ckpt.
            self.max_iterations = 10000


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
        _apply_flat_walk_symmetry(self)


def _apply_flat_walk_symmetry(runner_cfg) -> None:
    """Flat-walk / phase-2 mirror-symmetry loss (extracted verbatim, 2026-09-07)."""
    self = runner_cfg
    # NOTE(mirror-symmetry-campaign, BAKED 2026-08-13): the L/R symmetry-mirror loss is ON
    # by default for flat-walk training (coef 0.5). The 20k validation run broke the tripod
    # target that five reward versions and three weight campaigns never reached (tripod
    # 0.517 vs 0.401 baseline; duty 0.356/0.339 vs 0.146/0.556; 10/10 eval episodes
    # balanced) with zero degradation. Set KRABBY_SYM_LOSS_COEF=0 to disable (e.g. for
    # ablations) — see parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/2026-08-13_0035_mirror_symmetry/RESULTS.md and
    # crab_hex_mirror.py.
    import os

    _sym_coef = float(os.environ.get("KRABBY_SYM_LOSS_COEF", "0.5"))
    if _sym_coef > 0.0:
        from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            data_augmentation_func=(
                "parkour_tasks.crab_hex_forward_task.mdp.crab_hex_mirror:crab_hex_symmetry_augmentation"
            ),
            mirror_loss_coeff=_sym_coef,
        )


@configclass
class CrabHexStudentPPORunnerCfg(CrabHexParkourRslRlOnPolicyRunnerCfg):
    """Depth distillation runner for ``Isaac-Crab-Hex-Student-v0`` (not Go2 student cfg)."""

    max_iterations = 50000
    experiment_name = "crab_hex_student"
    # NOTE(phase-3 root cause, 2026-09-08): the RSL-RL vec-env wrapper clips RAW policy actions to
    # +-clip_actions before the joint action term scales them. The flat-walk / phase-2 runners use
    # 1.0; this runner left it unset (None -> no clip). Harmless on the legacy 2b2 student MDP whose
    # action term already clipped raw actions to +-1, but on the phase-3 MDP (the 2c teacher's
    # ``full`` action space, raw clip +-4.8) an unclipped teacher/student output drove joints at up
    # to ~5x the trained authority: the 2c teacher itself fell in 100% of episodes on the student
    # env and the first phase-3a distillation run never left the falling regime. Mirror the teacher.
    clip_actions = 1.0
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
    algorithm = ParkourRslRlDistillationAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
