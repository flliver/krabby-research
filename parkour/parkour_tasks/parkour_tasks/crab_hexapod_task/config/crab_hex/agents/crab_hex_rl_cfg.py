# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Policy layout sizes for ``CrabHexParkourObservations`` (see crab_hexapod_task.mdp.observations).
#
# Krabby ``crab_simple.usda``: ``num_joints = 18``, joint_pos action dim ``18``.
#   obs_buf_dim = 15 + 2 * num_joints + action_dim + num_contact  ->  15 + 36 + 18 + 6 = 75  (= num_prop; +2 root_lin_vel_xy)
#   priv_latent   = mass(1+3) + friction(1) + stiffness(N) + damping(N)  ->  4 + 1 + 18 + 18 = 41
#   policy flat len = (1 + history_length) * obs_buf_dim + 132 + 9 + priv_latent
#                   = 11 * 75 + 141 + 41 = 987  (141 = 132 scan + 9 priv_explicit)

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass

from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import (
    ParkourRslRlBaseCfg,
    ParkourRslRlOnPolicyRunnerCfg,
    ParkourRslRlPpoActorCriticCfg,
)


@configclass
class CrabHexParkourRslRlOnPolicyRunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    """Selects ``OnPolicyRunnerCrabHex`` (exploration-std clamp) in ``train.py`` / ``play.py``."""

    runner_class_name: str = "OnPolicyRunnerCrabHex"
    num_steps_per_env: int = 24
    save_interval: int = 100
    empirical_normalization: bool = False


@configclass
class CrabHexParkourRslRlPpoActorCriticCfg(ParkourRslRlPpoActorCriticCfg):
    class_name: str = "CrabHexActorCriticRMA"


@configclass
class CrabHexParkourRslRlBaseCfg(ParkourRslRlBaseCfg):
    """Crab simple: must match ``CrabHexParkourObservations`` tensor layout (see module docstring)."""

    num_prop: int = 75
    num_priv_latent: int = 41
    # num_scan=132, num_hist=10, num_priv_explicit=9 — inherited


@configclass
class CrabHexParkourRslRlStateHistEncoderCfg(CrabHexParkourRslRlBaseCfg):
    class_name: str = "StateHistoryEncoder"
    channel_size: int = 10


@configclass
class CrabHexParkourRslRlEstimatorCfg(CrabHexParkourRslRlBaseCfg):
    class_name: str = "DefaultEstimator"
    train_with_estimated_states: bool = True
    hidden_dims: list[int] = MISSING
    learning_rate: float = 1.0e-4


@configclass
class CrabHexParkourRslRlActorCfg(CrabHexParkourRslRlBaseCfg):
    class_name: str = "Actor"
    state_history_encoder: CrabHexParkourRslRlStateHistEncoderCfg = MISSING


@configclass
class CrabHexParkourRslRlDepthEncoderCfg(CrabHexParkourRslRlBaseCfg):
    backbone_class_name: str = "DepthOnlyFCBackbone58x87"
    encoder_class_name: str = "RecurrentDepthBackbone"
    depth_shape: tuple[int, int] = (87, 58)
    hidden_dims: int = 512
    learning_rate: float = 1.0e-3
    num_steps_per_env: int = 24 * 5
