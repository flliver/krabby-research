# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Policy layout sizes for ``CrabHexParkourObservations`` (see crab_hex_forward_task.mdp.observations).
#
# Krabby ``assets/crab.usda``: 24 articulation joints (6 legs x CamShaft / Body_Hip / Hip_Femur /
# Femur_Tibia -- ``*_Body_Hip_RevoluteJoint`` is passive for actions, tracking the cam shaft through
# its own actuator, but it is still an articulation joint, so ``asset.num_joints == 24``);
# joint_pos action dim ``18`` (``_CRAB_ACTUATED_JOINT_NAMES``).
#   base term (``ExtremeParkourObservations``): proprio = 13 + 2 * asset.num_joints + action_dim + num_contact
#   crab term adds ``_CRAB_EXTRA_BASE_DIM = 4`` (planar lin-vel 2 + gait-clock sin/cos 2)
#   priv_latent = mass(1) + com(3) + friction(1) + stiffness(N) + damping(N), N = asset.num_joints
#
# ``num_prop = 75`` / ``num_priv_latent = 41`` below are the NETWORK's slice sizes (``CrabHexActorCriticRMA``):
#   actor in_features = num_prop + num_scan + num_priv_latent + num_priv_explicit + num_prop * num_hist
#                     = 75 + 132 + 41 + 9 + 10 * 75 = 1007
# The live policy group is wider: the evals of every head since 2026-08-22 (incl. the policy of
# record) record ``obs_dim_actual: 1149`` in ``run_meta.json``, with the warning that num_prop=75
# does not match that width (train and play share the slicing, so it is self-consistent, but the
# "scan" slice is not the height scan). Reconcile these sizes against ``group_obs_dim["policy"]``
# at env creation instead of assuming they are equal; do not write a closed-form total here
# without re-deriving the layout from the live env.

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
