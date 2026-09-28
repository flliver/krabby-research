"""Crab hex-only MDP termination helpers (no edits to shared parkour modules)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.terminations import bad_orientation, illegal_contact, root_height_below_minimum
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terminate_crab_hex_failure(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit_angle: float = 1.05,
    minimum_root_height_z: float | None = None,
    contact_force_threshold: float | None = 15.0,
    hip_contact_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Early failure for tipped base, optional low root height, and optional hip ground contact.

    Stricter than parkour ``terminate_episode`` roll/pitch (1.5 rad) / height (-0.25) so collapsed
    hip-on-ground postures end episodes instead of burning the full horizon.

    Args:
        env: RL environment.
        asset_cfg: Robot articulation (default ``robot``).
        limit_angle: Max tilt (rad) vs upright; passed to :func:`bad_orientation`.
        minimum_root_height_z: If set, terminate when world-frame root ``z`` is below this value.
            If ``None``, this check is skipped (recommended until tuned on parkour terrain).
        contact_force_threshold: Newtons on ``.*_Hip``; hip contact above this triggers termination.
            If ``None``, hip contact is not checked (tilt-only failure).
        hip_contact_sensor_cfg: Contact sensor subset; default hips only (``.*_Hip``).
    """
    mask = bad_orientation(env, limit_angle, asset_cfg)

    if minimum_root_height_z is not None:
        mask = torch.logical_or(
            mask,
            root_height_below_minimum(env, minimum_root_height_z, asset_cfg),
        )

    if contact_force_threshold is not None:
        sensor_cfg = hip_contact_sensor_cfg or SceneEntityCfg("contact_forces", body_names=[".*_Hip"])
        mask = torch.logical_or(mask, illegal_contact(env, contact_force_threshold, sensor_cfg))
    return mask
