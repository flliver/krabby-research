"""Kinematically slave plant ``*_Body_Hip_RevoluteJoint`` to ``*_Body_CamShaft_RevoluteJoint``.

Uses the same Whitworth mapping as training MDP ``CrabHexDelayedJointPositionAction``.
Skipped when the articulation uses ``crab_hex_ref.usd`` joint names (single driven yaw DOF).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from firmware.sim.isaac_scene import _crab_hex_articulation_uses_ref_joint_names


@dataclass(frozen=True, slots=True)
class CamHipSlaveBinding:
    """Resolved cam-shaft and passive hip DOF indices (one env, all six legs when present)."""

    shaft_joint_ids: list[int]
    hip_joint_ids: list[int]


def build_cam_hip_slave_binding(articulation: Any) -> CamHipSlaveBinding | None:
    """Return index pairs on the main hex plant USD; ``None`` for ref USD or missing joints."""
    if not hasattr(articulation, "find_joints"):
        return None
    if _crab_hex_articulation_uses_ref_joint_names(articulation):
        return None
    try:
        shaft_ids, shaft_names = articulation.find_joints(
            [".*_Body_CamShaft_RevoluteJoint"], preserve_order=True
        )
    except (ValueError, TypeError):
        return None
    if not shaft_ids:
        return None
    hip_names = [
        name.replace("_Body_CamShaft_RevoluteJoint", "_Body_Hip_RevoluteJoint")
        for name in shaft_names
    ]
    try:
        hip_ids, _ = articulation.find_joints(hip_names, preserve_order=True)
    except (ValueError, TypeError):
        return None
    if len(hip_ids) != len(shaft_ids):
        return None
    return CamHipSlaveBinding(
        shaft_joint_ids=[int(i) for i in shaft_ids],
        hip_joint_ids=[int(i) for i in hip_ids],
    )


def _cam_shaft_to_hip(theta_shaft: Any, omega_shaft: Any) -> tuple[Any, Any]:
    from firmware.sim.isaac_scene import ensure_parkour_tasks_on_path

    ensure_parkour_tasks_on_path()
    from parkour_tasks.crab_hex_forward_task.mdp.crab_hex_cam_mapping import (
        cam_shaft_to_hip,
    )

    return cam_shaft_to_hip(theta_shaft, omega_shaft)


def apply_cam_hip_slave(binding: CamHipSlaveBinding, articulation: Any) -> None:
    """After physics: map shaft state to hip targets and sync simulated joint state."""
    import torch

    data = articulation.data
    theta_shaft = data.joint_pos[:, binding.shaft_joint_ids]
    omega_shaft = data.joint_vel[:, binding.shaft_joint_ids]
    if not torch.isfinite(theta_shaft).all() or not torch.isfinite(omega_shaft).all():
        theta_shaft = torch.nan_to_num(theta_shaft, nan=0.0, posinf=0.0, neginf=0.0)
        omega_shaft = torch.nan_to_num(omega_shaft, nan=0.0, posinf=0.0, neginf=0.0)
    theta_hip, omega_hip = _cam_shaft_to_hip(theta_shaft, omega_shaft)
    articulation.set_joint_position_target(
        theta_hip, joint_ids=binding.hip_joint_ids
    )
    articulation.set_joint_velocity_target(
        omega_hip, joint_ids=binding.hip_joint_ids
    )
    if not hasattr(articulation, "write_joint_state_to_sim"):
        return
    joint_pos = data.joint_pos.clone()
    joint_vel = data.joint_vel.clone()
    joint_pos[:, binding.hip_joint_ids] = theta_hip
    joint_vel[:, binding.hip_joint_ids] = omega_hip
    articulation.write_joint_state_to_sim(joint_pos, joint_vel)
