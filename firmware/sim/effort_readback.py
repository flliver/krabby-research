"""Read Isaac joint motor torque for current-sense cross-check (TASK-2 §4 secondary source)."""

from __future__ import annotations

from typing import Any


def _scalar_at_env_joint(buffer: Any, env_index: int, joint_index: int) -> float:
    try:
        cell = buffer[env_index, joint_index]
    except (TypeError, IndexError, KeyError):
        cell = buffer[env_index][joint_index]
    return float(cell.item() if hasattr(cell, "item") else cell)


def _scalar_at_env_column(buffer: Any, env_index: int, column: int) -> float:
    try:
        cell = buffer[env_index, column]
    except (TypeError, IndexError, KeyError):
        cell = buffer[env_index][column]
    return float(cell.item() if hasattr(cell, "item") else cell)


def read_applied_joint_torque_nm(
    articulation: Any,
    joint_index: int,
    *,
    env_index: int = 0,
) -> float | None:
    """Return applied joint torque (Nm), or ``None`` when sim data is unavailable.

    Prefers ``articulation.data.applied_torque`` (Isaac Lab articulation buffer).
    Falls back to ``ParkourDCMotor.applied_effort`` when the joint index maps to an
    actuator column.
    """
    data = getattr(articulation, "data", None)
    if data is not None:
        applied = getattr(data, "applied_torque", None)
        if applied is not None:
            return _scalar_at_env_joint(applied, env_index, joint_index)

    actuators = getattr(articulation, "actuators", None)
    if not actuators:
        return None
    for actuator in actuators.values():
        effort = getattr(actuator, "applied_effort", None)
        if effort is None:
            continue
        group_ids = getattr(actuator, "joint_indices", None)
        if group_ids is None:
            continue
        if isinstance(group_ids, slice):
            continue
        for col, jid in enumerate(group_ids):
            if int(jid) == int(joint_index):
                return _scalar_at_env_column(effort, env_index, col)
    return None
