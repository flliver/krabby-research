"""Articulation joint angle -> host hall edge counter (``saf`` telemetry source)."""

from __future__ import annotations

from typing import Any, Sequence

from firmware.sim.host_bridge import HostActuatorBridge
from firmware.sim.simulated_actuator import SimulatedActuator


def sync_hall_edge_counts(
    bridge: HostActuatorBridge,
    actuators: Sequence[SimulatedActuator],
    articulation: Any,
    *,
    env_index: int = 0,
) -> None:
    """Feed each slot's hall index from Isaac joint position and soft limits."""
    data = getattr(articulation, "data", None)
    if data is None:
        return
    joint_pos = getattr(data, "joint_pos", None)
    limits = getattr(data, "soft_joint_pos_limits", None)
    if limits is None:
        limits = getattr(data, "joint_pos_limits", None)
    if joint_pos is None or limits is None:
        return

    for actuator in actuators:
        joint_index = actuator.joint_index
        hall_slot = actuator.slot.slot_index
        position = float(joint_pos[env_index][joint_index].item())
        lower = float(limits[env_index][joint_index][0].item())
        upper = float(limits[env_index][joint_index][1].item())
        bridge.sync_hall_joint(hall_slot, position, lower, upper)
