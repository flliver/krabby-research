"""Push simulated joint positions onto host ``analogRead`` pot pins."""

from __future__ import annotations

from typing import Any, Sequence

from firmware.sim.host_bridge import HostActuatorBridge
from firmware.sim.pot_position import joint_angle_to_pot_adc
from firmware.sim.simulated_actuator import SimulatedActuator


def sync_pot_analog_reads(
    bridge: HostActuatorBridge,
    actuators: Sequence[SimulatedActuator],
    articulation: Any,
    *,
    env_index: int = 0,
) -> None:
    """Write each slot's ``pin_pot`` from the articulation joint angle and soft limits."""
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
        position = float(joint_pos[env_index][joint_index].item())
        lower = float(limits[env_index][joint_index][0].item())
        upper = float(limits[env_index][joint_index][1].item())
        adc = joint_angle_to_pot_adc(position, lower, upper)
        bridge.set_analog_read(actuator.slot.pin_pot, adc)
