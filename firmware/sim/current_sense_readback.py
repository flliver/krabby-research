"""Articulation wrench + drive state -> ``pin_is`` analogRead for host firmware."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from firmware.sim.current_sense_mapping import (
    CurrentSenseBreakpoints,
    map_wrench_to_pin_is_adc,
)
from firmware.sim.host_bridge import HostActuatorBridge
from firmware.sim.simulated_actuator import SimulatedActuator
from firmware.sim.wrench_readback import read_body_incoming_wrench


def sync_current_sense_analog_reads(
    bridge: HostActuatorBridge,
    actuators: Sequence[SimulatedActuator],
    articulation: Any,
    body_index_by_actuator: Mapping[str, int],
    *,
    env_index: int = 0,
    breakpoints: CurrentSenseBreakpoints | None = None,
) -> None:
    data = getattr(articulation, "data", None)
    if data is None:
        return
    joint_vel = getattr(data, "joint_vel", None)
    if joint_vel is None:
        return

    for actuator in actuators:
        actuator_id = actuator.slot.firmware_actuator_id
        body_index = body_index_by_actuator.get(actuator_id)
        if body_index is None:
            continue
        wrench = read_body_incoming_wrench(articulation, body_index, env_index=env_index)
        _, _, pin_en = actuator.slot.drive_pins
        en_energized = bridge.pin_digital(pin_en) != 0
        velocity = float(joint_vel[env_index][actuator.joint_index].item())
        adc = map_wrench_to_pin_is_adc(
            wrench,
            en_energized=en_energized,
            joint_velocity_rad_s=velocity,
            breakpoints=breakpoints,
        )
        bridge.set_analog_read(actuator.slot.pin_is, adc)
