"""Articulation joint angle -> host hall edge counter (``saf`` telemetry source)."""

from __future__ import annotations

from typing import Sequence

from firmware.sim.host_bridge import HostActuatorBridge
from firmware.sim.plant_readback import PlantReadbackView
from firmware.sim.pin_state import PinStateModel
from firmware.sim.simulated_actuator import SimulatedActuator


def sync_hall_edge_counts(
    bridge: HostActuatorBridge,
    actuators: Sequence[SimulatedActuator],
    plant: PlantReadbackView | None,
) -> None:
    """Feed each slot's hall index from Isaac joint position and soft limits."""
    if plant is None:
        return

    for actuator in actuators:
        travel = plant.joint_travel(actuator.joint_index)
        if travel is None:
            continue
        position, lower, upper = travel
        hall_slot = actuator.slot.slot_index
        bridge.sync_hall_joint(hall_slot, position, lower, upper)


def retarget_hall_baselines_when_idle(
    bridge: HostActuatorBridge,
    actuators: Sequence[SimulatedActuator],
    plant: PlantReadbackView | None,
    pin_state: PinStateModel,
) -> None:
    """After physics, align hall baselines for stopped actuators (no shaft motion on hardware)."""
    if plant is None:
        return

    for actuator in actuators:
        if actuator.signed_pwm_command(pin_state) != 0:
            continue
        travel = plant.joint_travel(actuator.joint_index)
        if travel is None:
            continue
        position, lower, upper = travel
        hall_slot = actuator.slot.slot_index
        bridge.retarget_hall_joint(hall_slot, position, lower, upper)
