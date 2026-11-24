"""Articulation joint angle -> host hall edge counter (``saf`` telemetry source).

Counts are signed and kept per board: a rising joint angle is extension and counts
down (``HALL_EXTEND_COUNT_SIGN``). Each actuator's counts go to the board whose
firmware drives it (``board_id_for_firmware_id``).
"""

from __future__ import annotations

from typing import Sequence

from firmware.sim.host_bridge import HostActuatorBridge
from firmware.sim.plant_readback import PlantReadbackView
from firmware.sim.pin_state import PinStateModel
from firmware.sim.pot_autocal_plant import hall_b_level
from firmware.sim.simulated_actuator import SimulatedActuator
from firmware.sim.simulated_robot_config import board_id_for_firmware_id


def actuator_board_id(actuator: SimulatedActuator) -> int:
    return board_id_for_firmware_id(actuator.slot.firmware_actuator_id)


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
        bridge.board_sync_hall_joint(
            actuator_board_id(actuator),
            actuator.slot.slot_index,
            position,
            lower,
            upper,
        )


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
        bridge.board_retarget_hall_joint(
            actuator_board_id(actuator),
            actuator.slot.slot_index,
            position,
            lower,
            upper,
        )


def sync_hall_b_levels(
    bridge: HostActuatorBridge,
    actuators: Sequence[SimulatedActuator],
) -> None:
    """Hall actuators: the pot pin carries hall B, which flips with every counted edge."""
    for actuator in actuators:
        if not actuator.slot.is_hall:
            continue
        count = bridge.board_hall_edge_count(actuator_board_id(actuator), actuator.slot.slot_index)
        bridge.set_analog_read(actuator.slot.pin_pot, hall_b_level(count))
