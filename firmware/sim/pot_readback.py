"""Articulation -> host sensor readback (pots now; current sense later)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Sequence

from firmware.sim.current_sense_readback import sync_current_sense_analog_reads
from firmware.sim.hall_readback import sync_hall_edge_counts
from firmware.sim.host_bridge import HostActuatorBridge
from firmware.sim.plant_readback import PlantReadbackView
from firmware.sim.pot_position import joint_angle_to_pot_adc
from firmware.sim.simulated_actuator import SimulatedActuator


def sync_pot_analog_reads(
    bridge: HostActuatorBridge,
    actuators: Sequence[SimulatedActuator],
    plant: PlantReadbackView | None,
) -> None:
    """Write each slot's ``pin_pot`` from the articulation joint angle and soft limits."""
    if plant is None:
        return

    for actuator in actuators:
        travel = plant.joint_travel(actuator.joint_index)
        if travel is None:
            continue
        position, lower, upper = travel
        adc = joint_angle_to_pot_adc(
            position,
            lower,
            upper,
            band=actuator.pot_band,
        )
        bridge.set_analog_read(actuator.slot.pin_pot, adc)


def sync_sensor_reads(
    bridge: HostActuatorBridge,
    actuators: Sequence[SimulatedActuator],
    plant: PlantReadbackView | None,
    *,
    wrench_body_index_by_actuator: Mapping[str, int] | None = None,
) -> None:
    """Refresh shim ``analogRead`` inputs from the plant before ``updateAll()``."""
    sync_pot_analog_reads(bridge, actuators, plant)
    if wrench_body_index_by_actuator and plant is not None:
        sync_current_sense_analog_reads(
            bridge,
            actuators,
            plant,
            wrench_body_index_by_actuator,
        )
    sync_hall_edge_counts(bridge, actuators, plant)
