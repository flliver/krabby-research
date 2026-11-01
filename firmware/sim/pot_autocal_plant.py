"""Travel-fraction pot plant for USB auto-cal (matches host unit test plant)."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Sequence

from firmware.sim.host_bridge import HostActuatorBridge
from firmware.sim.pin_state import PinStateModel
from firmware.sim.simulated_actuator import SimulatedActuator

# Same step as tests/unit/firmware/test_host_auto_calibration_plant.py
AUTOCAL_TRAVEL_STEP = 0.06
AUTOCAL_PWM_DRIVE_THRESHOLD = 50


def default_travel_fractions(actuators: Sequence[SimulatedActuator]) -> dict[str, float]:
    return {a.slot.firmware_actuator_id: 0.5 for a in actuators}


def sync_pot_reads_from_travel(
    bridge: HostActuatorBridge,
    actuators: Sequence[SimulatedActuator],
    travel: Mapping[str, float],
) -> None:
    for actuator in actuators:
        band = actuator.pot_band
        if band is None:
            continue
        aid = actuator.slot.firmware_actuator_id
        frac = travel.get(aid, 0.5)
        adc = int(round(band.min_adc + frac * (band.max_adc - band.min_adc)))
        bridge.set_analog_read(actuator.slot.pin_pot, adc)


def advance_travel_from_pwm(
    actuators: Sequence[SimulatedActuator],
    pin_state: PinStateModel,
    travel: MutableMapping[str, float],
    *,
    step: float = AUTOCAL_TRAVEL_STEP,
) -> None:
    for actuator in actuators:
        cmd = actuator.signed_pwm_command(pin_state)
        aid = actuator.slot.firmware_actuator_id
        frac = travel.get(aid, 0.5)
        if cmd >= AUTOCAL_PWM_DRIVE_THRESHOLD:
            travel[aid] = min(1.0, frac + step)
        elif cmd <= -AUTOCAL_PWM_DRIVE_THRESHOLD:
            travel[aid] = max(0.0, frac - step)
