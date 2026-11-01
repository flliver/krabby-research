"""Unit tests for auto-cal travel-fraction pot plant."""

from __future__ import annotations

from firmware.sim.pot_autocal_plant import (
    advance_travel_from_pwm,
    default_travel_fractions,
    sync_pot_reads_from_travel,
)
from firmware.sim.pin_state import PinStateModel
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


def _front_actuators():
    config = SimulatedRobotConfig.front_board_rev3()
    indices = {name: i for i, name in enumerate(config.isaac_joint_names())}
    return config, config.actuators(indices, max_joint_speed=1.0)


def test_travel_advances_with_signed_pwm():
    config, actuators = _front_actuators()
    travel = default_travel_fractions(actuators)
    slot = config.slot_by_firmware_id("FLHL")
    pin_state = PinStateModel()
    pwm_r, pwm_l, pin_en = slot.drive_pins
    pin_state.digital_write(pin_en, 1)
    pin_state.analog_write(pwm_r, 200)
    pin_state.analog_write(pwm_l, 0)

    flhl = next(a for a in actuators if a.slot.firmware_actuator_id == "FLHL")
    advance_travel_from_pwm([flhl], pin_state, travel)
    assert travel["FLHL"] > 0.5


def test_sync_pot_reads_from_travel_writes_band_adc():
    config, actuators = _front_actuators()

    class Bridge:
        def __init__(self) -> None:
            self.pots: dict[int, int] = {}

        def set_analog_read(self, pin: int, value: int) -> None:
            self.pots[pin] = value

    bridge = Bridge()
    band = config.pot_band_for_slot(0)
    flhy = actuators[0]
    sync_pot_reads_from_travel(bridge, [flhy], {"FLHY": 0.0})
    assert bridge.pots[flhy.slot.pin_pot] == band.min_adc
