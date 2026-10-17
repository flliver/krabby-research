"""Isaac integration: sensor readback (pot, hall, EEPROM, millis, current-sense).

Focused cases also live in sibling modules (``test_isaac_pot_readback``,
``test_isaac_hall_travel``, ``test_isaac_sim_clock``, …).
"""

from __future__ import annotations

import struct

import pytest

from firmware.sim.current_sense_mapping import (
    DEFAULT_LIFT_LOAD_ADC,
    DEFAULT_MOVING_NO_CONTACT_ADC,
)
from firmware.sim.isaac_scene import advance_sim, physics_step_ms
from tests.integration.firmware.conftest import park_at_default_pose, skip_unless_firmware_isaac

pytestmark = pytest.mark.isaacsim

SLOT = "FLHL"
EEPROM_ROLE_ADDR = 32
EEPROM_ROLE_MAGIC = 0xAB
CAL_MAGIC = 0xDEADBEEF
JOG_STEPS = 40
LIFT_SETTLE_STEPS = 120
LIFT_JOG_STEPS = 200
LIFT_JOG_PWM = 255
# OVERVIEW 2g target is ~30 of 1023; allow modest mapping jitter in Isaac.
LIFT_ADC_MIN = DEFAULT_LIFT_LOAD_ADC - 5
FRONT_LIFT_ACTUATORS = ("FLHL", "FLKL", "FRHL", "FRKL")


def _max_pin_is_over_steps(core, config, actuator_ids: tuple[str, ...], steps: int) -> int:
    peak = 0
    for _ in range(steps):
        core.step()
        for actuator_id in actuator_ids:
            slot = config.slot_by_firmware_id(actuator_id)
            peak = max(peak, core.bridge.analog_read(slot.pin_is))
    return peak


def test_digital_read_en_follows_jog(front_board_rig):
    core, _, config, _ = front_board_rig
    slot = config.slot_by_firmware_id(SLOT)
    _, _, pin_en = slot.drive_pins

    core.reset()
    assert core.bridge.pin_digital(pin_en) == 0

    core.jog(SLOT, 120)
    core.step()
    assert core.bridge.pin_digital(pin_en) == 1

    core.jog(SLOT, 0)
    core.step()
    assert core.bridge.pin_digital(pin_en) == 0


def test_pot_adc_increases_on_pwm_jog(front_board_rig):
    core, _, config, _ = front_board_rig
    slot = config.slot_by_firmware_id(SLOT)

    park_at_default_pose(core, core.articulation)
    core.step()
    adc_before = core.bridge.analog_read(slot.pin_pot)

    core.jog(SLOT, 255)
    for _ in range(JOG_STEPS):
        core.step()

    adc_after = core.bridge.analog_read(slot.pin_pot)
    assert adc_after > adc_before


def test_per_motor_pot_bands_distinct(front_board_rig):
    core, _, config, _ = front_board_rig

    park_at_default_pose(core, core.articulation)
    core.step()

    bands = {b.min_adc for b in config.pot_bands}
    assert len(bands) == len(config.slots)


def test_eeprom_role_bytes_round_trip(front_board_rig):
    core, _, _, _ = front_board_rig
    role_left = 2  # board_role.h ROLE_LEFT

    core.reset()
    core.bridge.eeprom_update(EEPROM_ROLE_ADDR, EEPROM_ROLE_MAGIC)
    core.bridge.eeprom_update(EEPROM_ROLE_ADDR + 1, role_left)

    assert core.bridge.eeprom_read(EEPROM_ROLE_ADDR) == EEPROM_ROLE_MAGIC
    assert core.bridge.eeprom_read(EEPROM_ROLE_ADDR + 1) == role_left


def test_eeprom_calibration_save_load_round_trip(front_board_rig):
    core, _, _, _ = front_board_rig
    actuator = "FLHL"
    min_stop, max_stop = 142, 891

    core.reset()
    core.bridge.set_actuator_stops(actuator, min_stop, max_stop)
    core.bridge.save_calibration()

    magic = struct.unpack_from("<I", bytes(core.bridge.eeprom_read(i) for i in range(4)))[0]
    assert magic == CAL_MAGIC

    core.bridge.set_actuator_stops(actuator, 0, 1023)
    core.bridge.load_calibration()

    assert core.bridge.actuator_min_stop(actuator) == min_stop
    assert core.bridge.actuator_max_stop(actuator) == max_stop


def test_hall_edges_advance_on_pwm_jog(front_board_rig):
    core, robot, config, indices = front_board_rig
    hall_slot = config.slot_by_firmware_id(SLOT).slot_index
    joint_index = indices[config.slot_by_firmware_id(SLOT).isaac_joint_name]

    park_at_default_pose(core, robot)
    core.step()
    start = core.bridge.hall_edge_count(hall_slot)
    start_pos = robot.data.joint_pos[0, joint_index].item()

    core.jog(SLOT, 255)
    for _ in range(JOG_STEPS):
        core.step()

    assert core.bridge.hall_edge_count(hall_slot) > start
    assert robot.data.joint_pos[0, joint_index].item() > start_pos + 0.04


def test_host_bridge_exposes_sensor_read_surface(front_board_rig):
    bridge = front_board_rig[0].bridge
    for name in (
        "analog_read",
        "pin_digital",
        "hall_edge_count",
        "eeprom_read",
        "eeprom_update",
        "save_calibration",
        "load_calibration",
        "millis",
    ):
        assert hasattr(bridge, name)


def test_pin_is_mapped_from_physics_on_jog(front_board_rig):
    core, _, config, _ = front_board_rig
    slot = config.slot_by_firmware_id(SLOT)

    park_at_default_pose(core, core.articulation)
    core.step()
    idle = core.bridge.analog_read(slot.pin_is)

    core.jog(SLOT, 200)
    for _ in range(JOG_STEPS):
        core.step()
    moving = core.bridge.analog_read(slot.pin_is)

    assert idle == 0
    assert moving >= DEFAULT_MOVING_NO_CONTACT_ADC


def test_current_sense_idle_vs_moving_breakpoints(front_board_rig):
    """Idle ~0 ADC, swinging leg ~2 (zero-g scene)."""
    core, _, config, _ = front_board_rig
    slot = config.slot_by_firmware_id(SLOT)

    park_at_default_pose(core, core.articulation)
    for _ in range(5):
        core.jog(SLOT, 0)
        core.step()
    assert core.bridge.analog_read(slot.pin_is) == 0

    core.jog(SLOT, 180)
    for _ in range(JOG_STEPS):
        core.step()
    assert core.bridge.analog_read(slot.pin_is) == DEFAULT_MOVING_NO_CONTACT_ADC


def test_millis_tracks_physics_step_dt(front_board_rig):
    core, _, _, _ = front_board_rig
    dt_ms = physics_step_ms(core.env)
    core.reset()

    steps = 25
    for _ in range(steps):
        core.step()

    assert core.millis() == steps * dt_ms


def test_pwm_jog_changes_pot_hall_and_pin_is(front_board_rig):
    core, robot, config, _ = front_board_rig
    slot = config.slot_by_firmware_id(SLOT)
    hall_slot = slot.slot_index

    park_at_default_pose(core, robot)
    core.step()
    pot0 = core.bridge.analog_read(slot.pin_pot)
    hall0 = core.bridge.hall_edge_count(hall_slot)
    is0 = core.bridge.analog_read(slot.pin_is)

    core.jog(SLOT, 255)
    for _ in range(JOG_STEPS):
        core.step()

    assert core.bridge.analog_read(slot.pin_pot) > pot0
    assert core.bridge.hall_edge_count(hall_slot) > hall0
    assert core.bridge.analog_read(slot.pin_is) >= is0


def test_pin_is_rises_when_leg_loads_ground(gravity_front_board):
    from firmware.sim.current_sense_mapping import DEFAULT_GROUND_CONTACT_ADC

    core, _robot, config = gravity_front_board
    slot = config.slot_by_firmware_id("FLKL")
    core.reset()

    for _ in range(100):
        advance_sim(core.env)

    core.jog("FLKL", 200)
    for _ in range(80):
        core.step()

    adc = core.bridge.analog_read(slot.pin_is)
    assert adc >= DEFAULT_GROUND_CONTACT_ADC, (
        f"expected pinIS >= {DEFAULT_GROUND_CONTACT_ADC} under load, got {adc}"
    )


def test_pin_is_hits_lift_band_under_significant_load(gravity_front_board):
    """Mapped pinIS reaches ~30 when front knees/hips drive body lift load (OVERVIEW 2g)."""
    core, _robot, config = gravity_front_board
    core.reset()

    for _ in range(LIFT_SETTLE_STEPS):
        advance_sim(core.env)

    for actuator_id in FRONT_LIFT_ACTUATORS:
        core.jog(actuator_id, LIFT_JOG_PWM)

    peak = _max_pin_is_over_steps(core, config, FRONT_LIFT_ACTUATORS, LIFT_JOG_STEPS)
    assert peak >= LIFT_ADC_MIN, (
        f"expected pinIS near {DEFAULT_LIFT_LOAD_ADC} under lift-scale load, peak {peak}"
    )
