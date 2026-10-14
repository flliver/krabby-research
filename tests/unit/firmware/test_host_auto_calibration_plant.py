"""Host: auto-cal stop-to-stop against a minimal pot plant (no Isaac)."""

from __future__ import annotations

import struct

import pytest

from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.simulated_robot_config import SimulatedRobotConfig

CAL_MAGIC = 0xDEADBEEF
CAL_DATA_MAGIC_OFFSET = 6 * 4 + 6 * 4
ADC_TOLERANCE = 4
FIRMWARE_DT_MS = 10
MAX_CAL_TICKS = 4_000


@pytest.fixture(scope="module")
def host():
    try:
        return load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))


def _signed_pwm(bridge, slot) -> int:
    pwm_r, pwm_l, pin_en = slot.drive_pins
    if bridge.pin_digital(pin_en) == 0:
        return 0
    return bridge.pin_pwm(pwm_r) - bridge.pin_pwm(pwm_l)


def _sync_pot_plant(bridge, config, travel: dict[str, float]) -> None:
    for slot in config.slots:
        band = config.pot_band_for_slot(slot.slot_index)
        frac = travel[slot.firmware_actuator_id]
        adc = int(round(band.min_adc + frac * (band.max_adc - band.min_adc)))
        bridge.set_analog_read(slot.pin_pot, adc)


def _advance_plant(bridge, config, travel: dict[str, float], step: float = 0.06) -> None:
    for slot in config.slots:
        cmd = _signed_pwm(bridge, slot)
        aid = slot.firmware_actuator_id
        if cmd >= 50:
            travel[aid] = min(1.0, travel[aid] + step)
        elif cmd <= -50:
            travel[aid] = max(0.0, travel[aid] - step)


def _drain_usb(host) -> list[str]:
    lines: list[str] = []
    while host.serial_usb_tx_available():
        chars: list[str] = []
        while host.serial_usb_tx_available():
            byte = host.serial_usb_tx_read()
            if byte == ord("\n"):
                break
            if byte != ord("\r"):
                chars.append(chr(byte))
        line = "".join(chars)
        if line:
            lines.append(line)
    return lines


def test_auto_calibration_discovers_plant_pot_limits(host):
    config = SimulatedRobotConfig.front_board_rev3()
    travel = {slot.firmware_actuator_id: 0.5 for slot in config.slots}

    host.reset()
    for slot in config.slots:
        band = config.pot_band_for_slot(slot.slot_index)
        host.set_actuator_stops(slot.firmware_actuator_id, band.min_adc, band.max_adc)
    host.firmware_setup()
    _drain_usb(host)
    host.serial_usb_rx_write(b"C\n")

    log: list[str] = []
    for _ in range(MAX_CAL_TICKS):
        _sync_pot_plant(host, config, travel)
        host.advance_millis(FIRMWARE_DT_MS)
        host.firmware_loop_once()
        _advance_plant(host, config, travel)
        log.extend(_drain_usb(host))
        if any("CALIBRATION COMPLETE" in line for line in log):
            break
    else:
        pytest.fail("host auto-cal did not complete")

    magic_bytes = bytes(
        host.eeprom_read(i) for i in range(CAL_DATA_MAGIC_OFFSET, CAL_DATA_MAGIC_OFFSET + 4)
    )
    assert struct.unpack_from("<I", magic_bytes)[0] == CAL_MAGIC

    for slot in config.slots:
        band = config.pot_band_for_slot(slot.slot_index)
        aid = slot.firmware_actuator_id
        mn = host.actuator_min_stop(aid)
        mx = host.actuator_max_stop(aid)
        assert mn < mx
        assert abs(mn - band.min_adc) <= ADC_TOLERANCE
        assert abs(mx - band.max_adc) <= ADC_TOLERANCE

    for slot in config.slots:
        host.set_actuator_stops(slot.firmware_actuator_id, 0, 1023)
    host.firmware_reset_loop_state()
    host.firmware_setup()
    for slot in config.slots:
        aid = slot.firmware_actuator_id
        band = config.pot_band_for_slot(slot.slot_index)
        assert abs(host.actuator_min_stop(aid) - band.min_adc) <= ADC_TOLERANCE
        assert abs(host.actuator_max_stop(aid) - band.max_adc) <= ADC_TOLERANCE
