"""Host firmware loop USB dispatch using the same wire bytes as KrabbyMCUSDK."""

import pytest

from firmware.krabby_mcu import parse_ver_reply
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.simulated_robot_config import SimulatedRobotConfig


@pytest.fixture(scope="module")
def host():
    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    return bridge


def _drain_usb_tx(host) -> None:
    while host.serial_usb_tx_available():
        host.serial_usb_tx_read()


def _read_usb_line(host) -> str:
    chars: list[str] = []
    while host.serial_usb_tx_available():
        byte = host.serial_usb_tx_read()
        if byte == ord("\n"):
            break
        if byte != ord("\r"):
            chars.append(chr(byte))
    return "".join(chars)


def _wire_joints(cmds_by_joint: dict[str, float]) -> bytes:
    parts = ["T"]
    for name, raw_val in cmds_by_joint.items():
        val = max(0.0, min(1.0, raw_val))
        parts.append(name)
        parts.append(f"{val:.3f}")
    return (" ".join(parts) + "\n").encode("utf-8")


def _wire_jog_batch(cmds_by_joint: dict[str, int]) -> bytes:
    parts = ["B"]
    for name, raw_pwm in cmds_by_joint.items():
        pwm = max(-255, min(255, int(raw_pwm)))
        parts.append(name)
        parts.append(str(pwm))
    return (" ".join(parts) + " \n").encode("utf-8")


def _wire_jog_single(joint_name: str, pwm: int) -> bytes:
    pwm = max(-255, min(255, int(pwm)))
    return f"J{joint_name} {pwm}\n".encode("utf-8")


def _loop(host, wire: bytes) -> None:
    host.reset()
    host.serial_usb_rx_write(wire)
    host.firmware_setup()
    _drain_usb_tx(host)
    host.firmware_loop_once()


@pytest.mark.parametrize(
    ("wire_fn", "joint", "pwm", "en_high"),
    [
        (lambda: _wire_jog_single("FLHY", 100), "FLHY", 100, True),
        (lambda: _wire_jog_batch({"FLHY": -80}), "FLHY", -80, True),
    ],
)
def test_mcu_emitter_jog_wires(host, wire_fn, joint, pwm, en_high):
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id(joint)
    _, _, pin_en = slot.drive_pins
    _loop(host, wire_fn())
    assert host.pin_digital(pin_en) == (1 if en_high else 0)


def test_mcu_emitter_joints_wire_drives_position_target(host):
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id("FLHY")
    pwm_r, _pwm_l, pin_en = slot.drive_pins
    pot_pin = slot.pin_pot

    host.reset()
    host.set_actuator_stops("FLHY", 100, 900)
    host.set_analog_read(pot_pin, 100)
    host.serial_usb_rx_write(_wire_joints({"FLHY": 1.0}))
    host.firmware_setup()
    _drain_usb_tx(host)
    host.firmware_loop_once()
    for _ in range(8):
        host.advance_millis(10)
        host.firmware_loop_once()

    assert host.pin_digital(pin_en) == 1
    assert host.pin_pwm(pwm_r) > 0


def test_mcu_emitter_calibrate_command(host):
    _loop(host, b"C\n")
    line = _read_usb_line(host)
    assert "Starting Auto-Calibration" in line


def test_mcu_emitter_hold_command(host):
    slot = SimulatedRobotConfig.front_board_rev3().slot_by_firmware_id("FLHY")
    _, _, pin_en = slot.drive_pins

    _loop(host, _wire_jog_single("FLHY", 100))
    assert host.pin_digital(pin_en) == 1

    host.serial_usb_rx_write(b"H\n")
    host.firmware_loop_once()
    assert host.pin_digital(pin_en) == 0


def test_mcu_emitter_version_wire(host):
    _loop(host, b"V\n")
    line = _read_usb_line(host)
    parsed = parse_ver_reply(line)
    assert parsed is not None
    assert len(parsed) == 3
    assert parsed[0][0] != "-"
