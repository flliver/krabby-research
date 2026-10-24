"""Host firmware V / H / C via serial inject (SDK-compatible V parsing)."""

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


def _read_usb_line(host) -> str:
    chars: list[str] = []
    while host.serial_usb_tx_available():
        byte = host.serial_usb_tx_read()
        if byte == ord("\n"):
            break
        if byte != ord("\r"):
            chars.append(chr(byte))
    return "".join(chars)


def _drain_usb_tx(host) -> list[str]:
    lines: list[str] = []
    while host.serial_usb_tx_available():
        line = _read_usb_line(host)
        if line:
            lines.append(line)
    return lines


def _loop(host, wire: bytes) -> None:
    host.reset()
    host.serial_usb_rx_write(wire)
    host.firmware_setup()
    _drain_usb_tx(host)
    host.firmware_loop_once()


def test_v_command_reply_is_parseable_by_sdk(host):
    _loop(host, b"V\n")
    line = _read_usb_line(host)
    parsed = parse_ver_reply(line)
    assert parsed is not None
    assert len(parsed) == 3
    assert parsed[0][0] != "-"


def test_h_command_deenergizes_all_front_joints(host):
    config = SimulatedRobotConfig.front_board_rev3()
    parts = ["B"]
    for slot in config.slots:
        parts.append(slot.firmware_actuator_id)
        parts.append("80")
    _loop(host, (" ".join(parts) + " \n").encode("ascii"))
    for slot in config.slots:
        assert host.pin_digital(slot.pin_en) == 1

    host.serial_usb_rx_write(b"H\n")
    host.firmware_loop_once()
    for slot in config.slots:
        assert host.pin_digital(slot.pin_en) == 0


def test_c_command_starts_auto_calibration(host):
    _loop(host, b"C\n")
    line = _read_usb_line(host)
    assert "Starting Auto-Calibration" in line
