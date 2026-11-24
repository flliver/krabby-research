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


ROBOTS = {
    "rev2_pot": SimulatedRobotConfig.rev2_pot,
    "rev3_hall": SimulatedRobotConfig.rev3_hall,
    "mixed": SimulatedRobotConfig.mixed,
}


@pytest.fixture(params=list(ROBOTS))
def robot(request) -> SimulatedRobotConfig:
    return ROBOTS[request.param]()


def _loop(host, wire: bytes, config: SimulatedRobotConfig | None = None) -> None:
    host.reset()
    if config is not None and config.quadrature:
        # Cleared by the reset; read when the joints are set up.
        host.board_set_hall_quadrature(config.board_id, True)
    host.serial_usb_rx_write(wire)
    host.firmware_setup()
    _drain_usb_tx(host)
    host.firmware_loop_once()


def _cal_lines(host) -> list[str]:
    return [line for line in _drain_usb_tx(host) if line.startswith("CAL ")]


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
    lines = _drain_usb_tx(host)
    assert lines[:3] == [
        "CAL ROBOT START pwm=128 passes=SEEK,SWEEP",
        "CAL FRONT PASS SEEK pwm=128",
        "CAL FRONT MOVE FLKL curl",
    ]
    assert host.firmware_calibration_active()


def test_c_command_starts_auto_calibration_on_every_robot(host, robot):
    _loop(host, b"C\n", robot)
    # A quadrature board also prints its SENS line; the CAL lines are the same.
    assert _cal_lines(host)[:3] == [
        "CAL ROBOT START pwm=128 passes=SEEK,SWEEP",
        "CAL FRONT PASS SEEK pwm=128",
        "CAL FRONT MOVE FLKL curl",
    ]
    assert host.firmware_calibration_active()


def test_h_command_aborts_auto_calibration(host, robot):
    _loop(host, b"C\n", robot)
    _drain_usb_tx(host)
    host.serial_usb_rx_write(b"H\n")
    host.firmware_loop_once()
    lines = _drain_usb_tx(host)
    assert "CAL FRONT ABORTED H" in lines
    assert "CAL ROBOT ABORTED H" in lines
    assert not host.firmware_calibration_active()


def test_v_command_still_answers_during_calibration(host, robot):
    _loop(host, b"C\n", robot)
    _drain_usb_tx(host)
    host.serial_usb_rx_write(b"V\n")
    host.firmware_loop_once()
    replies = [line for line in _drain_usb_tx(host) if line.startswith("VER ")]
    assert len(replies) == 1
    assert parse_ver_reply(replies[0]) is not None
    assert host.firmware_calibration_active()
