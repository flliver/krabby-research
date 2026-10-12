"""Host ~50 ms telemetry wire format: golden FRONT line + LEFT prefix contract."""

import pytest

from firmware.interfaces.joint_telemetry import JointTelemetry
from firmware.interfaces.telemetry_frame import TelemetryFrame
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.simulated_robot_config import SimulatedRobotConfig

TELEMETRY_INTERVAL_MS = 50

# Deterministic host line: all six FRONT joints at mid travel, pots/IS staged before setup.
GOLDEN_FRONT_TELEMETRY_LINE = (
    "FRONT; FLHY 0.500 500 12 0 0 0 0 0 0"
    ";FLHL 0.500 500 12 0 0 0 0 0 0"
    ";FLKL 0.500 500 12 0 0 0 0 0 0"
    ";FRHY 0.500 500 12 0 0 0 0 0 0"
    ";FRHL 0.500 500 12 0 0 0 0 0 0"
    ";FRKL 0.500 500 12 0 0 0 0 0 0"
)

# Shipping follower prefix uses a trailing space before ';' (see boardTelemetryRoleLabel).
GOLDEN_LEFT_PREFIX_LINE = "LEFT ; RLHY 0.123 512 12 1 0 0 128 3 1"


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


def _stage_front_board_sensors(host) -> None:
    config = SimulatedRobotConfig.front_board_rev3()
    for slot in config.slots:
        host.set_actuator_stops(slot.firmware_actuator_id, 100, 900)
        host.set_analog_read(slot.pin_pot, 500)
        host.set_analog_read(slot.pin_is, 12)


def _emit_telemetry_line(host) -> str:
    host.firmware_setup()
    _drain_usb_tx(host)
    host.advance_millis(TELEMETRY_INTERVAL_MS)
    host.firmware_loop_once()
    return _read_usb_line(host)


def test_host_emits_golden_front_telemetry_line(host):
    host.reset()
    _stage_front_board_sensors(host)
    line = _emit_telemetry_line(host)
    assert line == GOLDEN_FRONT_TELEMETRY_LINE


def test_golden_front_line_parses_with_existing_telemetry_frame(host):
    host.reset()
    _stage_front_board_sensors(host)
    line = _emit_telemetry_line(host)
    frame = TelemetryFrame.parse_line(line)
    assert TelemetryFrame.role_from_line(line) == "FRONT"
    assert len(frame.joints) == 6
    assert [joint.name for joint in frame.joints] == [
        "FLHY",
        "FLHL",
        "FLKL",
        "FRHY",
        "FRHL",
        "FRKL",
    ]
    for segment in line.split(";")[1:]:
        tokens = segment.split()
        assert len(tokens) == 10
        assert JointTelemetry.from_tokens(tokens) is not None


def test_left_prefix_golden_line_parses_without_parser_changes():
    assert TelemetryFrame.role_from_line(GOLDEN_LEFT_PREFIX_LINE) == "LEFT"
    frame = TelemetryFrame.parse_line(GOLDEN_LEFT_PREFIX_LINE)
    assert len(frame.joints) == 1
    assert frame.joints[0].name == "RLHY"
