"""Host ~50 ms telemetry wire format: golden FRONT line + LEFT prefix contract."""

import math

import pytest

from firmware.interfaces.joint_telemetry import ActuatorConnection, JointTelemetry
from firmware.interfaces.telemetry_frame import TelemetryFrame
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.pot_autocal_plant import (
    HALL_B_REST_HIGH_ADC,
    HALL_B_REST_LOW_ADC,
    hall_b_level,
)
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
    ";IMU 0.000 0.000 9.806 0.0000 0.0000 0.0000 25.0 1"
    ";BATT 26.40 1.250 33.00 100000.0 13.20 13.20 0 0"
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
    for segment in line.split(";")[1:7]:
        tokens = segment.split()
        assert len(tokens) == 10
        assert JointTelemetry.from_tokens(tokens) is not None
    frame = TelemetryFrame.parse_line(line)
    assert frame.imu is not None
    assert frame.battery is not None


def test_left_prefix_golden_line_parses_without_parser_changes():
    assert TelemetryFrame.role_from_line(GOLDEN_LEFT_PREFIX_LINE) == "LEFT"
    frame = TelemetryFrame.parse_line(GOLDEN_LEFT_PREFIX_LINE)
    assert len(frame.joints) == 1
    assert frame.joints[0].name == "RLHY"


def _advance_until_front_telemetry(host, *, max_ticks: int = 30) -> str:
    for _ in range(max_ticks):
        host.advance_millis(10)
        host.firmware_loop_once()
        if host.serial_usb_tx_available():
            line = _read_usb_line(host)
            if line.startswith("FRONT;"):
                return line
    raise AssertionError("no FRONT telemetry line on USB serial")


def test_front_telemetry_en_fields_follow_jog_on_usb_serial(host):
    """``printTelemetry`` enL/enR on the role-prefixed USB line track ``digitalRead(pinEn)``."""
    host.reset()
    _stage_front_board_sensors(host)
    host.firmware_setup()
    _drain_usb_tx(host)

    host.serial_usb_rx_write(b"JFLHY 180\n")
    line = _advance_until_front_telemetry(host)
    assert TelemetryFrame.role_from_line(line) == "FRONT"
    flhy = next(j for j in TelemetryFrame.parse_line(line).joints if j.name == "FLHY")
    assert flhy.en == (1, 1)

    host.serial_usb_rx_write(b"JFLHY 0\n")
    for _ in range(30):
        host.advance_millis(10)
        host.firmware_loop_once()
    line = _advance_until_front_telemetry(host)
    flhy = next(j for j in TelemetryFrame.parse_line(line).joints if j.name == "FLHY")
    assert flhy.en == (0, 0)


# Quadrature board, every slot a hall encoder, nothing homed: pos is nan and the
# state disconnected (2), as for every hall joint before; field 3 is the hall B
# level (pull-up kept, resting on a rail) and field 9 the signed count.
HALL_COUNTS = {"FLHY": 0, "FLHL": 37, "FLKL": -42, "FRHY": 0, "FRHL": -1, "FRKL": 1584}
GOLDEN_FRONT_HALL_TELEMETRY_LINE = (
    "FRONT; FLHY nan 19 12 0 0 0 0 0 2"
    ";FLHL nan 1016 12 0 0 0 0 37 2"
    ";FLKL nan 19 12 0 0 0 0 -42 2"
    ";FRHY nan 19 12 0 0 0 0 0 2"
    ";FRHL nan 1016 12 0 0 0 0 -1 2"
    ";FRKL nan 19 12 0 0 0 0 1584 2"
    ";IMU 0.000 0.000 9.806 0.0000 0.0000 0.0000 25.0 1"
    ";BATT 26.40 1.250 33.00 100000.0 13.20 13.20 0 0"
)
GOLDEN_FRONT_HALL_SENS_LINE = "SENS FRONT FLHY:? FLHL:? FLKL:? FRHY:? FRHL:? FRKL:?"


def _stage_front_board_halls(host) -> None:
    config = SimulatedRobotConfig.rev3_hall()
    host.board_set_hall_quadrature(config.board_id, True)
    for slot in config.slots:
        host.set_analog_read(slot.pin_pot, hall_b_level(HALL_COUNTS[slot.firmware_actuator_id]))
        host.set_analog_read(slot.pin_is, 12)


def _front_lines_after_setup(host) -> list[str]:
    host.firmware_setup()
    _drain_usb_tx(host)
    # hallHwInit() zeroed the counts at setup.
    config = SimulatedRobotConfig.rev3_hall()
    for slot in config.slots:
        host.board_set_hall_edge_count(
            config.board_id, slot.slot_index, HALL_COUNTS[slot.firmware_actuator_id]
        )
    host.advance_millis(TELEMETRY_INTERVAL_MS)
    host.firmware_loop_once()
    lines = []
    while host.serial_usb_tx_available():
        lines.append(_read_usb_line(host))
    return lines


def test_host_emits_golden_hall_telemetry_line_before_homing(host):
    host.reset()
    _stage_front_board_halls(host)
    lines = _front_lines_after_setup(host)
    assert GOLDEN_FRONT_HALL_TELEMETRY_LINE in lines
    # Printed at boot, on a change and every 2 s (the first is not guaranteed here:
    # the sim reset keeps the manager's last SENS state).
    for _ in range(210):
        host.advance_millis(10)
        host.firmware_loop_once()
        while host.serial_usb_tx_available():
            lines.append(_read_usb_line(host))
    assert GOLDEN_FRONT_HALL_SENS_LINE in lines
    # The SENS line is not telemetry and never parses as joints.
    assert not TelemetryFrame.is_telemetry_line(GOLDEN_FRONT_HALL_SENS_LINE)
    assert TelemetryFrame.parse_line(GOLDEN_FRONT_HALL_SENS_LINE).joints == ()


def test_golden_hall_line_parses_with_signed_counts_and_no_position(host):
    frame = TelemetryFrame.parse_line(GOLDEN_FRONT_HALL_TELEMETRY_LINE)
    assert TelemetryFrame.role_from_line(GOLDEN_FRONT_HALL_TELEMETRY_LINE) == "FRONT"
    assert [joint.name for joint in frame.joints] == list(HALL_COUNTS)
    for joint in frame.joints:
        assert math.isnan(joint.pos)
        assert joint.connection_state == ActuatorConnection.DISCONNECTED
        assert joint.hall_edges == HALL_COUNTS[joint.name]
        assert joint.pot in (HALL_B_REST_LOW_ADC, HALL_B_REST_HIGH_ADC)


def test_default_board_keeps_the_pot_golden_and_prints_no_sens_line(host):
    host.reset()
    _stage_front_board_sensors(host)
    lines = _front_lines_after_setup(host)
    for _ in range(210):
        host.advance_millis(10)
        host.firmware_loop_once()
        while host.serial_usb_tx_available():
            lines.append(_read_usb_line(host))
    assert not any(line.startswith("SENS ") for line in lines)
    # Without quadrature the counts are telemetry only (field 9); the rest is the pot golden.
    segments = GOLDEN_FRONT_TELEMETRY_LINE.split(";")
    for index, name in enumerate(HALL_COUNTS, start=1):
        segment = segments[index]
        tokens = segment.split()
        tokens[8] = str(HALL_COUNTS[name])
        segments[index] = segment[: len(segment) - len(segment.lstrip())] + " ".join(tokens)
    assert ";".join(segments) in lines
