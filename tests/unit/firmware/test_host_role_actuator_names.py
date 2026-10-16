"""After three-board role election, each role's telemetry must use that role's joint names.

Ensures create-after-election does not wire FRONT names onto LEFT/RIGHT (or vice versa).
"""

from __future__ import annotations

import pytest

from firmware.interfaces.telemetry_frame import TelemetryFrame
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.simulated_robot_config import SimulatedRobotConfig
from firmware.sim.three_board_boot import (
    ROLE_FRONT,
    ROLE_LEFT,
    ROLE_RIGHT,
    assert_one_of_each_role,
    boot_three_board_emulators,
)

TELEMETRY_INTERVAL_MS = 50

EXPECTED_NAMES_BY_ROLE = {
    "FRONT": tuple(
        slot.firmware_actuator_id
        for slot in SimulatedRobotConfig.front_board_rev3().slots
    ),
    "LEFT": tuple(
        slot.firmware_actuator_id
        for slot in SimulatedRobotConfig.left_board_rev3().slots
    ),
    "RIGHT": tuple(
        slot.firmware_actuator_id
        for slot in SimulatedRobotConfig.right_board_rev3().slots
    ),
}


@pytest.fixture
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


def _collect_role_telemetry_lines(host, *, max_ticks: int = 10) -> dict[str, str]:
    """Advance the three-board loop until FRONT/LEFT/RIGHT telemetry lines appear."""
    found: dict[str, str] = {}
    for _ in range(max_ticks):
        host.advance_millis(TELEMETRY_INTERVAL_MS)
        host.firmware_loop_once()
        while host.serial_usb_tx_available():
            line = _read_usb_line(host)
            role = TelemetryFrame.role_from_line(line)
            if role in EXPECTED_NAMES_BY_ROLE and role not in found:
                found[role] = line
        if len(found) == len(EXPECTED_NAMES_BY_ROLE):
            break
    return found


def test_three_board_election_assigns_one_of_each_role(host):
    roles = boot_three_board_emulators(host)
    assert_one_of_each_role(roles)
    assert roles["front"] == ROLE_FRONT
    assert roles["left"] == ROLE_LEFT
    assert roles["right"] == ROLE_RIGHT


def test_telemetry_joint_names_match_role_after_election(host):
    boot_three_board_emulators(host)
    host.enable_three_board_firmware_loop(True)
    host.firmware_setup()
    _drain_usb_tx(host)

    lines = _collect_role_telemetry_lines(host)
    assert set(lines) == set(EXPECTED_NAMES_BY_ROLE), f"missing roles in telemetry: {lines.keys()}"

    for role, line in lines.items():
        frame = TelemetryFrame.parse_line(line)
        names = tuple(joint.name for joint in frame.joints)
        assert names == EXPECTED_NAMES_BY_ROLE[role], (
            f"{role} telemetry names {names} != {EXPECTED_NAMES_BY_ROLE[role]}"
        )
