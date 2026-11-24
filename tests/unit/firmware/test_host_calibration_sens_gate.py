"""Host: the mixed-firmware gate of the robot calibration (review finding R1-8).

On a quadrature leader, ``C`` is refused while a linked follower has printed no
``SENS`` line since the leader booted: only this firmware on a quadrature build
prints SENS (at boot and every 2 s), so a silent follower runs other firmware or
has not booted yet. The refusal moves nothing. Three boards run the host sim's
three-board loop here; no plant is attached, so a run that does start is stopped
with ``H`` at once.
"""

from __future__ import annotations

import pytest

from firmware.krabby_mcu import cal_line_state, cal_robot_finished
from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.simulated_robot_config import KRABBY_HEX_FIRMWARE_ACTUATOR_IDS
from firmware.sim.three_board_boot import LEADER_BOARD_ID

LOOP_MS = 10
# ACTUATOR_SENS_PERIOD_MS (2000) plus margin, in loops.
SENS_PERIOD_LOOPS = 2100 // LOOP_MS
BOARDS = {"FRONT": 0, "LEFT": 1, "RIGHT": 2}
START = "CAL ROBOT START pwm=128 passes=SEEK"


def _no_sens(role: str) -> str:
    return (
        f"CAL ROBOT FAIL {role}: no SENS: flash all three boards with the same firmware, "
        "or wait a few seconds after connecting"
    )


@pytest.fixture
def host():
    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    return bridge


class _ThreeBoards:
    def __init__(self, host, quadrature_roles: tuple[str, ...]) -> None:
        self.host = host
        self.log: list[str] = []
        host.reset()
        # Cleared by the reset; read when each board sets up its joints.
        for role in quadrature_roles:
            host.board_set_hall_quadrature(BOARDS[role], True)
        host.run_three_board_role_election()
        host.enable_three_board_firmware_loop(True)
        host.firmware_setup()

    def loop(self, count: int = 1) -> None:
        for _ in range(count):
            self.host.advance_millis(LOOP_MS)
            self.host.firmware_loop_once()
            self.log.extend(self.host.drain_board_usb_lines(LEADER_BOARD_ID))
            self._check_nothing_driven()

    def _check_nothing_driven(self) -> None:
        driven = {
            aid: pwm
            for aid in KRABBY_HEX_FIRMWARE_ACTUATOR_IDS
            if (pwm := self.host.three_board_actuator_command_pwm(aid)) != 0
        }
        assert driven == {}, driven

    def send(self, wire: bytes) -> None:
        self.host.serial_usb_rx_write(wire)
        self.loop()

    def cal_lines(self) -> list[str]:
        return [line for line in self.log if line.startswith("CAL ")]


def _stop(boards: _ThreeBoards) -> None:
    """A run that started would drive a joint with no plant: stop it at once."""
    boards.host.serial_usb_rx_write(b"H\n")
    boards.host.advance_millis(LOOP_MS)
    boards.host.firmware_loop_once()
    boards.host.drain_board_usb_lines(LEADER_BOARD_ID)
    assert not boards.host.firmware_calibration_active()


@pytest.mark.parametrize(
    ("quadrature_roles", "silent"),
    [
        # Both followers run a build that prints no SENS line: LEFT is named first.
        (("FRONT",), "LEFT"),
        # Only RIGHT is silent.
        (("FRONT", "LEFT"), "RIGHT"),
    ],
)
def test_a_follower_without_sens_lines_refuses_the_run(host, quadrature_roles, silent):
    boards = _ThreeBoards(host, quadrature_roles)
    boards.loop(SENS_PERIOD_LOOPS)
    assert any(line.startswith("SENS FRONT ") for line in boards.log)
    boards.log.clear()

    boards.send(b"C SEEK\n")
    boards.loop(300)

    lines = boards.cal_lines()
    assert lines == [_no_sens(silent)]
    assert cal_robot_finished(lines[0])
    assert cal_line_state(lines[0]) == "fail"
    assert not host.firmware_calibration_active()


def test_c_right_after_connecting_waits_for_the_followers_sens_lines(host):
    boards = _ThreeBoards(host, ("FRONT", "LEFT", "RIGHT"))
    boards.send(b"C SEEK\n")  # the first loop: no follower has printed SENS yet
    boards.loop(50)
    assert boards.cal_lines() == [_no_sens("LEFT")]
    assert not host.firmware_calibration_active()

    boards.loop(SENS_PERIOD_LOOPS)
    assert any(line.startswith("SENS LEFT ") for line in boards.log)
    assert any(line.startswith("SENS RIGHT ") for line in boards.log)
    boards.log.clear()
    host.serial_usb_rx_write(b"C SEEK\n")
    host.advance_millis(LOOP_MS)
    host.firmware_loop_once()
    lines = host.drain_board_usb_lines(LEADER_BOARD_ID)
    assert START in lines, lines
    assert not any("no SENS" in line for line in lines)
    _stop(boards)


def test_a_leader_without_quadrature_halls_has_no_gate(host):
    """The sim's default (no quadrature, like Rev 2): no SENS lines at all, no gate."""
    boards = _ThreeBoards(host, ())
    boards.loop(SENS_PERIOD_LOOPS)
    assert not any(line.startswith("SENS ") for line in boards.log)
    boards.log.clear()
    host.serial_usb_rx_write(b"C SEEK\n")
    host.advance_millis(LOOP_MS)
    host.firmware_loop_once()
    lines = host.drain_board_usb_lines(LEADER_BOARD_ID)
    assert START in lines, lines
    _stop(boards)
