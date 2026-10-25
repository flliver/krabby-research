"""Three simulated MCU emulators elect FRONT/LEFT/RIGHT; leader USB is FRONT."""

from __future__ import annotations

import os

import pytest

from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.three_board_boot import (
    LEADER_BOARD_ID,
    ROLE_FRONT,
    ROLE_LEFT,
    ROLE_RIGHT,
    assert_one_of_each_role,
    boot_three_board_emulators,
)


@pytest.fixture
def bridge():
    try:
        host = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    return host


def test_three_board_role_counts(bridge):
    roles = boot_three_board_emulators(bridge)
    assert_one_of_each_role(roles)
    assert roles["front"] == ROLE_FRONT
    assert roles["left"] == ROLE_LEFT
    assert roles["right"] == ROLE_RIGHT


def test_leader_usb_port_is_front_board(bridge):
    roles = boot_three_board_emulators(bridge)
    assert_one_of_each_role(roles)
    assert bridge.leader_board_id() == LEADER_BOARD_ID

    front_lines = bridge.drain_board_usb_lines(LEADER_BOARD_ID)
    left_lines = bridge.drain_board_usb_lines(1)
    right_lines = bridge.drain_board_usb_lines(2)

    assert any("ROLE: FRONT" in line for line in front_lines)
    assert any("ROLE: LEFT" in line for line in left_lines)
    assert any("ROLE: RIGHT" in line for line in right_lines)
    assert not any("ROLE: FRONT" in line for line in left_lines + right_lines)


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_three_board_host_exposes_leader_pty(bridge):
    from firmware.sim.three_board_mcu_host import ThreeBoardMcuHost

    host = ThreeBoardMcuHost(bridge)
    try:
        path = host.start()
        assert path
        assert host.slave_path == path
        assert_one_of_each_role(host.roles_by_board)
        assert host.leader_board_id == LEADER_BOARD_ID
    finally:
        host.close()
