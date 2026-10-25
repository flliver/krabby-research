"""Boot three simulated MCU boards and elect FRONT/LEFT/RIGHT roles."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from firmware.sim.host_bridge import HostActuatorBridge

# Matches BoardId order in sim_serial.hpp and ROLE_* in board_role.h.
BOARD_IDS = ("front", "left", "right")
ROLE_UNKNOWN = 0
ROLE_FRONT = 1
ROLE_LEFT = 2
ROLE_RIGHT = 3
LEADER_BOARD_ID = 0


def boot_three_board_emulators(bridge: HostActuatorBridge) -> dict[str, int]:
    """Reset sim state, run SYNC election, return role enum per board name."""
    bridge.reset()
    roles = bridge.run_three_board_role_election()
    return {
        BOARD_IDS[i]: roles[i]
        for i in range(3)
    }


def assert_one_of_each_role(roles_by_board: dict[str, int]) -> None:
    values = list(roles_by_board.values())
    if sorted(values) != [ROLE_FRONT, ROLE_LEFT, ROLE_RIGHT]:
        raise AssertionError(f"expected one FRONT/LEFT/RIGHT, got {roles_by_board}")
