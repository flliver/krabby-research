"""Leg/joint numbers and krabby.dbc CAN ID helpers (11-bit)."""

from __future__ import annotations

from typing import Final

# --- Leg / joint IDs (1-based on the wire and in the CLI) ---

LEG_IDS: Final = (1, 2, 3, 4, 5, 6)
LEG_IDS_LEFT: Final = (1, 3, 5)
LEG_IDS_RIGHT: Final = (2, 4, 6)

JOINT_IDS: Final = (1, 2, 3)
JOINT_Y: Final = 1
JOINT_H: Final = 2
JOINT_K: Final = 3
JOINT_LETTERS: Final = "YHK"

LEG_CODE_UNASSIGNED: Final = 255
LEG_INDEX_UNASSIGNED: Final = 6

LEG_BUS_BY_ID: Final = {
    1: "can0",
    2: "can1",
    3: "can0",
    4: "can1",
    5: "can0",
    6: "can1",
}

LEG_ID_BY_DBC_SUFFIX: Final = {
    "FL": 1,
    "FR": 2,
    "ML": 3,
    "MR": 4,
    "RL": 5,
    "RR": 6,
}

# krabby.dbc message IDs
CAN_ID_SET_DUTY: Final = 0x100
CAN_ID_CALIBRATE_BASE: Final = 0x180
CAN_ID_TELEMETRY_BASE: Final = 0x200
CAN_ID_BOARD_INFO_BASE: Final = 0x300


def leg_id_to_index(leg_id: int) -> int:
    if not 1 <= leg_id <= 6:
        raise ValueError(f"leg_id out of range: {leg_id!r}")
    return leg_id - 1


def leg_index_to_id(leg_index: int) -> int:
    if not 0 <= leg_index <= 5:
        raise ValueError(f"leg_index out of range: {leg_index!r}")
    return leg_index + 1


def joint_id_to_index(joint_id: int) -> int:
    if joint_id not in JOINT_IDS:
        raise ValueError(f"joint_id out of range: {joint_id!r}")
    return joint_id - 1


def joint_index_to_id(joint_index: int) -> int:
    if not 0 <= joint_index <= 2:
        raise ValueError(f"joint_index out of range: {joint_index!r}")
    return joint_index + 1


def set_duty_can_id() -> int:
    return CAN_ID_SET_DUTY


def calibrate_can_id(leg_id: int) -> int:
    if leg_id not in LEG_IDS:
        raise ValueError(f"leg_id out of range: {leg_id!r}")
    return CAN_ID_CALIBRATE_BASE + leg_id


def calibrate_can_id_for_index(leg_index: int) -> int:
    return calibrate_can_id(leg_index_to_id(leg_index))


def telemetry_can_id(leg_id: int) -> int:
    if leg_id not in LEG_IDS:
        raise ValueError(f"leg_id out of range: {leg_id!r}")
    return CAN_ID_TELEMETRY_BASE + leg_id


def telemetry_can_id_for_index(leg_index: int) -> int:
    return telemetry_can_id(leg_index_to_id(leg_index))


def board_info_can_id(leg_id: int) -> int:
    if leg_id not in LEG_IDS:
        raise ValueError(f"leg_id out of range: {leg_id!r}")
    return CAN_ID_BOARD_INFO_BASE + leg_id


def board_info_can_id_for_index(leg_index: int) -> int:
    return board_info_can_id(leg_index_to_id(leg_index))


def telemetry_leg_id_from_frame(can_id: int) -> int | None:
    """Inverse of ``telemetry_can_id``; ``None`` if *can_id* is not a leg telemetry frame."""
    leg_id = can_id - CAN_ID_TELEMETRY_BASE
    return leg_id if leg_id in LEG_IDS else None


def board_info_leg_id_from_frame(can_id: int) -> int | None:
    """Inverse of ``board_info_can_id``; ``None`` if *can_id* is not a leg board-info frame."""
    leg_id = can_id - CAN_ID_BOARD_INFO_BASE
    return leg_id if leg_id in LEG_IDS else None


def leg_bus(leg_id: int) -> str:
    return LEG_BUS_BY_ID[leg_id]
