"""Leg/joint numbers and CAN extended-ID helpers for the leg bus."""

from __future__ import annotations

from typing import Final

# --- Normative operator / firmware identity (1-based) ---

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

# DBC message suffix → leg_id (1–6). HAL corner names are host-only.
LEG_ID_BY_DBC_SUFFIX: Final = {
    "FL": 1,
    "FR": 2,
    "ML": 3,
    "MR": 4,
    "RL": 5,
    "RR": 6,
}

CAN_ID_LEG_COMMAND_BASE: Final = 0x0C000000
CAN_ID_LEG_TELEMETRY_BASE: Final = 0x0C000100
CAN_ID_NODE_STATUS_BASE: Final = 0x0C000200
CAN_ID_QWIIC_IMU_BASE: Final = 0x0C000400
CAN_ID_QWIIC_BATTERY_BASE: Final = 0x0C000500  # QwiicBattery; not in DBC until implemented
CAN_ID_DISCOVERY_REQUEST: Final = 0x0C0007F0
CAN_ID_PROVISION_SET: Final = 0x0C0007E0


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


def leg_command_can_id(leg_index: int) -> int:
    return CAN_ID_LEG_COMMAND_BASE + leg_index


def leg_command_can_id_for_leg(leg_id: int) -> int:
    return leg_command_can_id(leg_id_to_index(leg_id))


def leg_telemetry_can_id(leg_index: int) -> int:
    return CAN_ID_LEG_TELEMETRY_BASE + leg_index


def leg_telemetry_can_id_for_leg(leg_id: int) -> int:
    return leg_telemetry_can_id(leg_id_to_index(leg_id))


def node_status_can_id(leg_index: int) -> int:
    return CAN_ID_NODE_STATUS_BASE + leg_index


def node_status_can_id_for_leg(leg_id: int) -> int:
    return node_status_can_id(leg_id_to_index(leg_id))


def qwiic_imu_can_id(leg_index: int) -> int:
    return CAN_ID_QWIIC_IMU_BASE + leg_index


def qwiic_imu_can_id_for_leg(leg_id: int) -> int:
    return qwiic_imu_can_id(leg_id_to_index(leg_id))


def qwiic_battery_can_id(leg_index: int) -> int:
    return CAN_ID_QWIIC_BATTERY_BASE + leg_index


def qwiic_battery_can_id_for_leg(leg_id: int) -> int:
    return qwiic_battery_can_id(leg_id_to_index(leg_id))


def leg_bus(leg_id: int) -> str:
    return LEG_BUS_BY_ID[leg_id]
