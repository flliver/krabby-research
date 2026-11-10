"""Leg index, HAL joint names, and CAN ID helpers (PROVISIONING.md, CAN_PROTOCOL.md)."""

from __future__ import annotations

from typing import Final

LEG_CODES: Final = ("FL", "FR", "ML", "MR", "RL", "RR")
JOINT_SUFFIXES: Final = ("hip_yaw", "hip_pitch", "knee")

LEG_CODE_UNASSIGNED: Final = 255
LEG_INDEX_UNASSIGNED: Final = 6
LEG_LABEL_UNASSIGNED: Final = "UNK"

LEG_CODE_BY_NAME: Final = {name: index for index, name in enumerate(LEG_CODES)}
LEG_NAME_BY_INDEX: Final = {index: name for name, index in LEG_CODE_BY_NAME.items()}

LEG_BUS: Final = {
    "FL": "can0",
    "FR": "can1",
    "ML": "can0",
    "MR": "can1",
    "RL": "can0",
    "RR": "can1",
}

CAN_ID_LEG_COMMAND_BASE: Final = 0x0C000000
CAN_ID_LEG_TELEMETRY_BASE: Final = 0x0C000100
CAN_ID_NODE_STATUS_BASE: Final = 0x0C000200
CAN_ID_DISCOVERY_REQUEST: Final = 0x0C0007F0
CAN_ID_PROVISION_SET: Final = 0x0C0007E0

MCU_JOINTS: Final = tuple(
    f"{leg}_{joint}" for leg in LEG_CODES for joint in JOINT_SUFFIXES
)


def leg_command_can_id(leg_index: int) -> int:
    return CAN_ID_LEG_COMMAND_BASE + leg_index


def leg_telemetry_can_id(leg_index: int) -> int:
    return CAN_ID_LEG_TELEMETRY_BASE + leg_index


def node_status_can_id(leg_index: int) -> int:
    return CAN_ID_NODE_STATUS_BASE + leg_index


def parse_mcu_joint_name(name: str) -> tuple[str, int]:
    """Return ``(leg_code, joint_index)`` for a HAL ``mcu_joints`` name."""
    for leg in LEG_CODES:
        prefix = f"{leg}_"
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix in JOINT_SUFFIXES:
            return leg, JOINT_SUFFIXES.index(suffix)
    raise ValueError(f"unknown mcu joint name: {name!r}")


def joint_to_leg_bus_index(name: str) -> tuple[str, int, int]:
    """Map HAL name to ``(socketcan_iface, leg_index, joint_index)``."""
    leg, joint_index = parse_mcu_joint_name(name)
    return LEG_BUS[leg], LEG_CODE_BY_NAME[leg], joint_index
