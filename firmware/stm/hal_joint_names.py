"""HAL and policy joint name strings (host edge only; wire uses leg_id / joint_id)."""

from __future__ import annotations

from typing import Final

# Corner tags for RobotDefinition.mc_joints only (leg_id 1..6 order).
_LEG_CORNER: Final = ("FL", "FR", "ML", "MR", "RL", "RR")
_JOINT_HAL_SUFFIX: Final = ("hip_yaw", "hip_pitch", "knee")

LEG_CODES: Final = _LEG_CORNER  # deprecated alias for GUI/tests importing legacy shape

MCU_JOINTS: Final = tuple(
    f"{leg}_{joint}" for leg in _LEG_CORNER for joint in _JOINT_HAL_SUFFIX
)


def mcu_joint_name(leg_id: int, joint_id: int) -> str:
    """Map numeric identity to HAL ``mcu_joints`` name for policy / GUI adapters."""
    if not 1 <= leg_id <= 6 or not 1 <= joint_id <= 3:
        raise ValueError(f"leg_id/joint_id out of range: {leg_id!r}, {joint_id!r}")
    return f"{_LEG_CORNER[leg_id - 1]}_{_JOINT_HAL_SUFFIX[joint_id - 1]}"


def parse_mcu_joint_name(name: str) -> tuple[int, int]:
    """Return ``(leg_id, joint_id)`` from a HAL ``mcu_joints`` name."""
    for leg_i, leg in enumerate(_LEG_CORNER):
        prefix = f"{leg}_"
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        if suffix in _JOINT_HAL_SUFFIX:
            return leg_i + 1, _JOINT_HAL_SUFFIX.index(suffix) + 1
    raise ValueError(f"unknown mcu joint name: {name!r}")


def joint_to_leg_bus(leg_id: int, joint_id: int) -> tuple[str, int, int]:
    """``(socketcan_iface, leg_index, joint_index)`` for transport."""
    from firmware.stm import ids

    if joint_id not in ids.JOINT_IDS:
        raise ValueError(f"joint_id out of range: {joint_id!r}")
    return ids.LEG_BUS_BY_ID[leg_id], ids.leg_id_to_index(leg_id), joint_id - 1
