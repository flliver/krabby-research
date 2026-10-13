"""Attach Task 3 three-board firmware to the IsaacSim env the HAL server already owns."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from hal.server.isaac.firmware_command_mapping import hal_joint_to_firmware_id
from hal.server.robot_definition import RobotDefinition

logger = logging.getLogger(__name__)

__all__ = [
    "attach_three_board_firmware_to_env",
    "log_firmware_joint_bindings",
]


def log_firmware_joint_bindings(
    robot_definition: RobotDefinition,
    joint_indices: Mapping[str, int],
    *,
    firmware_actuator_id_for_hal_joint: Mapping[str, str],
    isaac_joint_for_firmware_id: Mapping[str, str],
) -> None:
    """Log HAL joint → firmware id → Isaac DOF index (one process, existing articulation)."""
    mcu_joints = robot_definition.get_mcu_joints()
    logger.info(
        "Three-board firmware: binding %d MCU joints on existing Isaac env (one process)",
        len(mcu_joints),
    )
    for hal_name in mcu_joints:
        fw_id = firmware_actuator_id_for_hal_joint.get(hal_name) or hal_joint_to_firmware_id(
            hal_name
        )
        isaac_joint = isaac_joint_for_firmware_id[fw_id]
        dof_index = joint_indices[isaac_joint]
        logger.info(
            "Firmware joint binding: hal=%s firmware=%s isaac=%s dof_index=%d",
            hal_name,
            fw_id,
            isaac_joint,
            dof_index,
        )


def attach_three_board_firmware_to_env(
    env: Any,
    robot: Any,
    robot_definition: RobotDefinition,
    *,
    max_joint_speed: float = 2.0,
    bridge: Any | None = None,
) -> tuple[Any, Any]:
    """Construct ``ThreeBoardRobotCore`` on ``env`` / ``robot`` (no second scene).

    Raises:
        ValueError: Robot definition has no 18-joint MCU map.
        HostLibraryNotBuiltError: Host firmware library was not built.
    """
    from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
    from firmware.sim.isaac_scene import (
        configure_open_loop_velocity_drive,
        resolve_eighteen_joint_indices,
    )
    from firmware.sim.simulated_robot_config import ThreeBoardSimConfigs
    from firmware.sim.three_board_coordinator import ThreeBoardRobotCore
    from hal.server.isaac.firmware_loop_bindings import ThreeBoardCommandLoop

    mcu_joints = robot_definition.get_mcu_joints()
    if len(mcu_joints) != 18:
        raise ValueError(
            f"fullsim requires 18 mcu_joints on robot definition, got {len(mcu_joints)}"
        )

    try:
        host_bridge = bridge if bridge is not None else load_host_bridge()
    except HostLibraryNotBuiltError:
        raise

    configs = ThreeBoardSimConfigs.rev3()
    joint_indices = resolve_eighteen_joint_indices(robot, configs)
    configure_open_loop_velocity_drive(robot, joint_indices.values())

    core = ThreeBoardRobotCore.eighteen_joints(
        env=env,
        articulation=robot,
        joint_index_for_name=joint_indices,
        bridge=host_bridge,
        configs=configs,
        max_joint_speed=max_joint_speed,
    )
    core.reset()

    isaac_joint_for_fw = {
        slot.firmware_actuator_id: slot.isaac_joint_name for slot in configs.all_slots()
    }
    log_firmware_joint_bindings(
        robot_definition,
        joint_indices,
        firmware_actuator_id_for_hal_joint={
            name: hal_joint_to_firmware_id(name) for name in mcu_joints
        },
        isaac_joint_for_firmware_id=isaac_joint_for_fw,
    )
    return core, ThreeBoardCommandLoop(core)
