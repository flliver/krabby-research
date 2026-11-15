"""Isaac fullsim command path over STM CAN (vcan0 + KrabbyStmMCUSDK)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.firmware_command_mapping import hal_joint_to_firmware_id
from hal.server.isaac.firmware_serial_loop import FullsimFirmwareSession
from hal.server.robot_definition import RobotDefinition

logger = logging.getLogger(__name__)

_DEFAULT_CAN_IFACE = "vcan0"


class CanStmCommandLoop:
    """Drive ``CanRobotCore`` from HAL ``KrabbyStmMCUSDK`` SET_DUTY traffic on SocketCAN."""

    def __init__(self, can_core: Any, stm_hal_sdk: Any, mcu_joints: tuple[str, ...]) -> None:
        from firmware.sim.can_robot_core import leg_motor_from_actuator_id

        self._core = can_core
        self._stm_sdk = stm_hal_sdk
        self._mcu_joints = mcu_joints
        self._leg_motor_from_actuator_id = leg_motor_from_actuator_id

    def apply_hal_command(self, command: JointCommand) -> None:
        self._stm_sdk.apply_command(command)

    def jog_actuator(self, firmware_actuator_id: str, pwm: int) -> None:
        del firmware_actuator_id, pwm
        raise RuntimeError("can transport uses apply_hal_command(), not per-joint jog")

    def tick_firmware(self) -> None:
        t0 = time.perf_counter()
        self._core.tick_without_physics()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms >= 250.0:
            logger.warning("CanStmCommandLoop: tick_without_physics took %.1f ms", elapsed_ms)
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug("CanStmCommandLoop: tick_without_physics %.1f ms", elapsed_ms)

    def signed_command_pwm(self, firmware_actuator_id: str) -> int:
        duty = self.duty_for_firmware_id(firmware_actuator_id)
        return max(-255, min(255, int(round(duty * 255))))

    def duty_for_hal_joint(self, hal_name: str) -> float:
        fw_id = hal_joint_to_firmware_id(hal_name)
        return self.duty_for_firmware_id(fw_id)

    def duty_for_firmware_id(self, firmware_actuator_id: str) -> float:
        leg_id, motor_index = self._leg_motor_from_actuator_id(firmware_actuator_id)
        leg = self._core.legs[leg_id - 1]
        return float(self._core.bridge.get_duty(leg, motor_index))


def start_can_stm_session(
    env: Any,
    robot: Any,
    robot_definition: RobotDefinition,
    *,
    can_iface: str | None = None,
) -> FullsimFirmwareSession:
    """Attach six virtual STM leg boards on CAN and return a fullsim session."""
    from firmware.sim.can_board_bridge import StmHostLibraryNotBuiltError, load_stm_board_bridge
    from firmware.sim.can_robot_core import CanRobotCore
    from firmware.sim.isaac_scene import configure_open_loop_velocity_drive, resolve_eighteen_joint_indices
    from firmware.sim.simulated_robot_config import ThreeBoardSimConfigs
    from firmware.stm.can_transport import open_socketcan_fd
    from hal.server.jetson.krabby_stm_mcusdk import KrabbyStmMCUSDK

    iface = can_iface or os.environ.get("KRABBY_CAN_IFACE", _DEFAULT_CAN_IFACE)
    try:
        load_stm_board_bridge()
    except StmHostLibraryNotBuiltError:
        raise

    mcu_joints = robot_definition.get_mcu_joints()
    if len(mcu_joints) != 18:
        raise ValueError(f"can fullsim requires 18 mcu_joints, got {len(mcu_joints)}")

    configs = ThreeBoardSimConfigs.rev3()
    joint_indices = resolve_eighteen_joint_indices(robot, configs)
    configure_open_loop_velocity_drive(robot, list(joint_indices.values()))

    bus = open_socketcan_fd(iface)
    core = CanRobotCore.from_isaac(
        env,
        robot,
        joint_indices,
        bus=bus,
    )
    core.reset()

    stm_sdk = KrabbyStmMCUSDK(mcu_joints=mcu_joints, virtual=True, auto_connect=True)
    loop = CanStmCommandLoop(core, stm_sdk, mcu_joints)
    logger.info("Fullsim firmware transport: can (iface=%s, KrabbyStmMCUSDK virtual)", iface)
    return FullsimFirmwareSession(
        core=core,
        firmware_loop=loop,
        firmware_transport="can",
        can_bus=bus,
        stm_hal_sdk=stm_sdk,
    )