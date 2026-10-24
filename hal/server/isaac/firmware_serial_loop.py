"""Production KrabbyMCUSDK (pyserial) path for Isaac fullsim HAL.

Default fullsim transport remains in-process ``ThreeBoardCommandLoop`` (direct
``serial_usb_rx_write`` / host jog). Set ``firmware_transport='serial_sdk'`` on
``IsaacSimHalServer`` to drive the same ``B``/``J`` wire protocol as Jetson
teleop via a PTY-backed ``ThreeBoardMcuHost``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Literal

from hal.client.data_structures.hardware import JointCommand
from hal.server.isaac.firmware_command_mapping import joint_radians_to_jog_pwm
from hal.server.robot_definition import RobotDefinition

logger = logging.getLogger(__name__)

FirmwareTransport = Literal["in_process", "serial_sdk"]

__all__ = [
    "FirmwareTransport",
    "FullsimFirmwareSession",
    "SerialSdkCommandLoop",
    "start_serial_sdk_session",
]


class SerialSdkCommandLoop:
    """``FirmwareCommandLoop`` that sends ``JointCommand`` through HAL ``KrabbyMCUSDK``."""

    def __init__(
        self,
        hal_mcu: Any,
        three_board_core: Any,
        *,
        mcu_joints: tuple[str, ...] = (),
    ) -> None:
        self._hal_mcu = hal_mcu
        self._core = three_board_core
        self._mcu_joints = mcu_joints

    def apply_hal_command(self, command: JointCommand) -> None:
        """Jetson-equivalent jog batch over pyserial, then firmware tick(s) on Isaac."""
        cmd_dict = command.to_positions_dict()
        for name in self._mcu_joints:
            if name not in cmd_dict:
                raise ValueError(
                    f"command must contain key {name!r} (and all names in mcu_joints)"
                )
        jog_map = joint_radians_to_jog_pwm(cmd_dict, self._mcu_joints)
        active = {fw_id: pwm for fw_id, pwm in jog_map.items() if pwm != 0}
        firmware_mcu = self._hal_mcu._mcu  # noqa: SLF001 — pyserial wire to PTY
        if not active:
            firmware_mcu.send_command_joints_hold()
        else:
            firmware_mcu.send_commands_jog(active)

        active_fw = list(active.keys())

        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            self._core.tick_firmware_without_physics()
            if not active_fw:
                return
            if any(
                self._core.command_pwm_for_actuator(fw_id) != 0 for fw_id in active_fw
            ):
                return
            time.sleep(0.005)
        self._core.tick_firmware_without_physics()

    def jog_actuator(self, firmware_actuator_id: str, pwm: int) -> None:
        del firmware_actuator_id, pwm
        raise RuntimeError("serial_sdk transport uses apply_hal_command(), not per-joint jog")

    def tick_firmware(self) -> None:
        return

    def signed_command_pwm(self, firmware_actuator_id: str) -> int:
        return self._core.command_pwm_for_actuator(firmware_actuator_id)


@dataclass
class FullsimFirmwareSession:
    """Three-board core plus command loop; optional PTY/SDK handles for serial transport."""

    core: Any
    firmware_loop: Any
    mcu_host: Any | None = None
    hal_krabby_sdk: Any | None = None
    pty_slave_path: str = ""
    firmware_transport: FirmwareTransport = "in_process"

    def close(self) -> None:
        if self.hal_krabby_sdk is not None:
            try:
                self.hal_krabby_sdk.close()
            except Exception:  # noqa: BLE001 — best-effort shutdown
                logger.exception("Error closing HAL KrabbyMCUSDK")
            self.hal_krabby_sdk = None
        if self.mcu_host is not None:
            try:
                self.mcu_host.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error closing serial_sdk PTY host")
            self.mcu_host = None


def start_serial_sdk_session(
    core: Any,
    robot_definition: RobotDefinition,
    *,
    connect_timeout_s: float = 10.0,
) -> tuple[SerialSdkCommandLoop, FullsimFirmwareSession]:
    """Start PTY host on ``core.bridge`` and connect production ``KrabbyMCUSDK``."""
    import os

    if not hasattr(os, "openpty"):
        raise RuntimeError("serial_sdk fullsim transport requires Unix PTY support")

    from firmware.sim.serial_pty_bridge import SimMcuPtyBridge
    from hal.server.jetson.krabby_mcusdk import KrabbyMCUSDK as HalKrabbyMCUSDK

    mcu_joints = robot_definition.get_mcu_joints()
    if len(mcu_joints) != 18:
        raise ValueError(f"serial_sdk requires 18 mcu_joints, got {len(mcu_joints)}")

    # PTY relay only — ``ThreeBoardMcuHost``'s background firmware loop double-ticks
    # against ``ThreeBoardRobotCore.step()`` / ``tick_firmware_without_physics()``.
    core.reset()
    pty = SimMcuPtyBridge(core.bridge)
    port = pty.start()

    hal_mcu = HalKrabbyMCUSDK(tuple(mcu_joints), port=port, auto_connect=True)
    deadline = time.monotonic() + connect_timeout_s
    while time.monotonic() < deadline:
        firmware_sdk = hal_mcu._mcu  # noqa: SLF001 — wait for telemetry reader
        if (
            hal_mcu.is_connected()
            and firmware_sdk.role_last_seen
            and any(firmware_sdk.joints.get(name) is not None for name in ("FLHY", "RLHY", "RRHY"))
        ):
            break
        core.tick_firmware_without_physics()
        time.sleep(0.002)
    else:
        pty.close()
        hal_mcu.close()
        raise RuntimeError(
            "KrabbyMCUSDK did not receive telemetry over PTY within "
            f"{connect_timeout_s}s (port={port!r})"
        )

    logger.info(
        "Fullsim serial SDK on PTY %s (KrabbyMCUSDK wire protocol, same as Jetson teleop)",
        port,
    )
    loop = SerialSdkCommandLoop(hal_mcu, core, mcu_joints=tuple(mcu_joints))
    session = FullsimFirmwareSession(
        core=core,
        firmware_loop=loop,
        mcu_host=pty,
        hal_krabby_sdk=hal_mcu,
        pty_slave_path=port,
        firmware_transport="serial_sdk",
    )
    return loop, session
