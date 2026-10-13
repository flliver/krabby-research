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

    def __init__(self, hal_mcu: Any, three_board_core: Any) -> None:
        self._hal_mcu = hal_mcu
        self._core = three_board_core

    def apply_hal_command(self, command: JointCommand) -> None:
        """Jetson-equivalent jog batch over pyserial, then one firmware tick on Isaac."""
        self._hal_mcu.apply_command(command)
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
                logger.exception("Error closing ThreeBoardMcuHost")
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

    from firmware.sim.three_board_mcu_host import ThreeBoardMcuHost
    from hal.server.jetson.krabby_mcusdk import KrabbyMCUSDK as HalKrabbyMCUSDK

    mcu_joints = robot_definition.get_mcu_joints()
    if len(mcu_joints) != 18:
        raise ValueError(f"serial_sdk requires 18 mcu_joints, got {len(mcu_joints)}")

    host = ThreeBoardMcuHost(core.bridge)
    port = host.start()
    core.configure_sensors_and_stops()

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
        time.sleep(0.02)
    else:
        host.close()
        hal_mcu.close()
        raise RuntimeError(
            "KrabbyMCUSDK did not receive telemetry over PTY within "
            f"{connect_timeout_s}s (port={port!r})"
        )

    logger.info(
        "Fullsim serial SDK on PTY %s (KrabbyMCUSDK wire protocol, same as Jetson teleop)",
        port,
    )
    loop = SerialSdkCommandLoop(hal_mcu, core)
    session = FullsimFirmwareSession(
        core=core,
        firmware_loop=loop,
        mcu_host=host,
        hal_krabby_sdk=hal_mcu,
        pty_slave_path=port,
        firmware_transport="serial_sdk",
    )
    return loop, session
