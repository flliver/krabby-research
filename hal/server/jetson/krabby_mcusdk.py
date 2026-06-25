"""SDK interface for applying joint commands to the Krabby MCU.

This module provides a standardized SDK interface for applying joint commands
to the Krabby MCU (real hardware on Jetson).
"""

import logging
from typing import Optional

from hal.client.data_structures.hardware import JointCommand
from firmware.krabby_mcu import KrabbyMCUSDK as FirmwareKrabbyMCUSDK

logger = logging.getLogger(__name__)

JOINT_LIMIT_RAD = 0.5
JOINT_NEUTRAL = 0.5
NEUTRAL_RAD_THRESHOLD = 0.01
PWM_SCALE = 255 / JOINT_LIMIT_RAD
# JointTelemetry.cal_state: 0=UNCAL, 1=PARTIAL, 2=FULL. Only FULL is safe to drive
# closed-loop — UNCAL/PARTIAL joints don't have a trustworthy absolute position yet.
FULL_CAL_STATE = 2

_HAL_TO_FW_SUFFIX = {"hip_yaw": "HY", "hip_pitch": "HL", "knee": "KL"}


def _hal_to_firmware_name(hal_name: str) -> str:
    leg, _, suffix = hal_name.partition("_")
    return leg + _HAL_TO_FW_SUFFIX.get(suffix, suffix[:2].upper() if len(suffix) >= 2 else "??")


def _rad_to_pwm(rad: float) -> int:
    return max(-255, min(255, int(round(rad * PWM_SCALE))))


def _map_mcu_joints_to_normalized(command: dict[str, float], mcu_joints: tuple[str, ...]) -> dict[str, float]:
    out = {}
    for name in mcu_joints:
        r = command.get(name, 0.0)
        n = max(0.0, min(1.0, (r / JOINT_LIMIT_RAD) * 0.5 + JOINT_NEUTRAL))
        out[_hal_to_firmware_name(name)] = n
    return out


class KrabbyMCUSDK:
    """SDK for applying 18-joint HAL commands to the MCU.

    Accepts JointCommand; joint order is given by joint_names (e.g. robot_definition.get_joint_names()).
    mcu_joints comes from robot_definition.get_mcu_joints().
    """

    def __init__(
        self,
        mcu_joints: tuple[str, ...],
        port: Optional[str] = None,
        baud: int = 115200,
        auto_connect: bool = True,
    ):
        """Initialize MCU SDK. mcu_joints: joint names for the MCU (e.g. from robot_definition.get_mcu_joints())."""
        if FirmwareKrabbyMCUSDK is None:
            raise RuntimeError(
                "KrabbyMCUSDK firmware module not available. "
                "Cannot initialize MCU SDK without firmware package."
            )
        if len(mcu_joints) != 18:
            raise ValueError(
                f"mcu_joints must have 18 names (firmware expects 18 joints), got {len(mcu_joints)}"
            )
        
        self._mcu_joints = mcu_joints
        self._mcu = FirmwareKrabbyMCUSDK(port=port, baud=baud)
        self._connected = False
        
        logger.info(f"KrabbyMCUSDK initialized (port={port}, baud={baud}, auto_connect={auto_connect})")
        
        if auto_connect:
            self.connect()
    
    def connect(self) -> bool:
        """Connect to MCU. Returns True if successful."""
        if self._connected:
            logger.warning("MCU already connected")
            return True
        
        success = self._mcu.connect()
        if success:
            self._connected = True
            logger.info("MCU connected successfully")
        else:
            logger.error("Failed to connect to MCU")
            raise RuntimeError("Failed to connect to MCU")
        
        return success
    
    def is_connected(self) -> bool:
        """Return whether MCU is connected."""
        return self._connected and self._mcu.running
    
    def _route_by_cal_state(
        self, cmd_dict: dict[str, float], cmds_by_fw: dict[str, float]
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Split joints into closed-loop position targets vs open-loop jogs.

        A joint gets a normalized [0,1] position target (firmware ``T`` command,
        servoed against its own sensor) only once the firmware reports it FULLY
        calibrated. Until then its sensor reading isn't trustworthy, so a position
        servo would chase bad feedback — those joints stay on open-loop jog
        (neutral → 0, else PWM). Jogging a PARTIAL joint is also what drives it into
        an end-stop to self-heal to FULL, after which it graduates to position
        control automatically, no code change. See Task 2 §6.

        A joint with no telemetry yet (None) is treated as not-FULL → jog.
        """
        targets: dict[str, float] = {}
        jogs: dict[str, int] = {}
        for name in self._mcu_joints:
            fw = _hal_to_firmware_name(name)
            jt = self._mcu.joints.get(fw)
            if jt is not None and jt.cal_state == FULL_CAL_STATE:
                targets[fw] = cmds_by_fw[fw]
            else:
                rad = cmd_dict.get(name, 0.0)
                jogs[fw] = 0 if abs(rad) <= NEUTRAL_RAD_THRESHOLD else _rad_to_pwm(rad)
        return targets, jogs

    def apply_command(self, command: JointCommand) -> None:
        """Apply joint command to MCU (radians -> normalized, then send).

        Args:
            command: JointCommand with joint_positions, joint_names, and timestamps (order fixed on command).
        """
        if not self.is_connected():
            raise RuntimeError("MCU is not connected. Call connect() first.")

        cmd_dict = command.to_positions_dict()
        for name in self._mcu_joints:
            if name not in cmd_dict:
                raise ValueError(
                    f"command must contain key {name!r} (and all names in mcu_joints)"
                )

        cmds_by_fw = _map_mcu_joints_to_normalized(cmd_dict, self._mcu_joints)
        # Hybrid routing: closed-loop position targets for FULLY-calibrated joints,
        # open-loop jog for the rest (see _route_by_cal_state). As joints get
        # calibrated they migrate from jog to position control on their own — this
        # is what lets the HAL run before all 18 joints are calibrated/assembled.
        targets, jogs = self._route_by_cal_state(cmd_dict, cmds_by_fw)
        if targets:
            self._mcu.send_command_joints(targets)
        if jogs:
            self._mcu.send_commands_jog(jogs)

        joint_values_str = ", ".join(
            f"{name}={val:.4f}" for name, val in cmds_by_fw.items()
        )
        logger.info(
            f"KrabbyMCUSDK: Applied joint command (timestamp_ns={command.timestamp_ns}, observation_timestamp_ns={command.observation_timestamp_ns}) "
            f"[{len(targets)} pos / {len(jogs)} jog]: {joint_values_str}"
        )
        rad_vals = list(cmd_dict.values())
        if rad_vals:
            logger.debug(
                f"KrabbyMCUSDK: Joint command stats - "
                f"radians: min={min(rad_vals):.4f}, max={max(rad_vals):.4f}, "
                f"MCU normalized: min={min(cmds_by_fw.values()):.4f}, max={max(cmds_by_fw.values()):.4f}"
            )
    
    def close(self) -> None:
        """Close MCU connection and release resources."""
        self._mcu.close()
        self._connected = False
        logger.info("MCU connection closed")
