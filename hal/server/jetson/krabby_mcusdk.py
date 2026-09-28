"""SDK interface for applying joint commands to the Krabby MCU.

This module provides a standardized SDK interface for applying joint commands
to the Krabby MCU (real hardware on Jetson).
"""

import logging
import time
from typing import Optional

from hal.client.data_structures.hardware import JointCommand
from firmware.krabby_mcu import DEFAULT_BAUD, KrabbyMCUSDK as FirmwareKrabbyMCUSDK

logger = logging.getLogger(__name__)

JOINT_LIMIT_RAD = 0.5
JOINT_NEUTRAL = 0.5
NEUTRAL_RAD_THRESHOLD = 0.01
PWM_SCALE = 255 / JOINT_LIMIT_RAD

_HAL_TO_FW_SUFFIX = {"hip_yaw": "HY", "hip_pitch": "HL", "knee": "KL"}

# Current sense -> contact_forces (M17 Task 6 §3, Option A). FIRST-PASS MAPPING,
# expected to be refined in M15. The model's contact_forces is 5-wide but the hex
# has 6 legs: five legs fill the five slots and MR is dropped (the middle pair is
# redundant for a forward gait). A leg's load proxy is the sum of its joints' raw
# current (avgIS counts), mapped linearly so 0 -> -0.5 (no contact) and
# CONTACT_FULLSCALE -> +0.5, clipped. CONTACT_FULLSCALE is a placeholder until
# Task 4's loaded/unloaded current ranges are measured on the chassis.
CONTACT_LEGS = ("FL", "FR", "ML", "RL", "RR")
CONTACT_FULLSCALE = 300.0


def contact_forces_from_joints(joints) -> list[float]:
    """joints: firmware joint name -> JointTelemetry. A leg with no telemetry reads 0.0 (unknown)."""
    forces = []
    for leg in CONTACT_LEGS:
        currents = [jt.current for name, jt in joints.items() if jt is not None and name.startswith(leg)]
        if not currents:
            forces.append(0.0)
            continue
        forces.append(max(-0.5, min(0.5, sum(currents) / CONTACT_FULLSCALE - 0.5)))
    return forces


def _hal_to_firmware_name(hal_name: str) -> str:
    leg, _, suffix = hal_name.partition("_")
    return leg + _HAL_TO_FW_SUFFIX.get(suffix, suffix[:2].upper() if len(suffix) >= 2 else "??")


def _rad_to_pwm(rad: float) -> int:
    return max(-255, min(255, int(round(rad * PWM_SCALE))))


def _normalized_to_rad(n: float) -> float:
    """Inverse of the command mapping in _map_mcu_joints_to_normalized, so measured
    positions reach the model in the same radian scale as its commands."""
    return (n - JOINT_NEUTRAL) * 2.0 * JOINT_LIMIT_RAD


# Joint velocity is differentiated from successive measured positions (the MCU
# reports position only), smoothed with a single-pole EMA to suppress serial jitter.
JOINT_VEL_EMA_ALPHA = 0.2


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
        baud: int = DEFAULT_BAUD,
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
        self._last_pos: dict[str, tuple[float, float]] = {}  # HAL name -> (rad, t)
        self._vel: dict[str, float] = {}
        
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
    
    def _jog_all_joints(self, cmd_dict: dict[str, float]) -> None:
        """Jog all joints (neutral → 0, non-neutral → PWM)."""
        jog = {}
        for name in self._mcu_joints:
            rad = cmd_dict.get(name, 0.0)
            fw = _hal_to_firmware_name(name)

            jog[fw] = 0 if abs(rad) <= NEUTRAL_RAD_THRESHOLD else _rad_to_pwm(rad)
        self._mcu.send_commands_jog(jog)

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
        ### TODO: As a pot is not connected for all joints, we do not call self._mcu.send_command_joints(cmds_by_fw) for now.    
        ### Instead, we call self._jog_all_joints(cmd_dict) to jog all joints (neutral → 0, non-neutral → PWM).    
        self._jog_all_joints(cmd_dict)

        joint_values_str = ", ".join(
            f"{name}={val:.4f}" for name, val in cmds_by_fw.items()
        )
        logger.info(
            f"KrabbyMCUSDK: Applied joint command (timestamp_ns={command.timestamp_ns}, observation_timestamp_ns={command.observation_timestamp_ns}): "
            f"{joint_values_str}"
        )
        rad_vals = list(cmd_dict.values())
        if rad_vals:
            logger.debug(
                f"KrabbyMCUSDK: Joint command stats - "
                f"radians: min={min(rad_vals):.4f}, max={max(rad_vals):.4f}, "
                f"MCU normalized: min={min(cmds_by_fw.values()):.4f}, max={max(cmds_by_fw.values()):.4f}"
            )
    
    def joint_state(self, now: Optional[float] = None) -> tuple[dict[str, float], dict[str, float]]:
        """Measured (positions, velocities) in rad and rad/s, keyed by HAL joint name.

        Only joints with live, connected telemetry are included; the caller keeps
        its fallback for the rest. Velocity is 0.0 on a joint's first sample.
        """
        now = time.monotonic() if now is None else now
        joints = dict(self._mcu.joints)
        positions, velocities = {}, {}
        for name in self._mcu_joints:
            jt = joints.get(_hal_to_firmware_name(name))
            if jt is None or not jt.connected:
                continue
            rad = _normalized_to_rad(jt.pos)
            prev = self._last_pos.get(name)
            if prev is not None and now > prev[1]:
                raw = (rad - prev[0]) / (now - prev[1])
                self._vel[name] = self._vel.get(name, 0.0) + JOINT_VEL_EMA_ALPHA * (raw - self._vel.get(name, 0.0))
            self._last_pos[name] = (rad, now)
            positions[name] = rad
            velocities[name] = self._vel.get(name, 0.0)
        return positions, velocities

    def contact_forces(self) -> list[float]:
        """5-slot contact_forces from the latest per-joint current telemetry."""
        # Copy: the firmware SDK's reader thread updates .joints concurrently.
        return contact_forces_from_joints(dict(self._mcu.joints))

    def close(self) -> None:
        """Close MCU connection and release resources."""
        self._mcu.close()
        self._connected = False
        logger.info("MCU connection closed")
