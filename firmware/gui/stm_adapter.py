"""Mega-shaped GUI facade over ``KrabbyMCUCANSDK`` (TASK-2 GUI adapter)."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from firmware.interfaces.joint_telemetry import ActuatorConnection, JointTelemetry
from firmware.krabby_mcu import JOINT_GROUP_NAMES
from firmware.stm import ids
from firmware.stm.mcu_can import KrabbyMCUCANSDK

logger = logging.getLogger("KrabbyStmGuiAdapter")

_PWM_SCALE = 255.0


class _StmCanTarget:
    def __init__(self, *, virtual: bool) -> None:
        self._virtual = virtual

    def __str__(self) -> str:
        return "stm:vcan0" if self._virtual else "stm:can0+can1"


def _gui_name_to_leg_joint() -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    index = 0
    for _, names in JOINT_GROUP_NAMES:
        for name in names:
            leg_id = index // 3 + 1
            joint_id = index % 3 + 1
            out[name] = (leg_id, joint_id)
            index += 1
    return out


_GUI_JOINT_MAP: dict[str, tuple[int, int]] = _gui_name_to_leg_joint()


def _duty_from_pwm(pwm: int) -> float:
    clamped = max(-255, min(255, int(pwm)))
    return clamped / _PWM_SCALE


def _pwm_from_duty(duty: float) -> tuple[int, int]:
    scaled = int(round(float(duty) * _PWM_SCALE))
    if scaled >= 0:
        return scaled, 0
    return 0, -scaled


def _format_ver_line(board_info: dict[int, dict[str, Any]]) -> str:
    """Build a Mega-style ``VER`` line from six leg ``BOARD_INFO`` payloads."""
    versions: list[str] = []
    branches: list[str] = []
    commits: list[str] = []
    for leg_id in ids.LEG_IDS:
        info = board_info.get(leg_id) or {}
        if info:
            versions.append(
                f"{int(info.get('major', 0))}.{int(info.get('minor', 0))}.{int(info.get('patch', 0))}"
            )
            branches.append("-")
            commits.append(str(int(info.get("commit", 0))))
        else:
            versions.append("-")
            branches.append("-")
            commits.append("-")
    return (
        "VER "
        + "|".join(versions)
        + " "
        + "|".join(branches)
        + " "
        + "|".join(commits)
    )


def _board_info_ready(board_info: dict[int, dict[str, Any]]) -> bool:
    for leg_id in ids.LEG_IDS:
        info = board_info.get(leg_id) or {}
        if "major" in info or "commit" in info:
            return True
    return False


def _joint_telemetry_from_leg(
    name: str, leg_fields: dict[str, Any], joint_id: int
) -> Optional[JointTelemetry]:
    prefix = f"J{joint_id}_"
    pot_key = f"{prefix}POT"
    if pot_key not in leg_fields and f"{prefix}HALL" not in leg_fields:
        return None
    duty = float(leg_fields.get(f"{prefix}DUTY", 0.0))
    return JointTelemetry(
        name=name,
        pos=float(leg_fields.get(f"{prefix}POSITION", 0.0)),
        pot=int(leg_fields.get(pot_key, 0)),
        current=0,
        en=(1, 1),
        pwm=_pwm_from_duty(duty),
        saf=int(leg_fields.get(f"{prefix}HALL", 0)),
        connection_state=ActuatorConnection.CONNECTED,
    )


class KrabbyStmGuiAdapter:
    """Thin adapter so ``firmware.gui`` can use the CAN SDK like ``KrabbyMCUSDK``."""

    def __init__(
        self,
        *,
        virtual: bool = False,
        sdk: KrabbyMCUCANSDK | None = None,
    ) -> None:
        self.target = _StmCanTarget(virtual=virtual)
        self._sdk = sdk if sdk is not None else KrabbyMCUCANSDK(virtual=virtual)
        self._joints: Dict[str, Optional[JointTelemetry]] = {}
        self.imu = None
        self.battery = None
        self.last_feedback_ts: Optional[float] = None
        self.last_error: Optional[BaseException] = None
        self.last_cmd: Dict[str, Optional[float]] = {}
        self.role_last_seen: Dict[str, float] = {}

    @property
    def joints(self) -> Dict[str, Optional[JointTelemetry]]:
        self.refresh_joints_from_telemetry()
        return self._joints

    def connect(self) -> bool:
        try:
            self._sdk.connect()
            self.last_error = None
            self.send_command_joints_hold()
            logger.info("Connected to %s", self.target)
            return True
        except Exception as exc:
            logger.exception("Connection Failed")
            self.last_error = exc
            return False

    def close(self) -> None:
        self._sdk.close()

    def read_version(self, timeout: float = 1.0) -> Optional[str]:
        """Wait for ``BOARD_INFO`` and return a Mega-shaped ``VER`` line."""
        if not self._sdk._connected:  # noqa: SLF001
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            board_info = self._sdk.board_info
            if _board_info_ready(board_info):
                return _format_ver_line(board_info)
            time.sleep(0.02)
        return None

    def _leg_duties(self, leg_id: int) -> list[float]:
        with self._sdk._lock:  # noqa: SLF001 — read cached duties for single-joint jog
            row = self._sdk._duty[leg_id - 1]  # noqa: SLF001
            return [float(v) for v in row]

    def send_command_jog(self, joint_name: str, pwm: int) -> None:
        if joint_name not in _GUI_JOINT_MAP:
            raise ValueError(f"unknown joint name: {joint_name!r}")
        leg_id, joint_id = _GUI_JOINT_MAP[joint_name]
        duties = self._leg_duties(leg_id)
        duties[joint_id - 1] = _duty_from_pwm(pwm)
        self._sdk.set_duty(leg_id, duties[0], duties[1], duties[2])
        self.last_cmd[joint_name] = _duty_from_pwm(pwm)
        logger.debug(
            "CMD -> jog %s pwm=%s duty=%s", joint_name, pwm, self.last_cmd[joint_name]
        )

    def send_command_joints_hold(self) -> None:
        self._sdk.set_duties([0.0] * 18)
        for name in _GUI_JOINT_MAP:
            self.last_cmd[name] = 0.0
        logger.info("CMD -> hold (all duties zero)")

    def send_calibration_for_joint(self, joint_name: str) -> None:
        if joint_name not in _GUI_JOINT_MAP:
            raise ValueError(f"unknown joint name: {joint_name!r}")
        leg_id, joint_id = _GUI_JOINT_MAP[joint_name]
        if joint_id == 1:
            raise ValueError("calibration not supported for joint 1 (yaw)")
        self._sdk.send_calibration(leg_id, joint_id)
        logger.info("CMD -> calibrate leg %s joint %s (%s)", leg_id, joint_id, joint_name)

    def refresh_joints_from_telemetry(self) -> None:
        telem = self._sdk.telemetry
        updated = False
        for name, (leg_id, joint_id) in _GUI_JOINT_MAP.items():
            leg_fields = telem.get(leg_id) or {}
            jt = _joint_telemetry_from_leg(name, leg_fields, joint_id)
            if jt is not None:
                self._joints[name] = jt
                updated = True
        if updated:
            self.last_feedback_ts = time.time()