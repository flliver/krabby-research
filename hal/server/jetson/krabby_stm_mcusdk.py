"""Jetson HAL SDK: 18-joint commands to STM CAN-FD duties."""

from __future__ import annotations

import logging
from hal.client.data_structures.hardware import JointCommand
from firmware.stm.mcu_can import KrabbyMCUCANSDK

logger = logging.getLogger(__name__)

_DUTY_MIN = -1.0
_DUTY_MAX = 1.0


def _clamp_duty(value: float) -> float:
    if value < _DUTY_MIN:
        return _DUTY_MIN
    if value > _DUTY_MAX:
        return _DUTY_MAX
    return float(value)


class KrabbyStmMCUSDK:
    """Maps HAL ``JointCommand`` duties [-1, 1] to ``KrabbyMCUCANSDK.set_duties``."""

    def __init__(
        self,
        mcu_joints: tuple[str, ...],
        *,
        virtual: bool = False,
        auto_connect: bool = True,
    ) -> None:
        if len(mcu_joints) != 18:
            raise ValueError(
                f"mcu_joints must have 18 names (STM expects 18 duties), got {len(mcu_joints)}"
            )
        self._mcu_joints = mcu_joints
        self._sdk = KrabbyMCUCANSDK(virtual=virtual)
        self._bus_up = False

        logger.info(
            "KrabbyStmMCUSDK initialized (virtual=%s, auto_connect=%s)",
            virtual,
            auto_connect,
        )
        if auto_connect:
            self.connect()

    def connect(self) -> bool:
        if self._bus_up:
            logger.warning("STM CAN SDK already connected")
            return True
        self._sdk.connect()
        self._bus_up = True
        logger.info("STM CAN SDK connected")
        return True

    def _has_leg_telemetry(self) -> bool:
        telem = self._sdk.telemetry
        return any(fields for fields in telem.values())

    def is_connected(self) -> bool:
        """True after connect and at least one leg has reported telemetry."""
        if not self._bus_up:
            return False
        return self._has_leg_telemetry()

    def apply_command(self, command: JointCommand) -> None:
        if not self._bus_up:
            raise RuntimeError("STM CAN SDK is not connected. Call connect() first.")

        cmd_dict = command.to_positions_dict()
        duties: list[float] = []
        for name in self._mcu_joints:
            if name not in cmd_dict:
                raise ValueError(
                    f"command must contain key {name!r} (and all names in mcu_joints)"
                )
            duties.append(_clamp_duty(float(cmd_dict[name])))
        self._sdk.set_duties(tuple(duties))
        logger.debug(
            "KrabbyStmMCUSDK: set_duties for %d joints (timestamp_ns=%s)",
            len(duties),
            command.timestamp_ns,
        )

    def close(self) -> None:
        self._sdk.close()
        self._bus_up = False
        logger.info("STM CAN SDK connection closed")