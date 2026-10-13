"""Concrete ``FirmwareCommandLoop`` adapters for ``RobotCore`` / ``ThreeBoardRobotCore``."""

from __future__ import annotations

from firmware.sim.robot_core import RobotCore
from firmware.sim.three_board_coordinator import ThreeBoardRobotCore

class RobotCoreCommandLoop:
    """Single-board open-loop drive (six FRONT joints)."""

    def __init__(self, core: RobotCore) -> None:
        self._core = core

    def jog_actuator(self, firmware_actuator_id: str, pwm: int) -> None:
        self._core.jog(firmware_actuator_id, pwm)

    def tick_firmware(self) -> None:
        self._core.tick_firmware_without_physics()

    def signed_command_pwm(self, firmware_actuator_id: str) -> int:
        for actuator in self._core.actuators:
            if actuator.slot.firmware_actuator_id == firmware_actuator_id:
                return self._core.pin_state.signed_pwm_command(*actuator.slot.drive_pins)
        raise KeyError(f"unknown firmware actuator id: {firmware_actuator_id!r}")


class ThreeBoardCommandLoop:
    """Three emulated Mega boards on 18 joints (serial J jog + full firmware loop)."""

    def __init__(self, core: ThreeBoardRobotCore) -> None:
        self._core = core

    def jog_actuator(self, firmware_actuator_id: str, pwm: int) -> None:
        self._core.jog(firmware_actuator_id, pwm)

    def tick_firmware(self) -> None:
        self._core.tick_firmware_without_physics()

    def signed_command_pwm(self, firmware_actuator_id: str) -> int:
        return self._core.command_pwm_for_actuator(firmware_actuator_id)
