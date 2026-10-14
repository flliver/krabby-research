"""Protocol for in-process firmware command loops (no firmware.sim imports)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class FirmwareCommandLoop(Protocol):
    """In-process firmware loop: jog, tick, read per-actuator signed PWM."""

    def jog_actuator(self, firmware_actuator_id: str, pwm: int) -> None: ...

    def tick_firmware(self) -> None: ...

    def signed_command_pwm(self, firmware_actuator_id: str) -> int: ...
