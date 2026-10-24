"""In-memory Mega pin state recorded from host Arduino output shims."""

from __future__ import annotations

from dataclasses import dataclass

# Match AVR Arduino.h values used by actuator_manager.h
INPUT = 0
OUTPUT = 1
INPUT_PULLUP = 2
LOW = 0
HIGH = 1


@dataclass
class PinRecord:
    mode: int | None = None
    digital: int = LOW
    pwm: int = 0


class PinStateModel:
    """Per-pin mode, digital level, and PWM duty for sim and actuator readback."""

    def __init__(self) -> None:
        self._pins: dict[int, PinRecord] = {}

    def reset(self) -> None:
        self._pins.clear()

    def pin_mode(self, pin: int, mode: int) -> None:
        self._record(pin).mode = mode

    def digital_write(self, pin: int, value: int) -> None:
        self._record(pin).digital = HIGH if value else LOW

    def analog_write(self, pin: int, duty: int) -> None:
        if duty < 0:
            duty = 0
        elif duty > 255:
            duty = 255
        self._record(pin).pwm = duty

    def get_mode(self, pin: int) -> int | None:
        return self._record(pin).mode

    def digital_read(self, pin: int) -> int:
        """Recorded digital level from digitalWrite (not AVR digitalRead)."""
        return self._record(pin).digital

    def pwm_duty(self, pin: int) -> int:
        return self._record(pin).pwm

    def is_digital_high(self, pin: int) -> bool:
        return self.digital_read(pin) == HIGH

    def signed_pwm_command(self, pin_pwm_r: int, pin_pwm_l: int, pin_en: int) -> int:
        """Motor drive from recorded pins: (EN high) * (PWM_R - PWM_L)."""
        if not self.is_digital_high(pin_en):
            return 0
        return self.pwm_duty(pin_pwm_r) - self.pwm_duty(pin_pwm_l)

    def _record(self, pin: int) -> PinRecord:
        if pin not in self._pins:
            self._pins[pin] = PinRecord()
        return self._pins[pin]
