import math
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True, slots=True)
class ImuTelemetry:
    accel: Tuple[float, float, float]
    gyro: Tuple[float, float, float]
    temp_c: float
    valid: bool

    TAG = "IMU"
    STANDARD_GRAVITY_MPS2 = 9.80665
    VALID_TOKENS = ("0", "1")

    @classmethod
    def from_tokens(cls, tokens) -> Optional["ImuTelemetry"]:
        if not tokens:
            return None

        try:
            tag, ax, ay, az, gx, gy, gz, temp, valid_token = tokens
            accel = tuple(float(token) for token in (ax, ay, az))
            gyro = tuple(float(token) for token in (gx, gy, gz))
            temp_c = float(temp)
        except ValueError:
            return None

        if tag != cls.TAG or valid_token not in cls.VALID_TOKENS:
            return None
        if not all(math.isfinite(value) for value in accel + gyro):
            return None
        if not math.isfinite(temp_c):
            temp_c = float("nan")
        return cls(accel, gyro, temp_c, valid_token == "1")

    @classmethod
    def from_segment(cls, segment: str) -> Optional["ImuTelemetry"]:
        return cls.from_tokens(segment.split())

    @property
    def accel_g(self) -> Tuple[float, float, float]:
        return tuple(value / self.STANDARD_GRAVITY_MPS2 for value in self.accel)

    @property
    def gyro_dps(self) -> Tuple[float, float, float]:
        return tuple(math.degrees(value) for value in self.gyro)

    @property
    def roll_from_accel_deg(self) -> float:
        _, accel_y, accel_z = self.accel
        return math.degrees(math.atan2(accel_y, accel_z))

    @property
    def pitch_from_accel_deg(self) -> float:
        accel_x, accel_y, accel_z = self.accel
        return math.degrees(math.atan2(-accel_x, math.hypot(accel_y, accel_z)))

    def format_compact(self) -> str:
        accel_x, accel_y, accel_z = self.accel
        gyro_x, gyro_y, gyro_z = self.gyro
        stale = "" if self.valid else " STALE"
        return (
            f"a:({accel_x:.2f},{accel_y:.2f},{accel_z:.2f})m/s2 "
            f"g:({gyro_x:.3f},{gyro_y:.3f},{gyro_z:.3f})rad/s "
            f"{self.temp_c:.1f}C{stale}"
        )
