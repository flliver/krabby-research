import enum
from dataclasses import dataclass
from typing import Optional, Union


class PackVoltageRegion(enum.IntEnum):
    """Voltage-region codes carried in battery telemetry."""

    NORMAL = 0
    WARN = 1
    SOFT_CUT = 2
    HARD_CUT = 3
    OVER_VOLT = 4


def _known_or_raw(enum_type, value):
    """Keep a defined value as its enum member, an undefined one as a plain int."""
    try:
        return enum_type(value)
    except ValueError:
        return value


def _label(value) -> str:
    return value.name if isinstance(value, enum.IntEnum) else str(value)


@dataclass(frozen=True, slots=True)
class BatteryTelemetry:
    pack_volts: float
    pack_current_amperes: float
    pack_power_watts: float
    pack_charge_coulombs: float
    battery_a_volts: float
    battery_b_volts: float
    # Divergence assumed when either monitor is unavailable or a voltage is non-finite.
    divergence: bool
    # Unknown bytes are retained as plain ints rather than coerced or dropped, so
    # a firmware that gains a value stays parseable and the unknown stays visible.
    pack_region: Union[PackVoltageRegion, int]
    # Each flag reports whether all four reads from that monitor succeeded.
    pack_valid: bool
    midpoint_valid: bool

    TAG = "BATT"
    TOKEN_COUNT = 11
    VALID_TOKENS = ("0", "1")

    @classmethod
    def from_tokens(cls, tokens) -> Optional["BatteryTelemetry"]:
        if not tokens or len(tokens) != cls.TOKEN_COUNT:
            return None
        (tag, pack_v, pack_i, pack_w, pack_charge, battery_a, battery_b,
         divergence, region, pack_valid, midpoint_valid) = tokens
        if tag != cls.TAG:
            return None
        if any(token not in cls.VALID_TOKENS
               for token in (divergence, pack_valid, midpoint_valid)):
            return None
        try:
            region_value = int(region)
            if not 0 <= region_value <= 255:
                return None
            return cls(
                pack_volts=float(pack_v),
                pack_current_amperes=float(pack_i),
                pack_power_watts=float(pack_w),
                pack_charge_coulombs=float(pack_charge),
                battery_a_volts=float(battery_a),
                battery_b_volts=float(battery_b),
                divergence=divergence == "1",
                pack_region=_known_or_raw(PackVoltageRegion, region_value),
                pack_valid=pack_valid == "1",
                midpoint_valid=midpoint_valid == "1",
            )
        except ValueError:
            return None

    @classmethod
    def from_segment(cls, segment: str) -> Optional["BatteryTelemetry"]:
        return cls.from_tokens(segment.split())

    @staticmethod
    def format_battery_voltage(volts: float) -> str:
        return f"{volts:.2f}"

    @property
    def split_available(self) -> bool:
        return self.pack_valid and self.midpoint_valid

    def format_compact(self) -> str:
        region = _label(self.pack_region)
        divergence = " DIVERGE" if self.divergence else ""
        down = "".join(
            f" {name}:DOWN"
            for name, ok in (("pack", self.pack_valid), ("mid", self.midpoint_valid))
            if not ok
        )
        return (
            f"pack:{self.pack_volts:.2f}V "
            f"{self.pack_current_amperes:+.2f}A "
            f"{self.pack_power_watts:.1f}W "
            f"A:{self.format_battery_voltage(self.battery_a_volts)}V "
            f"B:{self.format_battery_voltage(self.battery_b_volts)}V "
            f"q:{self.pack_charge_coulombs:.0f}C {region}{divergence}{down}"
        )
