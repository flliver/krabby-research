import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class BatteryTelemetry:
    pack_v: float
    pack_i: float
    pack_w: float
    pack_charge: float
    batt_a_v: float
    batt_b_v: float
    divergence_flag: int
    power_state: int

    TAG = "BATT"

    @classmethod
    def from_tokens(cls, tokens) -> Optional["BatteryTelemetry"]:
        if not tokens or len(tokens) != 9:
            return None
        try:
            tag = tokens[0]
            if tag != cls.TAG:
                return None
            pack_v = float(tokens[1])
            pack_i = float(tokens[2])
            pack_w = float(tokens[3])
            pack_charge = float(tokens[4])
            batt_a_v = float(tokens[5])
            batt_b_v = float(tokens[6])
            divergence_flag = int(tokens[7])
            power_state = int(tokens[8])
        except ValueError:
            return None

        fields = (pack_v, pack_i, pack_w, pack_charge, batt_a_v, batt_b_v)
        if not all(math.isfinite(value) for value in fields):
            return None
        if divergence_flag not in (0, 1):
            return None
        if power_state < 0 or power_state > 6:
            return None

        return cls(
            pack_v=pack_v,
            pack_i=pack_i,
            pack_w=pack_w,
            pack_charge=pack_charge,
            batt_a_v=batt_a_v,
            batt_b_v=batt_b_v,
            divergence_flag=divergence_flag,
            power_state=power_state,
        )

    @classmethod
    def from_segment(cls, segment: str) -> Optional["BatteryTelemetry"]:
        return cls.from_tokens(segment.split())

    def format_compact(self) -> str:
        return (
            f"pack {self.pack_v:.2f}V {self.pack_i:.3f}A {self.pack_w:.2f}W "
            f"A {self.batt_a_v:.2f}V B {self.batt_b_v:.2f}V "
            f"div={self.divergence_flag} pwr={self.power_state}"
        )
