from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Tuple, Optional

# Must match actuator_manager.h telemetry output.
# Segment format: <name> <pos> <pot> <current> <enL> <enR> <pwmL> <pwmR> <saf> [<connection>]
# connection: locally composed state, 0 unknown / 1 connected / 2 disconnected.


class ActuatorConnection(IntEnum):
    UNKNOWN = 0
    CONNECTED = 1
    DISCONNECTED = 2


@dataclass(frozen=True, slots=True)
class JointTelemetry:
    name: str
    pos: float
    pot: int
    current: int
    en: Tuple[int, int]
    pwm: Tuple[int, int]
    saf: int
    connection_state: ActuatorConnection = ActuatorConnection.UNKNOWN

    @classmethod
    def from_tokens(cls, tokens) -> Optional["JointTelemetry"]:
        if not tokens:
            return None
        if not tokens or len(tokens) not in (9, 10):
            return None
        name, pos, pot, cur, enL, enR, pwmL, pwmR, saf = tokens[:9]
        try:
            position = float(pos)
            connection_state = (
                ActuatorConnection(int(tokens[9]))
                if len(tokens) == 10
                else ActuatorConnection.UNKNOWN
            )
            # A non-finite position was the legacy disconnection encoding.
            if not math.isfinite(position):
                connection_state = ActuatorConnection.DISCONNECTED
            return cls(
                name=name,
                pos=position,
                pot=int(pot),
                current=int(cur),
                en=(int(enL), int(enR)),
                pwm=(int(pwmL), int(pwmR)),
                saf=int(saf),
                connection_state=connection_state,
            )
        except ValueError:
            return None

    @property
    def connected(self) -> bool:
        return (
            math.isfinite(self.pos)
            and self.connection_state is not ActuatorConnection.DISCONNECTED
        )

    def format_compact(self, target: Optional[float] = None) -> str:
        if not self.connected:
            return f"{self.name}:DISC,{self.pot},{self.current}"
        pos_part = f"{self.pos:.3f}"
        if target is not None:
            pos_part = f"{pos_part}/{target:.3f}"
        return (
            f"{self.name}:{pos_part},{self.pot},{self.current},"
            f"({self.en[0]},{self.en[1]}),({self.pwm[0]},{self.pwm[1]}),{self.saf}"
        )
