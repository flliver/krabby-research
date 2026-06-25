from dataclasses import dataclass
from typing import Tuple, Optional

# Wire format: must match firmware (actuator_manager.h + arduino.ino).
# Line starts with a role prefix "FRONT; ", "UNKNOWN; ", "LEFT; ", or "RIGHT; " then semicolon-separated segments.
# Forwarded lines from left/right already include their role (LEFT; / RIGHT; ).
# Example: "FRONT; FLHY 0.123 0 512 1 0 0 128 0;FLHL ...;..."
# Segment format: <name> <pos> <pot> <current> <enL> <enR> <pwmL> <pwmR> <saf> [<cal_state>]
# saf: cumulative HallA edge count since boot (pins depend on KRABBY_PIN_REV in board_pins.h).
# cal_state (optional 10th token, older firmware omits it): 0=UNCALIBRATED, 1=PARTIALLY_
# CALIBRATED (Hall, unanchored — pos is relative until it self-heals against an end-stop),
# 2=FULLY_CALIBRATED. M17 Task 2 §6.5.

CAL_STATE_NAMES = {0: "UNCAL", 1: "PARTIAL", 2: "FULL"}


@dataclass
class JointTelemetry:
    name: str
    pos: float
    pot: int
    current: int
    en: Tuple[int, int]
    pwm: Tuple[int, int]
    saf: int
    cal_state: int = 0  # 0=UNCAL, 1=PARTIAL, 2=FULL (default for pre-cal-state firmware)

    # Role prefix (first segment of a line); not a joint.
    ROLE_PREFIXES = ("JT", "FRONT", "UNKNOWN", "LEFT", "RIGHT")

    @classmethod
    def from_tokens(cls, tokens) -> Optional["JointTelemetry"]:
        if not tokens:
            return None
        if tokens[0] in cls.ROLE_PREFIXES:
            tokens = tokens[1:] if tokens[0] == "JT" else None
        # 9 tokens (legacy) or 10 (with cal_state). Anything else is a corrupt segment.
        if not tokens or len(tokens) not in (9, 10):
            return None
        cal_state = tokens[9] if len(tokens) == 10 else "0"
        name, pos, pot, cur, enL, enR, pwmL, pwmR, saf = tokens[:9]
        try:
            return cls(
                name=name,
                pos=float(pos),
                pot=int(pot),
                current=int(cur),
                en=(int(enL), int(enR)),
                pwm=(int(pwmL), int(pwmR)),
                saf=int(saf),
                cal_state=int(cal_state),
            )
        except ValueError:
            return None

    @property
    def cal_state_name(self) -> str:
        return CAL_STATE_NAMES.get(self.cal_state, "?")

    @classmethod
    def parse_line(cls, line: str):
        joints = []
        for seg in line.strip().split(";"):
            seg = seg.strip()
            if not seg:
                continue
            tokens = seg.split()
            jt = cls.from_tokens(tokens)
            if jt:
                joints.append(jt)
        return joints

    def format_compact(self, target: Optional[float] = None) -> str:
        pos_part = f"{self.pos:.3f}"
        if target is not None:
            pos_part = f"{pos_part}/{target:.3f}"
        return (
            f"{self.name}:{pos_part},{self.pot},{self.current},"
            f"({self.en[0]},{self.en[1]}),({self.pwm[0]},{self.pwm[1]}),{self.saf}"
        )
