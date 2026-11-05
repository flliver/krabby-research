"""Power messages exchanged by the leader MCU and Orin."""
import enum
from dataclasses import dataclass
from typing import Optional, Union

POWER_MSG_PREFIX = "PWR"
POWER_MSG_SCHEMA = 1


class PowerMessageType(enum.Enum):
    POWERING_DOWN = "POWERING_DOWN"
    SHUTDOWN_ACK = "SHUTDOWN_ACK"
    RESUMING = "RESUMING"
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"


class PoweringDownReason(enum.Enum):
    UNDER_VOLTAGE_SOFT = "under_voltage_soft"


class EmergencyShutdownReason(enum.Enum):
    HARD_CUT = "hard_cut"
    OVER_VOLTAGE = "over_voltage"


class ResumingReason(enum.Enum):
    VOLTAGE_RECOVERED = "voltage_recovered"


PowerMessageReason = Union[
    PoweringDownReason,
    EmergencyShutdownReason,
    ResumingReason,
]


@dataclass
class PowerMessage:
    type: PowerMessageType
    reason: Optional[PowerMessageReason] = None
    schema_version: int = POWER_MSG_SCHEMA

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != POWER_MSG_SCHEMA
        ):
            raise ValueError(
                f"unsupported power-message schema: {self.schema_version!r}"
            )
        expected_reason_type = {
            PowerMessageType.POWERING_DOWN: PoweringDownReason,
            PowerMessageType.EMERGENCY_SHUTDOWN: EmergencyShutdownReason,
            PowerMessageType.RESUMING: ResumingReason,
        }.get(self.type)
        if expected_reason_type is None:
            if self.reason is not None:
                raise ValueError(f"{self.type.value} does not carry a reason")
        elif not isinstance(self.reason, expected_reason_type):
            raise ValueError(
                f"{self.type.value} requires {expected_reason_type.__name__}"
            )

    def format_line(self) -> str:
        """Render the wire line (no trailing newline)."""
        parts = [POWER_MSG_PREFIX, str(self.schema_version), self.type.value]
        if self.reason is not None:
            parts.append(self.reason.value)
        return " ".join(parts)

    @classmethod
    def parse(cls, line: str) -> Optional["PowerMessage"]:
        """Parse a supported, well-formed PWR line."""
        tokens = line.split()
        if len(tokens) < 3 or tokens[0] != POWER_MSG_PREFIX:
            return None
        try:
            schema = int(tokens[1])
        except ValueError:
            return None
        if schema != POWER_MSG_SCHEMA:
            return None
        try:
            mtype = PowerMessageType(tokens[2])
        except ValueError:
            return None
        reason_type = {
            PowerMessageType.POWERING_DOWN: PoweringDownReason,
            PowerMessageType.EMERGENCY_SHUTDOWN: EmergencyShutdownReason,
            PowerMessageType.RESUMING: ResumingReason,
        }.get(mtype)
        expected_token_count = 3 if reason_type is None else 4
        if len(tokens) != expected_token_count:
            return None
        reason: Optional[PowerMessageReason] = None
        if reason_type is not None:
            try:
                reason = reason_type(tokens[3])
            except ValueError:
                return None
        return cls(type=mtype, reason=reason, schema_version=schema)
