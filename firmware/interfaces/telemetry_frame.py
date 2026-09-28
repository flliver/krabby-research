from dataclasses import dataclass
from typing import Optional, Tuple

from firmware.interfaces.imu_telemetry import ImuTelemetry
from firmware.interfaces.joint_telemetry import JointTelemetry


@dataclass(frozen=True, slots=True)
class TelemetryFrame:
    joints: Tuple[JointTelemetry, ...] = ()
    imu: Optional[ImuTelemetry] = None

    ROLE_TAGS = ("FRONT", "UNKWN", "LEFT", "RIGHT")

    @classmethod
    def role_from_line(cls, line: str) -> Optional[str]:
        role, delimiter, _ = line.partition(";")
        if not delimiter:
            return None
        role = role.strip()
        return role if role in cls.ROLE_TAGS else None

    @classmethod
    def is_telemetry_line(cls, line: str) -> bool:
        return cls.role_from_line(line) is not None

    @classmethod
    def parse_line(cls, line: str) -> "TelemetryFrame":
        joints = []
        imu = None
        for segment in line.strip().split(";"):
            tokens = segment.split()
            if not tokens:
                continue
            if tokens[0] == ImuTelemetry.TAG:
                parsed_imu = ImuTelemetry.from_tokens(tokens)
                if parsed_imu is not None:
                    imu = parsed_imu
            else:
                joint = JointTelemetry.from_tokens(tokens)
                if joint is not None:
                    joints.append(joint)
        return cls(joints=tuple(joints), imu=imu)
