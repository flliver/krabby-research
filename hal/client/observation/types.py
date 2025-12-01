"""HAL observation types.

Generic HAL types for robot control.
"""

import time
from dataclasses import dataclass


@dataclass
class NavigationCommand:
    """Navigation command for robot movement.

    Attributes:
        timestamp_ns: Timestamp in nanoseconds
        schema_version: Schema version string (e.g., "1.0")
        vx: Forward velocity (m/s)
        vy: Lateral velocity (m/s)
        yaw_rate: Angular velocity (rad/s)
    """

    timestamp_ns: int
    schema_version: str = "1.0"
    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate navigation command."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if not isinstance(self.schema_version, str):
            raise ValueError("schema_version must be a string")

    @classmethod
    def create_now(cls, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0) -> "NavigationCommand":
        """Create navigation command with current timestamp."""
        return cls(
            timestamp_ns=time.time_ns(),
            schema_version="1.0",
            vx=vx,
            vy=vy,
            yaw_rate=yaw_rate,
        )
