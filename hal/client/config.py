"""HAL client configuration classes.

``HalServerConfig`` is re-exported here when ``hal.server`` is installed, so::

    from hal.client.config import HalClientConfig, HalServerConfig

works in full installs. When only the client package is installed,
``HalServerConfig`` is not available (so the client can be used without
the server package).
"""

from dataclasses import dataclass
from typing import Optional

try:
    from hal.server.config import HalServerConfig  # re-export when hal.server is installed
except ModuleNotFoundError:
    HalServerConfig = None  # type: ignore[misc, assignment]


@dataclass
class HalClientConfig:
    """Configuration for HAL client.
    
    Attributes:
        observation_endpoint: Observation endpoint (SUB socket connect address)
            Examples:
                - "inproc://hal_observation" (same process, in-memory)
                - "tcp://localhost:6001" (network, localhost only)
                - "tcp://192.168.1.100:6001" (network, remote host)
        command_endpoint: PUSH target for joint commands (portal teleop → operator bind, inference →
            policy bind on Jetson), or ``None`` for observation-only clients.
            Examples:
                - "inproc://hal_commands" (same process, in-memory)
                - "tcp://localhost:6002" (network, localhost only)
        timeout_s: Timeout in seconds (legacy, not used with PUSH/PULL pattern)
        action_dim: Optional action dimension for validation. If provided,
            commands will be validated to ensure they match this dimension.
            This is the number of joints/actuators (typically 12 for quadruped
            robots with 3 DOF per leg).
            
            **Why optional?** This parameter is optional because the HAL client
            can work with different robot configurations. If not provided, no
            dimension validation is performed. It's recommended to provide this
            for production use to catch dimension mismatches early.
        polling_frequency_hz: Polling frequency in Hz for observation updates
            (default 100.0). Used to calculate poll timeout.
    """
    
    observation_endpoint: str = ""
    command_endpoint: Optional[str] = None
    timeout_s: float = 0.05
    action_dim: Optional[int] = None
    polling_frequency_hz: float = 100.0
    
    def __post_init__(self) -> None:
        """Validate client config."""
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if self.command_endpoint is not None and not str(self.command_endpoint).strip():
            raise ValueError("command_endpoint cannot be empty; use None for observation-only mode")
        if not self.observation_endpoint:
            raise ValueError("observation_endpoint must be provided")
        if self.polling_frequency_hz <= 0:
            raise ValueError("polling_frequency_hz must be > 0")


__all__ = ["HalClientConfig", "HalServerConfig"]

