"""HAL client configuration classes.

This module also re-exports ``HalServerConfig`` for backwards compatibility
so existing imports like::

    from hal.client.config import HalClientConfig, HalServerConfig

continue to work even though ``HalServerConfig`` lives in ``hal.server.config``.
"""

from dataclasses import dataclass
from typing import Optional

from hal.server.config import HalServerConfig  # re-export for backwards compatibility


@dataclass
class HalClientConfig:
    """Configuration for HAL client.
    
    Attributes:
        observation_endpoint: Observation endpoint (SUB socket connect address)
            Examples:
                - "inproc://hal_observation" (same process, in-memory)
                - "tcp://localhost:6001" (network, localhost only)
                - "tcp://192.168.1.100:6001" (network, remote host)
        command_endpoint: Command endpoint (REQ socket connect address)
            Examples:
                - "inproc://hal_commands" (same process, in-memory)
                - "tcp://localhost:6002" (network, localhost only)
        timeout_s: Timeout in seconds for REQ/REP operations (default 0.05)
        action_dim: Optional action dimension for validation. If provided,
            commands will be validated to ensure they match this dimension.
            This is the number of joints/actuators (typically 12 for quadruped
            robots with 3 DOF per leg).
        polling_frequency_hz: Polling frequency in Hz for observation updates
            (default 100.0). Used to calculate poll timeout.
    """
    
    observation_endpoint: str = ""
    command_endpoint: str = ""
    timeout_s: float = 0.05
    action_dim: Optional[int] = None
    polling_frequency_hz: float = 100.0
    
    def __post_init__(self) -> None:
        """Validate client config."""
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if not self.command_endpoint:
            raise ValueError("command_endpoint must be provided")
        if not self.observation_endpoint:
            raise ValueError("observation_endpoint must be provided")
        if self.polling_frequency_hz <= 0:
            raise ValueError("polling_frequency_hz must be > 0")
    
    @classmethod
    def from_base_port(
        cls, 
        base_port: int = 6000, 
        timeout_s: float = 0.05,
        action_dim: Optional[int] = None,
        polling_frequency_hz: float = 100.0,
    ) -> "HalClientConfig":
        """Create config from base port (assumes localhost).
        
        Args:
            base_port: Base port number (default 6000)
            timeout_s: Timeout in seconds (default 0.05)
            action_dim: Optional action dimension for validation
            polling_frequency_hz: Polling frequency in Hz (default 100.0)
        
        Returns:
            HalClientConfig instance
        """
        return cls(
            observation_endpoint=f"tcp://localhost:{base_port + 1}",
            command_endpoint=f"tcp://localhost:{base_port + 2}",
            timeout_s=timeout_s,
            action_dim=action_dim,
            polling_frequency_hz=polling_frequency_hz,
        )
    
    @classmethod
    def from_endpoints(
        cls,
        observation_endpoint: str,
        command_endpoint: str,
        timeout_s: float = 0.05,
        action_dim: Optional[int] = None,
        polling_frequency_hz: float = 100.0,
    ) -> "HalClientConfig":
        """Create config from explicit endpoints.
        
        Args:
            observation_endpoint: Observation endpoint (required)
            command_endpoint: Command endpoint (required)
            timeout_s: Timeout in seconds (default 0.05)
            action_dim: Optional action dimension for validation (default None)
            polling_frequency_hz: Polling frequency in Hz (default 100.0)
        
        Returns:
            HalClientConfig instance
        """
        if observation_endpoint is None:
            raise ValueError("observation_endpoint must be provided")
        if command_endpoint is None:
            raise ValueError("command_endpoint must be provided")
        
        return cls(
            observation_endpoint=observation_endpoint,
            command_endpoint=command_endpoint,
            timeout_s=timeout_s,
            action_dim=action_dim,
            polling_frequency_hz=polling_frequency_hz,
        )


__all__ = ["HalClientConfig", "HalServerConfig"]

