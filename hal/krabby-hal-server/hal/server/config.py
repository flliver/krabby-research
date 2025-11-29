"""HAL server configuration classes."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HalServerConfig:
    """Configuration for HAL server.
    
    Supports both port-based and explicit endpoint configuration.
    If endpoints are provided, they take precedence over base_port.
    
    Attributes:
        base_port: Base port number (default 6000)
        observation_bind: Observation endpoint where server publishes observations.
            Examples:
                - "inproc://hal_observation" (same process, in-memory)
                - "tcp://*:6001" (network, all interfaces, port 6001)
                - "tcp://localhost:6001" (network, localhost only)
            For PUB sockets, this is the bind address where the server listens
            for subscribers to connect.
        command_bind: Command endpoint where server receives commands.
            Examples:
                - "inproc://hal_commands" (same process, in-memory)
                - "tcp://*:6002" (network, all interfaces, port 6002)
            For REP sockets, this is the bind address where the server listens
            for command requests.
        observation_buffer_size: Buffer size for observation PUB socket
            (default 1 for latest-only semantics). Only the latest message
            is kept, older messages are automatically dropped.
            
            **Optional parameter:** Has default value of 1, so it's optional
            to provide when creating config. Must be >= 1 if provided.
            Cannot be None (type is `int`, not `Optional[int]`).
            
            **Note:** This only applies to observation channels (PUB/SUB).
            Commands use REQ/REP pattern which doesn't use HWM/buffer size
            (REQ/REP ensures guaranteed delivery without buffer limits).
    """
    
    base_port: int = 6000
    observation_bind: Optional[str] = None
    command_bind: Optional[str] = None
    observation_buffer_size: int = 1  # Renamed from hwm
    
    def __post_init__(self) -> None:
        """Validate and set default endpoints if not provided."""
        if self.observation_buffer_size < 1:
            raise ValueError("observation_buffer_size must be >= 1")
        
        # If observation_bind not provided, use port-based default
        if self.observation_bind is None:
            if self.base_port < 1024 or self.base_port > 65535:
                raise ValueError("base_port must be in range [1024, 65535]")
            self.observation_bind = f"tcp://*:{self.base_port + 1}"
        
        # Only validate base_port if we're using port-based endpoints
        using_explicit_endpoints = (
            self.observation_bind is not None
            and self.command_bind is not None
        )
        if not using_explicit_endpoints:
            if self.base_port < 1024 or self.base_port > 65535:
                raise ValueError("base_port must be in range [1024, 65535]")
        
        # If endpoints not provided, use port-based defaults
        if self.command_bind is None:
            self.command_bind = f"tcp://*:{self.base_port + 2}"
    
    @classmethod
    def from_endpoints(
        cls,
        observation_bind: str,
        command_bind: str,
        observation_buffer_size: int = 1,
    ) -> "HalServerConfig":
        """Create config from explicit endpoints.
        
        Args:
            observation_bind: Observation endpoint (required)
            command_bind: Command endpoint (required)
            observation_buffer_size: Buffer size for latest-only semantics (default 1, optional)
        
        Returns:
            HalServerConfig instance
        """
        if observation_bind is None:
            raise ValueError("observation_bind must be provided")
        if command_bind is None:
            raise ValueError("command_bind must be provided")
        
        return cls(
            base_port=0,  # Not used when endpoints are explicit
            observation_bind=observation_bind,
            command_bind=command_bind,
            observation_buffer_size=observation_buffer_size,
        )

