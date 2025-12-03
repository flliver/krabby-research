"""HAL server configuration classes."""

from dataclasses import dataclass


@dataclass
class HalServerConfig:
    """Configuration for HAL server.
    
    Requires explicit endpoint configuration for all bindings.
    
    Attributes:
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
            For PULL sockets, this is the bind address where the server listens
            for command requests.
        observation_buffer_size: Buffer size for observation PUB socket
            (default 1 for latest-only semantics). Only the latest message
            is kept, older messages are automatically dropped.
            
            **Optional parameter:** Has default value of 1, so it's optional
            to provide when creating config. Must be >= 1 if provided.
            Cannot be None (type is `int`, not `Optional[int]`).
            
            **Note:** This only applies to observation channels (PUB/SUB).
            Commands use PUSH/PULL pattern with HWM=5 for backpressure.
    """
    
    observation_bind: str
    command_bind: str
    observation_buffer_size: int = 1
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.observation_buffer_size < 1:
            raise ValueError("observation_buffer_size must be >= 1")
        
        if not self.observation_bind:
            raise ValueError("observation_bind is required and cannot be empty")
        
        if not self.command_bind:
            raise ValueError("command_bind is required and cannot be empty")

