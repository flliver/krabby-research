"""HAL configuration classes for server and client setup."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HalServerConfig:
    """Configuration for HAL server.

    Supports both port-based and explicit endpoint configuration.
    If endpoints are provided, they take precedence over base_port.

    Attributes:
        base_port: Base port number (default 6000)
        observation_bind: Observation endpoint (PUB socket bind address)
        command_bind: Command endpoint (REP socket bind address)
        hwm: High-watermark for PUB sockets (default 1 for latest-only)
    """

    base_port: int = 6000
    observation_bind: Optional[str] = None
    command_bind: Optional[str] = None
    hwm: int = 1

    def __post_init__(self) -> None:
        """Validate and set default endpoints if not provided."""
        if self.hwm < 1:
            raise ValueError("hwm must be >= 1")
        
        # If observation_bind not provided, use port-based default
        if self.observation_bind is None:
            if self.base_port < 1024 or self.base_port > 65535:
                raise ValueError("base_port must be in range [1024, 65535]")
            self.observation_bind = f"tcp://*:{self.base_port + 1}"
        
        # Only validate base_port if we're using port-based endpoints
        # (i.e., when explicit endpoints are not provided)
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
        hwm: int = 1,
    ) -> "HalServerConfig":
        """Create config from explicit endpoints.

        Args:
            observation_bind: Observation endpoint (required)
            command_bind: Command endpoint (required)
            hwm: High-watermark (default 1)

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
            hwm=hwm,
        )


@dataclass
class HalClientConfig:
    """Configuration for HAL client.

    Attributes:
        observation_endpoint: Observation endpoint (SUB socket connect address)
        command_endpoint: Command endpoint (REQ socket connect address)
        timeout_s: Timeout in seconds for REQ/REP operations (default 0.05)
        action_dim: Optional action dimension for validation (default None)
    """

    observation_endpoint: str = ""
    command_endpoint: str = ""
    timeout_s: float = 0.05
    action_dim: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate client config."""
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if not self.command_endpoint:
            raise ValueError("command_endpoint must be provided")
        if not self.observation_endpoint:
            raise ValueError("observation_endpoint must be provided")

    @classmethod
    def from_base_port(cls, base_port: int = 6000, timeout_s: float = 0.05) -> "HalClientConfig":
        """Create config from base port (assumes localhost).

        Args:
            base_port: Base port number (default 6000)
            timeout_s: Timeout in seconds (default 0.05)

        Returns:
            HalClientConfig instance
        """
        return cls(
            observation_endpoint=f"tcp://localhost:{base_port + 1}",
            command_endpoint=f"tcp://localhost:{base_port + 2}",
            timeout_s=timeout_s,
        )

    @classmethod
    def from_endpoints(
        cls,
        observation_endpoint: str,
        command_endpoint: str,
        timeout_s: float = 0.05,
        action_dim: Optional[int] = None,
    ) -> "HalClientConfig":
        """Create config from explicit endpoints.

        Args:
            observation_endpoint: Observation endpoint (required)
            command_endpoint: Command endpoint (required)
            timeout_s: Timeout in seconds (default 0.05)
            action_dim: Optional action dimension for validation (default None)

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
        )

