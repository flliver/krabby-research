"""HAL server base class using ZMQ for communication."""

import logging
from typing import Optional

import numpy as np
import zmq

from hal.server.config import HalServerConfig  # Internal import - config is in same package
# OBS_DIM is not used in HAL server

logger = logging.getLogger(__name__)

# Schema version for messages
SCHEMA_VERSION = "1.0"

# Topics for PUB/SUB channels
TOPIC_OBSERVATION = b"observation"  # Complete observation in training format


class HalServerBase:
    """Base class for HAL server.
    
    Provides observation publishing and joint command receiving.
    Uses latest-only semantics (buffer size = 1).
    
    Note: This class uses ZMQ internally as an implementation detail.
    The ZMQ logic is black-boxed - users of this class don't need to know
    about ZMQ. If you need to switch to a different transport later,
    you can create a new implementation with the same interface.
    """

    def __init__(self, config: HalServerConfig):
        """Initialize HAL server.
        
        Server manages its own ZMQ context. For inproc connections,
        clients should use the same context (obtained via get_transport_context()).
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.context = zmq.Context()  # Server owns ZMQ context
        self.observation_socket: Optional[zmq.Socket] = None
        self.command_socket: Optional[zmq.Socket] = None
        self._initialized = False
        self._debug_enabled = False

    def get_transport_context(self):
        """Get transport context for inproc connections.
        
        Returns the ZMQ context that clients can use for inproc connections
        to ensure they're in the same process.
        
        Returns:
            ZMQ context for inproc connections
        """
        return self.context

    def initialize(self) -> None:
        """Initialize ZMQ context and sockets."""
        if self._initialized:
            return

        # Create PUB socket for observation (complete observation in training format)
        self.observation_socket = self.context.socket(zmq.PUB)
        self.observation_socket.setsockopt(zmq.SNDHWM, self.config.observation_buffer_size)
        self.observation_socket.bind(self.config.observation_bind)

        # Create PULL socket for commands (PUSH/PULL pattern with backpressure)
        self.command_socket = self.context.socket(zmq.PULL)
        self.command_socket.setsockopt(zmq.RCVHWM, 5)  # Default HWM of 5 for backpressure
        self.command_socket.bind(self.config.command_bind)

        self._initialized = True
        logger.info(f"HAL server initialized: observation={self.config.observation_bind}, "
                   f"command={self.config.command_bind}")

    def close(self) -> None:
        """Close all sockets and context."""
        if not self._initialized:
            return

        if self.observation_socket:
            self.observation_socket.close()
            self.observation_socket = None

        if self.command_socket:
            self.command_socket.close()
            self.command_socket = None

        if self.context:
            self.context.term()
            self.context = None

        self._initialized = False
        logger.info("HAL server closed")

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug logging.

        When enabled, emits structured logs for all ZMQ messages (send/receive).
        When disabled, no debug logs are emitted to avoid overhead.

        Args:
            enabled: True to enable debug logging, False to disable
        """
        self._debug_enabled = enabled
        if enabled:
            logger.info("Debug logging enabled for HAL server")
        else:
            logger.info("Debug logging disabled for HAL server")

    def is_debug_enabled(self) -> bool:
        """Check if debug logging is enabled.

        Returns:
            True if debug logging is enabled, False otherwise
        """
        return self._debug_enabled

    def set_observation(self, hw_obs: "KrabbyHardwareObservations") -> None:
        """Set/publish hardware observation to clients.

        Sends topic-prefixed multipart message: [topic, schema_version, ...hw_obs_parts]
        The timestamp is included in the hw_obs metadata, so no separate timestamp is needed.

        Args:
            hw_obs: KrabbyHardwareObservations instance

        Raises:
            ValueError: If observation is invalid
            RuntimeError: If server not initialized
        """
        if not self._initialized:
            raise RuntimeError("Server not initialized. Call initialize() first.")

        # Runtime type validation: Validate input type
        from hal.client.data_structures.hardware import KrabbyHardwareObservations
        if not isinstance(hw_obs, KrabbyHardwareObservations):
            raise ValueError(f"hw_obs must be KrabbyHardwareObservations, got {type(hw_obs)}")

        # Serialize hardware observation
        topic = TOPIC_OBSERVATION
        schema_version = SCHEMA_VERSION.encode("utf-8")
        hw_obs_parts = hw_obs.to_bytes()

        # Debug logging (conditional to avoid overhead when disabled)
        if self._debug_enabled:
            logger.debug(
                f"[ZMQ SEND] observation: topic={topic.decode('utf-8')}, "
                f"schema={schema_version.decode('utf-8')}, "
                f"timestamp_ns={hw_obs.timestamp_ns}"
            )

        # Send multipart message: [topic, schema_version, ...hw_obs_parts]
        # Timestamp is already included in hw_obs metadata
        try:
            self.observation_socket.send_multipart(
                [topic, schema_version] + hw_obs_parts, zmq.NOBLOCK
            )
            if self._debug_enabled:
                logger.debug(f"[ZMQ SEND] observation: message sent successfully")
        except zmq.Again:
            if self._debug_enabled:
                logger.debug("[ZMQ SEND] observation: buffer full (HWM reached), message dropped")
            logger.warning("Observation socket buffer full (HWM reached), message dropped")

    def get_joint_command(self, timeout_ms: int = 100) -> Optional["KrabbyDesiredJointPositions"]:
        """Get latest joint command from clients.

        Uses non-blocking poll to check for commands. If command received,
        validates payload and returns full command instance with timestamp and metadata.

        Runtime validation includes:
        - Payload size validation (must be multiple of 4 bytes for float32)
        - Dtype validation (must be float32)
        - Shape validation (must be 1D array)
        - Value validation (no NaN or Inf)

        Args:
            timeout_ms: Poll timeout in milliseconds (default 100ms)

        Returns:
            KrabbyDesiredJointPositions instance with joint positions and timestamp,
            or None if no command received or validation failed

        Raises:
            RuntimeError: If server not initialized
        """
        if not self._initialized:
            raise RuntimeError("Server not initialized. Call initialize() first.")

        # Poll for incoming command
        if self.command_socket.poll(timeout_ms, zmq.POLLIN):
            # Receive command as multipart message
            command_parts = self.command_socket.recv_multipart(zmq.NOBLOCK)

            # Debug logging (conditional to avoid overhead when disabled)
            if self._debug_enabled:
                total_size = sum(len(p) for p in command_parts)
                logger.debug(
                    f"[ZMQ RECV] command: received {len(command_parts)} parts, {total_size} bytes total"
                )

            # Deserialize to KrabbyDesiredJointPositions
            # Validation is handled by from_bytes() and __post_init__()
            from hal.client.data_structures.hardware import KrabbyDesiredJointPositions
            command = KrabbyDesiredJointPositions.from_bytes(command_parts)

            # Debug logging after deserialization
            if self._debug_enabled:
                logger.debug(
                    f"[ZMQ RECV] command: shape={command.joint_positions.shape}, "
                    f"dtype={command.joint_positions.dtype}, "
                    f"min={command.joint_positions.min():.3f}, max={command.joint_positions.max():.3f}, "
                    f"timestamp_ns={command.timestamp_ns}"
                )

            return command

        return None

