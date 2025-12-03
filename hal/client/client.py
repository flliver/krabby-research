"""HAL client implementation using ZMQ."""

import logging
import time
from typing import Optional

import numpy as np
import zmq

# InferenceResponse is not used in HAL client
from hal.client.config import HalClientConfig
from hal.client.data_structures.hardware import KrabbyHardwareObservations

logger = logging.getLogger(__name__)

# Topics for PUB/SUB channels
TOPIC_OBSERVATION = b"observation"  # Complete observation in training format

# Schema version
SCHEMA_VERSION = "1.0"


class HalClient:
    """HAL client for subscribing to observations and sending commands.

    Uses SUB socket for observations (complete observation in training format, HWM=1 for latest-only),
    and PUSH socket for commands (PUSH/PULL pattern with backpressure, HWM=5).
    """

    def __init__(self, config: HalClientConfig, context: Optional[zmq.Context] = None):
        """Initialize HAL client.

        Args:
            config: Client configuration
            context: Optional shared ZMQ context (useful for inproc connections)
        """
        self.config = config
        self.context: Optional[zmq.Context] = context
        self._context_owned = context is None  # Track if we own the context
        self.observation_socket: Optional[zmq.Socket] = None
        self.command_socket: Optional[zmq.Socket] = None

        # Latest buffers (HWM=1 ensures only latest is kept)
        self._latest_hw_obs: Optional[KrabbyHardwareObservations] = None

        self._initialized = False
        self._debug_enabled = False

    def initialize(self) -> None:
        """Initialize ZMQ context and sockets."""
        if self._initialized:
            return

        if self.context is None:
            self.context = zmq.Context()
            self._context_owned = True

        # Create SUB socket for observation (complete observation in training format)
        self.observation_socket = self.context.socket(zmq.SUB)
        self.observation_socket.setsockopt(zmq.RCVHWM, 1)  # Latest-only
        self.observation_socket.setsockopt(zmq.SUBSCRIBE, TOPIC_OBSERVATION)
        if not self.config.observation_endpoint:
            raise ValueError("observation_endpoint must be set in config")
        self.observation_socket.connect(self.config.observation_endpoint)

        # Create PUSH socket for commands (PUSH/PULL pattern with backpressure)
        self.command_socket = self.context.socket(zmq.PUSH)
        self.command_socket.setsockopt(zmq.SNDHWM, 5)  # Default HWM of 5 for backpressure
        self.command_socket.connect(self.config.command_endpoint)

        self._initialized = True
        logger.info(f"HAL client initialized: observation={self.config.observation_endpoint}, "
                   f"command={self.config.command_endpoint}")

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

        if self.context and self._context_owned:
            self.context.term()
            self.context = None

        self._initialized = False
        logger.info("HAL client closed")

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
            logger.info("Debug logging enabled for HAL client")
        else:
            logger.info("Debug logging disabled for HAL client")

    def is_debug_enabled(self) -> bool:
        """Check if debug logging is enabled.

        Returns:
            True if debug logging is enabled, False otherwise
        """
        return self._debug_enabled

    def poll(self, timeout_ms: int = 10) -> Optional[KrabbyHardwareObservations]:
        """Poll for latest hardware observation messages (non-blocking).

        Updates latest buffers with newest messages. Old messages are
        automatically dropped due to buffer size=1 (latest-only semantics).

        Args:
            timeout_ms: Poll timeout in milliseconds (default 10ms)

        Returns:
            KrabbyHardwareObservations if new data was received,
            None if timeout or no new data available.

        Raises:
            RuntimeError: If client not initialized
            ValueError: If message format is invalid (wrong schema version, wrong number of parts)
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        # Poll observation socket for hardware observations
        if self.observation_socket.poll(timeout_ms, zmq.POLLIN):
            try:
                parts = self.observation_socket.recv_multipart(zmq.NOBLOCK)
                if len(parts) >= 3:
                    topic, schema_version_bytes = parts[0], parts[1]
                    schema_version = schema_version_bytes.decode("utf-8")

                    # Debug logging (conditional to avoid overhead when disabled)
                    if self._debug_enabled:
                        logger.debug(
                            f"[ZMQ RECV] observation: topic={topic.decode('utf-8')}, "
                            f"schema={schema_version}, "
                            f"num_parts={len(parts)}"
                        )

                    # Validate schema version
                    if schema_version != SCHEMA_VERSION:
                        error_msg = f"Unsupported schema version: {schema_version}, expected {SCHEMA_VERSION}"
                        logger.error(f"[ZMQ RECV] observation: {error_msg}")
                        raise ValueError(error_msg)

                    # Expected format: [topic, schema_version, ...hw_obs_parts (6 parts)]
                    # Total: 2 + 6 = 8 parts
                    # Timestamp is included in hw_obs metadata, no separate timestamp part
                    if len(parts) != 8:
                        error_msg = f"Invalid number of parts: {len(parts)}, expected 8"
                        logger.error(f"[ZMQ RECV] observation: {error_msg}")
                        raise ValueError(error_msg)

                    # Extract hardware observation parts (parts 2-7)
                    hw_obs_parts = parts[2:8]

                    # Deserialize hardware observation - let errors propagate
                    # Timestamp is already in the hw_obs metadata
                    hw_obs = KrabbyHardwareObservations.from_bytes(hw_obs_parts)

                    # Update latest buffer
                    self._latest_hw_obs = hw_obs
                    if self._debug_enabled:
                        logger.debug(f"[ZMQ RECV] observation: KrabbyHardwareObservations created successfully")
                    
                    return hw_obs
            except zmq.ZMQError:
                pass  # No message available (expected for NOBLOCK)
            # Let all other exceptions propagate (fail fast)
        
        # No new data received (timeout or no message available)
        return None

    def put_joint_command(self, cmd: "KrabbyDesiredJointPositions") -> None:
        """Put/send joint command to server.

        Args:
            cmd: KrabbyDesiredJointPositions containing joint positions

        Raises:
            RuntimeError: If client not initialized
            ValueError: If command is invalid
            zmq.ZMQError: If sending command fails
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        from hal.client.data_structures.hardware import KrabbyDesiredJointPositions
        if not isinstance(cmd, KrabbyDesiredJointPositions):
            raise ValueError(f"cmd must be KrabbyDesiredJointPositions, got {type(cmd)}")

        # Validate joint positions
        if cmd.joint_positions is None or len(cmd.joint_positions) == 0:
            raise ValueError("Cannot send empty joint positions")

        # Serialize and send as multipart message (metadata + array)
        # Use blocking send for backpressure - will block if PULL socket buffer is full
        command_parts = cmd.to_bytes()

        # Debug logging (conditional to avoid overhead when disabled)
        if self._debug_enabled:
            logger.debug(
                f"[ZMQ SEND] command: payload_size={sum(len(p) for p in command_parts)} bytes, "
                f"joint_positions_shape={cmd.joint_positions.shape}, "
                f"joint_positions_dtype={cmd.joint_positions.dtype}, "
                f"min={cmd.joint_positions.min():.3f}, max={cmd.joint_positions.max():.3f}, "
                f"timestamp_ns={cmd.timestamp_ns}"
            )

        # Blocking send - will block if PULL socket buffer is full (backpressure)
        # Let ZMQ errors propagate
        self.command_socket.send_multipart(command_parts)

        if self._debug_enabled:
            logger.debug("[ZMQ SEND] command: message sent successfully")

