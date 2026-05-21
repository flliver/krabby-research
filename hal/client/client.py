"""HAL client implementation using ZMQ."""

import logging
import time
from typing import Optional

import numpy as np
import zmq

# InferenceResponse is not used in HAL client
from hal.client.config import HalClientConfig
from hal.client.data_structures.hardware import HardwareObservations, JointCommand

logger = logging.getLogger(__name__)

# Topics for PUB/SUB channels
TOPIC_OBSERVATION = b"observation"  # Complete observation in training format

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
        self._latest_hw_obs: Optional[HardwareObservations] = None

        self._initialized = False
        self._debug_enabled = False

    def initialize(self) -> None:
        """Initialize ZMQ context and sockets."""
        if self._initialized:
            return

        if self.context is None:
            self.context = zmq.Context()
            self._context_owned = True

        # Create SUB socket for observation (single-part messages: topic prefix + blob)
        self.observation_socket = self.context.socket(zmq.SUB)
        self.observation_socket.setsockopt(zmq.RCVHWM, 1)  # Keep only one message
        self.observation_socket.setsockopt(zmq.CONFLATE, 1)  # When multiple arrive before recv, keep only the latest
        self.observation_socket.setsockopt(zmq.SUBSCRIBE, TOPIC_OBSERVATION)
        if not self.config.observation_endpoint:
            raise ValueError("observation_endpoint must be set in config")
        self.observation_socket.connect(self.config.observation_endpoint)

        if self.config.command_endpoint is not None:
            # Create PUSH socket for commands (PUSH/PULL pattern with backpressure)
            self.command_socket = self.context.socket(zmq.PUSH)
            self.command_socket.setsockopt(zmq.SNDHWM, 5)  # Default HWM of 5 for backpressure
            self.command_socket.connect(self.config.command_endpoint)
        else:
            self.command_socket = None

        self._initialized = True
        logger.info(
            "HAL client initialized: %s observation=%s, command=%s",
            self.__class__.__name__,
            self.config.observation_endpoint,
            self.config.command_endpoint if self.config.command_endpoint else "(observation-only)",
        )

    def close(self) -> None:
        """Close all sockets and context. Blocks until sockets finish."""
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
        logger.info("HAL client closed: %s", self.__class__.__name__)

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

    def poll(self, timeout_ms: int = 10) -> Optional[HardwareObservations]:
        """Poll for latest hardware observation messages (non-blocking).

        Updates latest buffers with newest messages. Old messages are
        automatically dropped due to buffer size=1 (latest-only semantics).

        Args:
            timeout_ms: Poll timeout in milliseconds (default 10ms)

        Returns:
            HardwareObservations if new data was received,
            None if timeout or no new data available.

        Raises:
            RuntimeError: If client not initialized
            ValueError: If message format is invalid (wrong number of parts)
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        # Poll observation socket (single-part frame: topic prefix + blob; CONFLATE keeps latest)
        has_message = self.observation_socket.poll(timeout_ms, zmq.POLLIN)
        if self._debug_enabled:
            wall_ns = time.time_ns()
            if has_message:
                logger.debug(
                    f"[HAL client] poll: message available, recv (single frame)... wall_ns={wall_ns}",
                )
            else:
                try:
                    events = self.observation_socket.getsockopt(zmq.EVENTS)
                    logger.debug(
                        f"[HAL client] poll: no message (timeout={timeout_ms} ms); EVENTS={events}, wall_ns={wall_ns}",
                    )
                except Exception as e:
                    logger.debug(f"[HAL client] poll: no message; could not get socket state: {e}")
        if has_message:
            frame = self.observation_socket.recv()
            if not frame.startswith(TOPIC_OBSERVATION):
                raise ValueError(
                    f"Observation frame must start with topic {TOPIC_OBSERVATION!r}, got {len(frame)} bytes"
                )
            payload = frame[len(TOPIC_OBSERVATION) :]
            if len(payload) < 4:
                raise ValueError(
                    f"Observation payload too short (need at least 4 bytes for metadata length), got {len(payload)}"
                )
            hw_obs = HardwareObservations.from_bytes(payload)
            self._latest_hw_obs = hw_obs
            if self._debug_enabled:
                logger.debug(
                    f"[HAL client] observation stored and returned, timestamp_ns={hw_obs.timestamp_ns}, wall_ns={time.time_ns()}",
                )
            return hw_obs

        # No new data received (timeout)
        return None

    def put_joint_command(self, cmd: "JointCommand") -> None:
        """Put/send joint command to server.

        Args:
            cmd: JointCommand containing joint positions

        Raises:
            RuntimeError: If client not initialized
            ValueError: If command is invalid
            zmq.ZMQError: If sending command fails
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        if self.command_socket is None:
            raise RuntimeError("Observation-only HAL client cannot send joint commands")

        if not isinstance(cmd, JointCommand):
            raise ValueError(f"cmd must be JointCommand, got {type(cmd)}")

        # Validate joint positions
        cmd_dict = cmd.to_positions_dict()
        if not cmd_dict:
            raise ValueError("Cannot send empty joint positions")

        # Single-part message (blob)
        command_blob = cmd.to_bytes()
        if self._debug_enabled:
            vals = list(cmd_dict.values())
            logger.debug(
                f"[ZMQ SEND] command: payload_size={len(command_blob)} bytes, joint_positions_len={len(vals)}, "
                f"min={min(vals):.3f}, max={max(vals):.3f}, timestamp_ns={cmd.timestamp_ns}",
            )
        self.command_socket.send(command_blob)

        if self._debug_enabled:
            logger.debug("[ZMQ SEND] command: message sent successfully")

