"""HAL server base class using ZMQ for communication."""

import logging
from typing import Optional

import numpy as np
import zmq

from hal.server.config import HalServerConfig  # Internal import - config is in same package
from hal.client.observation.types import OBS_DIM

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

        # Create REP socket for commands
        self.command_socket = self.context.socket(zmq.REP)
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

    def set_observation(self, observation: np.ndarray, timestamp_ns: Optional[int] = None) -> None:
        """Set/publish observation to clients.

        Sends topic-prefixed multipart message: [topic, schema_version, payload, timestamp_bytes]

        Observation format: [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]

        Args:
            observation: Complete observation as float32 array of shape (OBS_DIM,)
            timestamp_ns: Optional timestamp in nanoseconds (defaults to current time)

        Raises:
            ValueError: If observation is invalid
            RuntimeError: If server not initialized
        """
        if not self._initialized:
            raise RuntimeError("Server not initialized. Call initialize() first.")

        # Runtime type validation: Validate input type
        if not isinstance(observation, np.ndarray):
            raise ValueError(f"observation must be numpy array, got {type(observation)}")
        
        # Runtime type validation: Validate dtype
        if observation.dtype != np.float32:
            observation = observation.astype(np.float32, copy=False)
        
        # Runtime type validation: Validate shape
        if observation.shape != (OBS_DIM,):
            raise ValueError(f"observation shape {observation.shape} != expected ({OBS_DIM},)")
        
        # Runtime type validation: Validate values (check for NaN/Inf)
        if not np.isfinite(observation).all():
            nan_count = np.isnan(observation).sum()
            inf_count = np.isinf(observation).sum()
            raise ValueError(f"observation contains invalid values: {nan_count} NaN, {inf_count} Inf")

        # Get timestamp
        if timestamp_ns is None:
            import time
            timestamp_ns = time.time_ns()

        # Serialize message
        topic = TOPIC_OBSERVATION
        schema_version = SCHEMA_VERSION.encode("utf-8")
        payload = observation.tobytes()
        timestamp_bytes = timestamp_ns.to_bytes(8, byteorder="big", signed=False)

        # Debug logging (conditional to avoid overhead when disabled)
        if self._debug_enabled:
            logger.debug(
                f"[ZMQ SEND] observation: topic={topic.decode('utf-8')}, "
                f"schema={schema_version.decode('utf-8')}, "
                f"payload_size={len(payload)} bytes, "
                f"observation_shape={observation.shape}, "
                f"observation_dtype={observation.dtype}, "
                f"timestamp_ns={timestamp_ns}"
            )

        # Send multipart message
        try:
            self.observation_socket.send_multipart(
                [topic, schema_version, payload, timestamp_bytes], zmq.NOBLOCK
            )
            if self._debug_enabled:
                logger.debug(f"[ZMQ SEND] observation: message sent successfully")
        except zmq.Again:
            if self._debug_enabled:
                logger.debug("[ZMQ SEND] observation: buffer full (HWM reached), message dropped")
            logger.warning("Observation socket buffer full (HWM reached), message dropped")

    def get_joint_command(self, timeout_ms: int = 100) -> Optional[np.ndarray]:
        """Get latest joint command from clients.

        Uses non-blocking poll to check for commands. If command received,
        validates payload and sends acknowledgement.

        Runtime validation includes:
        - Payload size validation (must be multiple of 4 bytes for float32)
        - Dtype validation (must be float32)
        - Shape validation (must be 1D array)
        - Value validation (no NaN or Inf)

        Args:
            timeout_ms: Poll timeout in milliseconds (default 100ms)

        Returns:
            Joint command as float32 array, or None if no command received or validation failed

        Raises:
            RuntimeError: If server not initialized
        """
        if not self._initialized:
            raise RuntimeError("Server not initialized. Call initialize() first.")

        # Poll for incoming command
        if self.command_socket.poll(timeout_ms, zmq.POLLIN):
            try:
                # Receive command
                command_bytes = self.command_socket.recv(zmq.NOBLOCK)

                # Debug logging (conditional to avoid overhead when disabled)
                if self._debug_enabled:
                    logger.debug(
                        f"[ZMQ RECV] command: received {len(command_bytes)} bytes"
                    )

                # Runtime type validation: Validate payload size
                if len(command_bytes) % 4 != 0:
                    error_msg = f"Invalid command payload size: {len(command_bytes)} bytes (must be multiple of 4 for float32)"
                    logger.error(f"[ZMQ RECV] command: {error_msg}")
                    self.command_socket.send(b"error: invalid payload size")
                    return None

                # Deserialize to numpy array
                command = np.frombuffer(command_bytes, dtype=np.float32)

                # Runtime type validation: Validate dtype
                if command.dtype != np.float32:
                    error_msg = f"Invalid command dtype: {command.dtype} (expected float32)"
                    logger.error(f"[ZMQ RECV] command: {error_msg}")
                    self.command_socket.send(b"error: invalid dtype")
                    return None

                # Runtime type validation: Validate shape (must be 1D)
                if command.ndim != 1:
                    error_msg = f"Invalid command shape: {command.shape} (expected 1D array)"
                    logger.error(f"[ZMQ RECV] command: {error_msg}")
                    self.command_socket.send(b"error: invalid shape")
                    return None

                # Runtime type validation: Validate values (check for NaN/Inf)
                if not np.isfinite(command).all():
                    nan_count = np.isnan(command).sum()
                    inf_count = np.isinf(command).sum()
                    error_msg = f"Invalid command values: {nan_count} NaN, {inf_count} Inf"
                    logger.error(f"[ZMQ RECV] command: {error_msg}")
                    self.command_socket.send(b"error: invalid values")
                    return None

                # Debug logging after deserialization
                if self._debug_enabled:
                    logger.debug(
                        f"[ZMQ RECV] command: shape={command.shape}, "
                        f"dtype={command.dtype}, "
                        f"min={command.min():.3f}, max={command.max():.3f}"
                    )

                # Send acknowledgement
                if self._debug_enabled:
                    logger.debug("[ZMQ SEND] command: sending ok acknowledgement")
                self.command_socket.send(b"ok")
                return command

            except zmq.ZMQError as e:
                logger.error(f"Error receiving command: {e}")
                if self._debug_enabled:
                    logger.debug(f"[ZMQ ERROR] command: {e}")
                self.command_socket.send(b"error: receive failed")
                return None
            except Exception as e:
                logger.error(f"Unexpected error processing command: {e}")
                if self._debug_enabled:
                    logger.debug(f"[ZMQ ERROR] command: {e}")
                self.command_socket.send(b"error: processing failed")
                return None

        return None

    def recv_joint_command_bytes(self, timeout_ms: int = 100) -> Optional[bytes]:
        """Receive joint command as raw bytes with runtime type validation.
        
        This method provides zero-copy access to the command bytes for direct
        tensor creation (e.g., using torch.frombuffer()). The bytes are validated
        but not deserialized to NumPy, avoiding the read-only buffer issue.
        
        Runtime validation includes:
        - Payload size validation (must be multiple of 4 bytes for float32)
        - Dtype validation (must be float32)
        - Shape validation (must be 1D array)
        - Value validation (no NaN or Inf)
        
        Args:
            timeout_ms: Poll timeout in milliseconds (default 100ms)
            
        Returns:
            Validated command bytes, or None if no command received or validation failed
            
        Raises:
            RuntimeError: If server not initialized
        """
        if not self._initialized:
            raise RuntimeError("Server not initialized. Call initialize() first.")

        # Poll for incoming command
        if self.command_socket.poll(timeout_ms, zmq.POLLIN):
            try:
                # Receive command
                command_bytes = self.command_socket.recv(zmq.NOBLOCK)

                # Debug logging (conditional to avoid overhead when disabled)
                if self._debug_enabled:
                    logger.debug(
                        f"[ZMQ RECV] command: received {len(command_bytes)} bytes"
                    )

                # Runtime type validation: Validate payload size
                if len(command_bytes) % 4 != 0:
                    error_msg = f"Invalid command payload size: {len(command_bytes)} bytes (must be multiple of 4 for float32)"
                    logger.error(f"[ZMQ RECV] command: {error_msg}")
                    self.command_socket.send(b"error: invalid payload size")
                    return None

                # Deserialize to numpy array for validation only
                command = np.frombuffer(command_bytes, dtype=np.float32)

                # Runtime type validation: Validate dtype
                if command.dtype != np.float32:
                    error_msg = f"Invalid command dtype: {command.dtype} (expected float32)"
                    logger.error(f"[ZMQ RECV] command: {error_msg}")
                    self.command_socket.send(b"error: invalid dtype")
                    return None

                # Runtime type validation: Validate shape (must be 1D)
                if command.ndim != 1:
                    error_msg = f"Invalid command shape: {command.shape} (expected 1D array)"
                    logger.error(f"[ZMQ RECV] command: {error_msg}")
                    self.command_socket.send(b"error: invalid shape")
                    return None

                # Runtime type validation: Validate values (check for NaN/Inf)
                if not np.isfinite(command).all():
                    nan_count = np.isnan(command).sum()
                    inf_count = np.isinf(command).sum()
                    error_msg = f"Invalid command values: {nan_count} NaN, {inf_count} Inf"
                    logger.error(f"[ZMQ RECV] command: {error_msg}")
                    self.command_socket.send(b"error: invalid values")
                    return None

                # Debug logging after validation
                if self._debug_enabled:
                    logger.debug(
                        f"[ZMQ RECV] command: shape={command.shape}, "
                        f"dtype={command.dtype}, "
                        f"min={command.min():.3f}, max={command.max():.3f}"
                    )

                # Send acknowledgement
                if self._debug_enabled:
                    logger.debug("[ZMQ SEND] command: sending ok acknowledgement")
                self.command_socket.send(b"ok")
                return command_bytes

            except zmq.ZMQError as e:
                logger.error(f"Error receiving command: {e}")
                if self._debug_enabled:
                    logger.debug(f"[ZMQ ERROR] command: {e}")
                self.command_socket.send(b"error: receive failed")
                return None
            except Exception as e:
                logger.error(f"Unexpected error processing command: {e}")
                if self._debug_enabled:
                    logger.debug(f"[ZMQ ERROR] command: {e}")
                self.command_socket.send(b"error: processing failed")
                return None

        return None

