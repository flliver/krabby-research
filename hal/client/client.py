"""HAL client implementation using ZMQ."""

import logging
import time
from typing import Optional

import numpy as np
import zmq

from hal.client.commands.types import InferenceResponse
from hal.client.config import HalClientConfig
from hal.client.observation.types import (
    NavigationCommand,
    OBS_DIM,
    ParkourModelIO,
    ParkourObservation,
)

logger = logging.getLogger(__name__)

# Topics for PUB/SUB channels
TOPIC_OBSERVATION = b"observation"  # Complete observation in training format

# Schema version
SCHEMA_VERSION = "1.0"


class HalClient:
    """HAL client for subscribing to observations and sending commands.

    Uses SUB socket for observations (complete observation in training format, HWM=1 for latest-only),
    and REQ socket for commands (request-response pattern).
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
        self._latest_observation: Optional[ParkourObservation] = None
        self._latest_nav_cmd: Optional[NavigationCommand] = None

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

        # Create REQ socket for commands
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.setsockopt(zmq.RCVTIMEO, int(self.config.timeout_s * 1000))
        self.command_socket.setsockopt(zmq.SNDTIMEO, int(self.config.timeout_s * 1000))
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

    def poll(self, timeout_ms: int = 10) -> Optional[np.ndarray]:
        """Poll for latest observation messages (non-blocking).

        Updates latest buffers with newest messages. Old messages are
        automatically dropped due to buffer size=1 (latest-only semantics).

        Args:
            timeout_ms: Poll timeout in milliseconds (default 10ms)

        Returns:
            Observation array (shape: (OBS_DIM,)) if new data was received,
            None if timeout or no new data available.

        Raises:
            RuntimeError: If client not initialized
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        # Poll observation socket (complete observation in training format)
        if self.observation_socket.poll(timeout_ms, zmq.POLLIN):
            try:
                parts = self.observation_socket.recv_multipart(zmq.NOBLOCK)
                if len(parts) >= 3:
                    topic, schema_version_bytes, payload = parts[0], parts[1], parts[2]
                    schema_version = schema_version_bytes.decode("utf-8")

                    # Debug logging (conditional to avoid overhead when disabled)
                    if self._debug_enabled:
                        logger.debug(
                            f"[ZMQ RECV] observation: topic={topic.decode('utf-8')}, "
                            f"schema={schema_version}, "
                            f"payload_size={len(payload)} bytes"
                        )

                    # Validate schema version
                    if schema_version != SCHEMA_VERSION:
                        if self._debug_enabled:
                            logger.debug(
                                f"[ZMQ RECV] observation: unsupported schema version: {schema_version}, "
                                f"expected {SCHEMA_VERSION}"
                            )
                        logger.warning(f"Unsupported schema version: {schema_version}, expected {SCHEMA_VERSION}")
                        return None

                    # Runtime type validation: Validate payload size
                    expected_size = OBS_DIM * 4  # float32 = 4 bytes
                    if len(payload) != expected_size:
                        logger.error(
                            f"[ZMQ RECV] observation: Invalid payload size: {len(payload)} bytes, "
                            f"expected {expected_size} bytes (OBS_DIM={OBS_DIM} * 4 bytes)"
                        )
                        return None

                    # Deserialize payload - observation in training format
                    observation_array = np.frombuffer(payload, dtype=np.float32)

                    # Runtime type validation: Validate dtype
                    if observation_array.dtype != np.float32:
                        logger.error(
                            f"[ZMQ RECV] observation: Invalid dtype: {observation_array.dtype}, expected float32"
                        )
                        return None

                    # Runtime type validation: Validate shape matches training format
                    if observation_array.shape != (OBS_DIM,):
                        logger.error(
                            f"[ZMQ RECV] observation: Invalid shape: {observation_array.shape}, expected ({OBS_DIM},)"
                        )
                        return None

                    # Runtime type validation: Validate values (check for NaN/Inf)
                    if not np.isfinite(observation_array).all():
                        nan_count = np.isnan(observation_array).sum()
                        inf_count = np.isinf(observation_array).sum()
                        logger.error(
                            f"[ZMQ RECV] observation: Invalid values: {nan_count} NaN, {inf_count} Inf"
                        )
                        return None

                    # Debug logging after deserialization
                    if self._debug_enabled:
                        logger.debug(
                            f"[ZMQ RECV] observation: shape={observation_array.shape}, "
                            f"dtype={observation_array.dtype}, "
                            f"min={observation_array.min():.3f}, max={observation_array.max():.3f}"
                        )

                    # Extract timestamp if provided (4th part), otherwise use current time
                    timestamp_ns = time.time_ns()
                    if len(parts) >= 4:
                        try:
                            timestamp_ns = int.from_bytes(parts[3], byteorder="big", signed=False)
                        except (ValueError, IndexError):
                            pass  # Use current time if timestamp parsing fails

                    # Create ParkourObservation (zero-copy view of the buffer)
                    # Note: np.frombuffer creates a view, not a copy
                    observation_copy = observation_array.copy()  # Copy here because buffer is owned by ZMQ
                    self._latest_observation = ParkourObservation(
                        timestamp_ns=timestamp_ns,
                        schema_version=schema_version,
                        observation=observation_copy,
                    )
                    if self._debug_enabled:
                        logger.debug(f"[ZMQ RECV] observation: ParkourObservation created successfully")
                    
                    # Return the observation array
                    return observation_copy
            except zmq.ZMQError:
                pass  # No message available
            except Exception as e:
                if self._debug_enabled:
                    logger.debug(f"[ZMQ ERROR] observation: {e}")
                logger.error(f"Error processing observation message: {e}")
        
        # No new data received (timeout or no message available)
        return None

    def build_model_io(self, max_age_ns: int = 10_000_000) -> Optional[ParkourModelIO]:
        """Build ParkourModelIO from latest observation and navigation command.
        
        Combines the latest observation (from poll()) and navigation command into
        a ParkourModelIO structure for model inference.
        
        Checks that all required data has valid timestamps and is synchronized.
        
        Args:
            max_age_ns: Maximum age difference in nanoseconds (default 10ms)
        
        Returns:
            ParkourModelIO if all components available and synchronized, None otherwise
        """
        if self._latest_observation is None or self._latest_nav_cmd is None:
            return None

        # Check timestamps are recent
        now_ns = time.time_ns()
        if (now_ns - self._latest_observation.timestamp_ns) > max_age_ns:
            return None
        
        # Check synchronization - allow nav_cmd to be older than observation
        observation_age_ns = now_ns - self._latest_observation.timestamp_ns
        if observation_age_ns > max_age_ns:
            return None

        # Check schema versions are compatible
        schema_versions = {
            self._latest_observation.schema_version,
            self._latest_nav_cmd.schema_version,
        }
        if len(schema_versions) > 1:
            logger.warning(f"Incompatible schema versions: {schema_versions}")
            return None

        return ParkourModelIO(
            timestamp_ns=now_ns,
            schema_version=self._latest_observation.schema_version,
            nav_cmd=self._latest_nav_cmd,
            observation=self._latest_observation,
        )

    def signal_ready(self, timeout_ms: int = 1000) -> bool:
        """Signal to server that client is ready (handshake for PUB/SUB connection).
        
        This ensures the PUB/SUB connection is established before the server starts publishing.
        Should be called after initialize() to establish the connection.
        
        Args:
            timeout_ms: Maximum time to wait for server acknowledgement (default 1000ms)
            
        Returns:
            True if server acknowledged, False if timeout or error
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")
        
        try:
            # Send ready signal (use blocking send for handshake)
            # REQ socket requires blocking send in REQ/REP pattern
            self.command_socket.send(b"ready")
            
            # Wait for acknowledgement
            if self.command_socket.poll(timeout_ms, zmq.POLLIN):
                response = self.command_socket.recv()
                if response == b"ready_ack":
                    logger.info("Server acknowledged handshake - PUB/SUB connection established")
                    return True
                else:
                    logger.warning(f"Unexpected handshake response: {response}")
                    return False
            else:
                logger.warning(f"Handshake timeout after {timeout_ms}ms")
                return False
        except zmq.ZMQError as e:
            logger.error(f"Error during handshake: {e}")
            return False

    def set_navigation_command(self, nav: NavigationCommand) -> None:
        """Set navigation command.

        Args:
            nav: Navigation command with timestamp
        """
        self._latest_nav_cmd = nav

    def put_joint_command(self, cmd: InferenceResponse) -> bool:
        """Put/send joint command to server.

        Uses action array directly from inference response (zero-copy).

        Args:
            cmd: Inference response containing action array

        Returns:
            True if command sent successfully, False otherwise

        Raises:
            RuntimeError: If client not initialized
            ValueError: If command is invalid
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        if not cmd.success or cmd.action is None:
            logger.error("Cannot send failed inference response")
            return False

        # Get action tensor directly (zero-copy)
        action_tensor = cmd.get_action()

        # Validate action dimension if config provides it
        if hasattr(self.config, 'action_dim') and self.config.action_dim is not None:
            try:
                cmd.validate_action_dim(self.config.action_dim)
            except ValueError as e:
                logger.error(f"Invalid action dimension: {e}")
                return False

        # Convert to numpy for ZMQ serialization (only when needed)
        # This creates a copy if on GPU, view if on CPU
        action_numpy = cmd.get_action_numpy()

        # Serialize and send (tobytes() creates a copy, but this is necessary for ZMQ)
        try:
            command_bytes = action_numpy.tobytes()

            # Debug logging (conditional to avoid overhead when disabled)
            if self._debug_enabled:
                logger.debug(
                    f"[ZMQ SEND] command: payload_size={len(command_bytes)} bytes, "
                    f"action_shape={action_numpy.shape}, "
                    f"action_dtype={action_numpy.dtype}, "
                    f"min={action_numpy.min():.3f}, max={action_numpy.max():.3f}"
                )

            self.command_socket.send(command_bytes, zmq.NOBLOCK)

            if self._debug_enabled:
                logger.debug("[ZMQ SEND] command: message sent, waiting for acknowledgement")

            # Wait for acknowledgement
            if self.command_socket.poll(int(self.config.timeout_s * 1000), zmq.POLLIN):
                response = self.command_socket.recv()
                if self._debug_enabled:
                    logger.debug(f"[ZMQ RECV] command: acknowledgement={response.decode('utf-8', errors='ignore')}")

                if response == b"ok":
                    if self._debug_enabled:
                        logger.debug("[ZMQ SEND] command: command accepted")
                    return True
                else:
                    if self._debug_enabled:
                        logger.debug(f"[ZMQ RECV] command: command rejected: {response.decode('utf-8', errors='ignore')}")
                    logger.error(f"Command rejected: {response.decode('utf-8', errors='ignore')}")
                    return False
            else:
                if self._debug_enabled:
                    logger.debug("[ZMQ RECV] command: timeout waiting for acknowledgement")
                logger.error("Command timeout - no acknowledgement received")
                return False

        except zmq.ZMQError as e:
            if self._debug_enabled:
                logger.debug(f"[ZMQ ERROR] command: {e}")
            logger.error(f"Error sending command: {e}")
            return False

