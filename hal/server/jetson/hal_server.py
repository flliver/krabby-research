"""Jetson HAL server implementation."""

import logging
import time
from typing import Optional

import numpy as np
from scipy.ndimage import zoom

from hal.server import HalServerBase, HalServerConfig
from hal.client.data_structures.hardware import HardwareObservations, JointCommand
from hal.server.jetson.camera import ZedCamera, create_zed_camera

# Import model-specific constant here (HAL server knows about model requirements)
from compute.parkour.parkour_types import NUM_SCAN

logger = logging.getLogger(__name__)


class JetsonHalServer(HalServerBase):
    """HAL server for Jetson robot deployment.

    Integrates with ZED camera and real sensors to publish observations.
    Applies joint commands to real actuators.
    """

    def __init__(
        self,
        config: HalServerConfig,
        camera_resolution: tuple[int, int] = (640, 480),
        camera_fps: int = 30,
    ):
        """Initialize Jetson HAL server.

        Args:
            config: HAL server configuration
            camera_resolution: ZED camera resolution (width, height). Default (640, 480)
            camera_fps: ZED camera FPS. Default 30
        """
        super().__init__(config)
        self.camera_resolution = camera_resolution
        self.camera_fps = camera_fps
        self.zed_camera: Optional[ZedCamera] = None
        self.state_source = None  # IMU/encoders (placeholder, real implementation in future)
        self.actuator_sink = None  # Motors (placeholder, real implementation in future)
        # Track last commanded joint positions for state echo (placeholder)
        self._last_joint_positions: Optional[np.ndarray] = None
        self._action_dim = 12  # Default action dimension (12 DOF for Unitree Go2)

    def initialize_camera(self) -> None:
        """Initialize ZED camera.

        Creates ZED camera wrapper with error handling.
        """
        logger.info("Initializing ZED camera...")
        self.zed_camera = create_zed_camera(
            resolution=self.camera_resolution,
            fps=self.camera_fps,
            depth_mode="PERFORMANCE",
            depth_feature_dim=NUM_SCAN,  # Use model-specific constant (132 for parkour)
        )

        if self.zed_camera is None:
            logger.error("Failed to initialize ZED camera")
            raise RuntimeError("ZED camera initialization failed")
        elif not self.zed_camera.is_ready():
            logger.error("ZED camera initialized but not ready")
            raise RuntimeError("ZED camera is not ready")
        else:
            logger.info("ZED camera initialized successfully")

    def initialize_sensors(self) -> None:
        """Initialize state sensors (IMU/encoders).

        This is a placeholder. Real implementation would:
        - Initialize IMU
        - Initialize encoders
        - Configure sensor parameters
        """
        # Placeholder - actual implementation needs sensor drivers
        logger.info("Sensor initialization (placeholder)")

    def initialize_actuators(self) -> None:
        """Initialize actuators (motors).

        This is a placeholder. Real implementation would:
        - Initialize motor controllers
        - Configure motor parameters
        - Enable motors
        """
        # Placeholder - actual implementation needs motor drivers
        logger.info("Actuator initialization (placeholder)")

    def _build_depth_features(self) -> Optional[np.ndarray]:
        """Build depth features from ZED camera.

        Returns:
            Depth features as float32 array, or None if unavailable
        """
        if self.zed_camera is None:
            logger.warning("ZED camera not initialized")
            return None

        # Get depth features (includes capture, validation, and feature extraction)
        depth_features = self.zed_camera.get_depth_features()
        if depth_features is None:
            logger.warning("Failed to get depth features from ZED camera")
            return None

        # Return features array
        return depth_features

    def _build_state_vector(self) -> Optional[np.ndarray]:
        """Build state vector from sensors (placeholder implementation).

        This is a placeholder that:
        - Uses identity base pose (zero position, identity quaternion)
        - Uses zero velocities (dead-reckoning placeholder)
        - Echoes last commanded joint positions
        - Uses zero joint velocities

        Format: base_pos(3), base_quat(4), base_lin_vel(3), base_ang_vel(3),
                joint_pos(ACTION_DIM), joint_vel(ACTION_DIM)

        Returns:
            State vector as float32 array, or None if no joint positions available yet
        """
        # If we have real state source, use it
        if self.state_source is not None:
            # Real implementation would get state from IMU/encoders
            # For now, fall through to placeholder
            pass

        # Placeholder implementation: identity pose + echo joint positions
        # Base position: (0, 0, 0) - identity position
        base_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Base quaternion: (0, 0, 0, 1) - identity quaternion (no rotation)
        base_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # Base velocities: (0, 0, 0) - zero velocities (dead-reckoning placeholder)
        base_lin_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        base_ang_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Joint positions: echo last commanded targets
        if self._last_joint_positions is not None:
            joint_pos = self._last_joint_positions.astype(np.float32)
        else:
            # If no commands received yet, use zeros
            joint_pos = np.zeros(self._action_dim, dtype=np.float32)

        # Joint velocities: (0, 0, ...) - zero velocities (placeholder)
        joint_vel = np.zeros(self._action_dim, dtype=np.float32)

        # Concatenate into state vector
        state_vector = np.concatenate([
            base_pos,
            base_quat,
            base_lin_vel,
            base_ang_vel,
            joint_pos,
            joint_vel,
        ]).astype(np.float32)

        return state_vector

    def set_observation(self) -> None:
        """Set observation from real sensors as hardware observations.
        
        Constructs HardwareObservations from raw sensor data.
        """
        try:
            # Build state vector
            state_vector = self._build_state_vector()
            if state_vector is None:
                logger.warning("Failed to get state vector, cannot publish telemetry")
                return

            # Extract components from state vector
            # Format: base_pos(3), base_quat(4), base_lin_vel(3), base_ang_vel(3),
            #         joint_pos(ACTION_DIM), joint_vel(ACTION_DIM)
            joint_pos = state_vector[13:13+self._action_dim]
            
            # Pad or truncate joint positions to 18 DOF (Krabby has 18 joints)
            joint_positions = np.zeros(18, dtype=np.float32)
            num_joints = min(len(joint_pos), 18)
            joint_positions[:num_joints] = joint_pos[:num_joints]

            # Get camera data (placeholder - ZED camera provides depth features, not raw images)
            # For now, create dummy RGB and depth maps
            # TODO: Get actual RGB images and depth map from ZED camera
            camera_height, camera_width = self.camera_resolution[1], self.camera_resolution[0]  # Jetson variable resolution
            rgb_camera_1 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
            rgb_camera_2 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
            
            # Try to get depth map from ZED camera
            # If get_depth_map() is not available, create dummy depth map
            depth_map = np.zeros((camera_height, camera_width), dtype=np.float32)
            confidence_map = np.ones((camera_height, camera_width), dtype=np.float32)
            
            if self.zed_camera is not None:
                # Try to get raw depth map if available
                if hasattr(self.zed_camera, 'get_depth_map'):
                    depth_map_data = self.zed_camera.get_depth_map()
                    if depth_map_data is not None:
                        # Resize if needed
                        if depth_map_data.shape != (camera_height, camera_width):
                            zoom_factors = (camera_height / depth_map_data.shape[0], 
                                          camera_width / depth_map_data.shape[1])
                            depth_map = zoom(depth_map_data, zoom_factors, order=1).astype(np.float32)
                        else:
                            depth_map = depth_map_data.astype(np.float32)
                
                # Try to get RGB images if available
                if hasattr(self.zed_camera, 'get_rgb_images'):
                    rgb_images = self.zed_camera.get_rgb_images()
                    if rgb_images is not None and len(rgb_images) >= 2:
                        rgb_camera_1 = rgb_images[0]
                        rgb_camera_2 = rgb_images[1]

            # Create hardware observation
            hw_obs = HardwareObservations(
                joint_positions=joint_positions,
                rgb_camera_1=rgb_camera_1,
                rgb_camera_2=rgb_camera_2,
                depth_map=depth_map,
                confidence_map=confidence_map,
                camera_height=camera_height,
                camera_width=camera_width,
                timestamp_ns=time.time_ns(),
            )

            # Publish hardware observation via base-class publisher
            super().set_observation(hw_obs)

        except Exception as e:
            logger.error(f"Error publishing telemetry: {e}", exc_info=True)

    def apply_command(self) -> None:
        """Apply joint command from transport layer to actuators.
        
        **Synchronous method** that applies commands **immediately** (no queuing).
        Gets the latest command from the transport layer and applies it directly
        to the robot actuators. Does not perform any background work to keep the
        robot moving - the main loop must call this method regularly at the target
        control rate (typically 100 Hz).
        
        Currently a placeholder implementation that:
        - Stores the command for state echo
        - Logs the command with timestamp
        - Does not actually apply to motors (placeholder for future implementation)
        
        The robot continues moving based on the last applied command until the next
        command is received. If this method is not called regularly, the robot will
        stop moving after the last command's effect completes.
        
        Raises:
            RuntimeError: If no command received from transport layer (timeout after 10ms)
        """
        # Get command instance (includes timestamp and metadata)
        command_instance = self.get_joint_command(timeout_ms=10)
        if command_instance is None:
            raise RuntimeError("No command received from transport layer")

        # Extract joint positions array from command
        command = command_instance.joint_positions

        # Store command for state echo (placeholder: echo joint state from last commanded targets)
        self._last_joint_positions = command.copy()
        self._action_dim = len(command)

        # Log command with timestamp (as specified in requirements)
        timestamp_ns = time.time_ns()
        logger.info(
            f"[JOINT COMMAND] timestamp={timestamp_ns}, "
            f"joint_pos={command.tolist()}, "
            f"shape={command.shape}, "
            f"dtype={command.dtype}"
        )

        # Apply to actuators (placeholder, real implementation in future)
        # Example:
        # if self.actuator_sink is not None:
        #     self.actuator_sink.set_joint_positions(command)

    def close(self) -> None:
        """Close camera and all server resources."""
        # Close camera first
        if self.zed_camera is not None:
            self.zed_camera.close()
            self.zed_camera = None

        # Close server resources (sockets, context)
        super().close()

