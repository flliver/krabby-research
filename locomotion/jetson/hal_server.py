"""Jetson HAL server implementation."""

import logging
import time
from typing import Optional

import numpy as np

from hal.server.server import HalServerBase
from hal.server.config import HalServerConfig
from hal.observation.types import (
    NUM_PROP,
    NUM_SCAN,
    NUM_PRIV_EXPLICIT,
    NUM_PRIV_LATENT,
    HISTORY_DIM,
    OBS_DIM,
)
from locomotion.jetson.camera import ZedCamera, create_zed_camera

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
        # depth_feature_dim is now fixed to NUM_SCAN (132) to match training format
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
        """Set observation from real sensors in training format.
        
        Builds complete observation array matching training format:
        [num_prop(53), num_scan(132), num_priv_explicit(9), num_priv_latent(29), history(530)]
        """
        try:
            # Build depth features (scan features)
            scan_features = self._build_depth_features()
            if scan_features is None:
                logger.warning("Failed to get scan features, cannot publish telemetry")
                return

            # Build state vector
            state_vector = self._build_state_vector()
            if state_vector is None:
                logger.warning("Failed to get state vector, cannot publish telemetry")
                return

            # Extract components from state vector
            # Format: base_pos(3), base_quat(4), base_lin_vel(3), base_ang_vel(3),
            #         joint_pos(ACTION_DIM), joint_vel(ACTION_DIM)
            base_pos = state_vector[0:3]
            base_quat = state_vector[3:7]
            base_lin_vel = state_vector[7:10]
            base_ang_vel = state_vector[10:13]
            joint_pos = state_vector[13:13+self._action_dim]
            joint_vel = state_vector[13+self._action_dim:13+2*self._action_dim]

            # Build proprioceptive features (53 features)
            # Training format from observations.py:
            # root_ang_vel_b * 0.25 (3)
            # imu_obs (roll, pitch) (2)
            # delta_yaw (1) - placeholder 0 for now
            # delta_yaw (1)
            # delta_next_yaw (1) - placeholder 0 for now
            # commands (vx, vy) (2) - placeholder 0 for now
            # commands (vx) (1) - placeholder 0 for now
            # env_idx_tensor (1) - placeholder 0 for now
            # invert_env_idx_tensor (1) - placeholder 1 for now
            # joint_pos - default_joint_pos (ACTION_DIM)
            # joint_vel * 0.05 (ACTION_DIM)
            # action_history_buf[:, -1] (ACTION_DIM) - placeholder 0 for now
            # contact_fill (4) - placeholder 0 for now
            # Total: 3 + 2 + 1 + 1 + 1 + 2 + 1 + 1 + 1 + ACTION_DIM + ACTION_DIM + ACTION_DIM + 4 = 19 + 3*ACTION_DIM
            # For ACTION_DIM=12: 19 + 36 = 55, but training shows 53, so some components may be different
            
            # Simplified proprioceptive features (placeholder - should match training exactly)
            # For now, use a minimal set that matches the structure
            proprioceptive = np.zeros(NUM_PROP, dtype=np.float32)
            proprioceptive[0:3] = base_ang_vel * 0.25  # root_ang_vel_b * 0.25
            # Roll, pitch from quaternion (simplified - should use euler_xyz_from_quat)
            # For now, use placeholder
            proprioceptive[3:5] = 0.0  # imu_obs (roll, pitch) - placeholder
            proprioceptive[5:6] = 0.0  # delta_yaw placeholder
            proprioceptive[6:7] = 0.0  # delta_yaw placeholder
            proprioceptive[7:8] = 0.0  # delta_next_yaw placeholder
            proprioceptive[8:10] = 0.0  # commands placeholder
            proprioceptive[10:11] = 0.0  # commands placeholder
            proprioceptive[11:12] = 0.0  # env_idx_tensor placeholder
            proprioceptive[12:13] = 1.0  # invert_env_idx_tensor placeholder (flat terrain)
            proprioceptive[13:13+self._action_dim] = joint_pos  # joint_pos - default_joint_pos (assuming default is 0)
            proprioceptive[13+self._action_dim:13+2*self._action_dim] = joint_vel * 0.05  # joint_vel * 0.05
            # Remaining features: action_history, contact_fill
            # Pad to NUM_PROP (53)
            if len(proprioceptive) < NUM_PROP:
                # Should not happen, but handle gracefully
                padding = np.zeros(NUM_PROP - len(proprioceptive), dtype=np.float32)
                proprioceptive = np.concatenate([proprioceptive, padding])
            elif len(proprioceptive) > NUM_PROP:
                proprioceptive = proprioceptive[:NUM_PROP]

            # Privileged explicit features (9 features)
            # Training format: base_lin_vel * 2.0 (3), zeros (6)
            priv_explicit = np.zeros(NUM_PRIV_EXPLICIT, dtype=np.float32)
            priv_explicit[0:3] = base_lin_vel * 2.0
            # Remaining 6 are zeros (as in training)

            # Privileged latent features (29 features)
            # Training format: body_mass (1), body_com (3), friction (1), joint_stiffness (12), joint_damping (12)
            # Total: 1 + 3 + 1 + 12 + 12 = 29
            priv_latent = np.zeros(NUM_PRIV_LATENT, dtype=np.float32)
            # Placeholder values (would need real sensor data)
            priv_latent[0] = 1.0  # body_mass placeholder
            priv_latent[1:4] = 0.0  # body_com placeholder
            priv_latent[4] = 0.5  # friction placeholder
            priv_latent[5:17] = 0.0  # joint_stiffness placeholder (normalized)
            priv_latent[17:29] = 0.0  # joint_damping placeholder (normalized)

            # History features (530 features = 10 * 53)
            # Placeholder: zeros for now (would need to maintain history buffer)
            history = np.zeros(HISTORY_DIM, dtype=np.float32)

            # Build complete observation array in training format
            observation = np.concatenate([
                proprioceptive,  # 53
                scan_features,  # 132
                priv_explicit,  # 9
                priv_latent,  # 29
                history,  # 530
            ], dtype=np.float32)

            # Validate shape
            if observation.shape != (OBS_DIM,):
                logger.error(
                    f"Observation shape {observation.shape} != expected ({OBS_DIM},). "
                    f"Components: proprioceptive={len(proprioceptive)}, scan={len(scan_features)}, "
                    f"priv_explicit={len(priv_explicit)}, priv_latent={len(priv_latent)}, history={len(history)}"
                )
                return

            # Publish complete observation in training format via base-class publisher
            super().set_observation(observation)

        except Exception as e:
            logger.error(f"Error publishing telemetry: {e}", exc_info=True)

    def move(self, joint_positions: Optional[np.ndarray] = None) -> bool:
        """Move robot joints to specified positions.
        
        Renamed from apply_joint_command() to use verb-free naming.
        Gets command from transport layer and applies to actuators.
        
        This is a placeholder that:
        - Stores the command for state echo
        - Logs the command with timestamp
        - Does not actually apply to motors (placeholder for future implementation)
        
        Args:
            joint_positions: Optional joint positions array (shape: (ACTION_DIM,)).
                If None, gets command from transport layer.
        
        Returns:
            True if command applied successfully, False otherwise
        """
        import time

        try:
            # Get command from transport if not provided
            if joint_positions is None:
                command = self.get_joint_command(timeout_ms=10)
                if command is None:
                    return False
            else:
                command = joint_positions

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
            #     return True

            # Command is logged and stored for state echo
            return True

        except Exception as e:
            logger.error(f"Error applying joint command: {e}")
            return False

    def close(self) -> None:
        """Close camera and all server resources."""
        # Close camera first
        if self.zed_camera is not None:
            self.zed_camera.close()
            self.zed_camera = None

        # Close server resources (sockets, context)
        super().close()

