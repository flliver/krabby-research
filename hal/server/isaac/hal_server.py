"""IsaacSim HAL server implementation."""

import logging
import time

import numpy as np
import torch

from hal.server import HalServerBase, HalServerConfig
from hal.client.data_structures.hardware import KrabbyHardwareObservations

logger = logging.getLogger(__name__)


class IsaacSimHalServer(HalServerBase):
    """HAL server for IsaacSim environment.
    
    Extracts observations from IsaacSim environment and publishes via HAL.
    Applies joint commands received via HAL to the environment.
    """

    def __init__(self, config: HalServerConfig, env=None):
        """Initialize IsaacSim HAL server.
        
        Args:
            config: HAL server configuration
            env: IsaacSim environment. If provided, environment component
                references will be cached via _cache_references().
        
        Note:
            If env is provided, this will call _cache_references() to extract
            and cache references to scene, robot, sensors, and managers.
        """
        super().__init__(config)
        self.env = env
        self.scene = None
        self.robot = None
        self.observation_manager = None
        self.action_manager = None

        if env is not None:
            self._cache_references()

    def _cache_references(self) -> None:
        """Cache references to environment components for efficient access.
        
        This private method exists to keep __init__ clean and readable by extracting
        the complex initialization logic for environment component references into
        a separate method. The method handles:
        - Finding and caching scene, robot, and camera sensor references
        - Caching observation and action managers
        - Error handling and validation of required components
        
        This separation improves code maintainability since the caching logic is
        substantial (~60 lines) and includes error handling that would otherwise
        clutter __init__.
        
        **Important**: This method is only called from __init__ during server
        initialization. All initialization logic for environment references should
        remain here, not be mixed into __init__.
        
        Raises:
            RuntimeError: If robot entity cannot be found in scene
        """
        if self.env is None:
            return

        try:
            # Cache scene reference
            self.scene = self.env.scene

            # Cache robot reference (assuming robot is named "robot" in scene)
            # This matches the pattern used in observations.py: env.scene[cfg.params["asset_cfg"].name]
            if "robot" in self.scene:
                self.robot = self.scene["robot"]
            else:
                # Try to find robot by iterating scene entities
                for name, entity in self.scene.items():
                    if hasattr(entity, "data") and hasattr(entity.data, "joint_pos"):
                        self.robot = entity
                        logger.info(f"Found robot entity: {name}")
                        break

            # Cache camera sensors if available
            self.camera_sensors = {}
            if hasattr(self.scene, 'sensors'):
                # Try to find camera sensors
                for sensor_name, sensor in self.scene.sensors.items():
                    # Check if it's a camera-like sensor (RayCasterCamera, Camera, etc.)
                    if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
                        # Check for depth or RGB outputs
                        if 'distance_to_camera' in sensor.data.output or 'distance_to_image_plane' in sensor.data.output:
                            self.camera_sensors[sensor_name] = sensor
                            logger.info(f"Found camera sensor: {sensor_name}")
                # Also check scene entities for cameras
                for name, entity in self.scene.items():
                    if hasattr(entity, 'data') and hasattr(entity.data, 'output'):
                        if 'distance_to_camera' in entity.data.output or 'distance_to_image_plane' in entity.data.output:
                            self.camera_sensors[name] = entity
                            logger.info(f"Found camera entity: {name}")

            # Cache managers
            self.observation_manager = self.env.observation_manager
            self.action_manager = self.env.action_manager

            # Verify we have required references
            if self.robot is None:
                raise RuntimeError(
                    "Robot not found in scene. "
                    "IsaacSim HAL server requires a robot entity in the scene. "
                    f"Available entities: {list(self.scene.keys()) if self.scene else 'None'}"
                )
            if self.observation_manager is None:
                logger.warning("Observation manager not available")
            if self.action_manager is None:
                logger.warning("Action manager not available")

            logger.info("Cached environment references successfully")
        except Exception as e:
            logger.error(f"Could not cache all environment references: {e}", exc_info=True)

    def set_observation(self) -> None:
        """Set observation from IsaacSim environment as hardware observations.
        
        Extracts raw sensor data from environment and constructs KrabbyHardwareObservations.
        Extracts:
        - Joint positions from robot entity
        - Depth maps from camera sensors
        - RGB images if available from camera sensors or render products
        """
        if self.env is None:
            raise RuntimeError("No environment set, cannot set observation")

        if self.robot is None:
            raise RuntimeError("Robot not available, cannot set observation")

        # Extract joint positions from robot
        joint_positions = np.zeros(18, dtype=np.float32)
        if hasattr(self.robot, 'data') and hasattr(self.robot.data, 'joint_pos'):
            joint_pos = self.robot.data.joint_pos
            if isinstance(joint_pos, torch.Tensor):
                # Handle batched data (num_envs, num_joints) - take first environment
                if joint_pos.ndim == 2:
                    joint_pos = joint_pos[0]
                joint_pos = joint_pos.cpu().numpy()
            num_joints = min(len(joint_pos), 18)
            joint_positions[:num_joints] = joint_pos[:num_joints].astype(np.float32)

        # Extract camera data from sensors
        camera_height, camera_width = 480, 640
        rgb_camera_1 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        rgb_camera_2 = np.zeros((camera_height, camera_width, 3), dtype=np.uint8)
        depth_map = np.zeros((camera_height, camera_width), dtype=np.float32)
        confidence_map = np.ones((camera_height, camera_width), dtype=np.float32)

        # Try to get depth data from camera sensors
        camera_list = list(self.camera_sensors.values()) if self.camera_sensors else []
        
        if len(camera_list) > 0:
            # Get depth from first camera
            camera_0 = camera_list[0]
            if hasattr(camera_0, 'data') and hasattr(camera_0.data, 'output'):
                # Try different depth output formats
                depth_data = None
                if 'distance_to_camera' in camera_0.data.output:
                    depth_data = camera_0.data.output["distance_to_camera"]
                elif 'distance_to_image_plane' in camera_0.data.output:
                    depth_data = camera_0.data.output["distance_to_image_plane"]
                
                if depth_data is not None:
                    # Convert to numpy
                    if isinstance(depth_data, torch.Tensor):
                        # Handle batched data - take first environment
                        if depth_data.ndim > 2:
                            depth_data = depth_data[0]
                        # Remove channel dimension if present
                        if depth_data.ndim == 3 and depth_data.shape[-1] == 1:
                            depth_data = depth_data.squeeze(-1)
                        depth_np = depth_data.detach().cpu().numpy().astype(np.float32)
                        
                        # Resize if needed
                        if depth_np.shape != (camera_height, camera_width):
                            from scipy.ndimage import zoom
                            zoom_factors = (camera_height / depth_np.shape[0], 
                                          camera_width / depth_np.shape[1])
                            depth_map = zoom(depth_np, zoom_factors, order=1).astype(np.float32)
                        else:
                            depth_map = depth_np
            
            # Try to get RGB from second camera or render product
            if len(camera_list) > 1:
                camera_1 = camera_list[1]
                # Try to get RGB if available
                if hasattr(camera_1, 'data') and hasattr(camera_1.data, 'output'):
                    if 'rgb' in camera_1.data.output:
                        rgb_data = camera_1.data.output["rgb"]
                        if isinstance(rgb_data, torch.Tensor):
                            if rgb_data.ndim > 3:
                                rgb_data = rgb_data[0]
                            rgb_np = rgb_data.detach().cpu().numpy()
                            # Convert to uint8 if needed
                            if rgb_np.dtype != np.uint8:
                                rgb_np = (rgb_np * 255).astype(np.uint8)
                            if rgb_np.shape[:2] != (camera_height, camera_width):
                                from scipy.ndimage import zoom
                                zoom_factors = (camera_height / rgb_np.shape[0], 
                                              camera_width / rgb_np.shape[1], 1)
                                rgb_camera_2 = zoom(rgb_np, zoom_factors, order=1).astype(np.uint8)
                            else:
                                rgb_camera_2 = rgb_np

        # Try to get RGB from render product if available (for first camera)
        if hasattr(self.env, 'render') and self.env.render_mode == "rgb_array":
            try:
                rgb_data = self.env.render()
                if rgb_data is not None and rgb_data.size > 0:
                    # rgb_data is typically (H, W, 3) uint8
                    if rgb_data.shape[:2] != (camera_height, camera_width):
                        from scipy.ndimage import zoom
                        zoom_factors = (camera_height / rgb_data.shape[0], 
                                      camera_width / rgb_data.shape[1], 1)
                        rgb_camera_1 = zoom(rgb_data, zoom_factors, order=1).astype(np.uint8)
                    else:
                        rgb_camera_1 = rgb_data.astype(np.uint8)
            except Exception as e:
                logger.debug(f"Could not get RGB from render: {e}")

        # Create hardware observation
        hw_obs = KrabbyHardwareObservations(
            joint_positions=joint_positions,
            rgb_camera_1=rgb_camera_1,
            rgb_camera_2=rgb_camera_2,
            depth_map=depth_map,
            confidence_map=confidence_map,
            timestamp_ns=time.time_ns(),
        )

        # Publish hardware observation via base-class publisher
        super().set_observation(hw_obs)


    def apply_command(self) -> None:
        """Apply joint command from transport layer to IsaacSim environment.
        
        **Synchronous method** that applies commands **immediately** (no queuing).
        Gets the latest command from the transport layer and applies it directly
        to the action manager. Does not perform any background work to keep the
        robot moving - the main loop must call this method regularly at the target
        control rate (typically 100 Hz).
        
        The robot continues moving based on the last applied command until the next
        command is received. If this method is not called regularly, the robot will
        stop moving after the last command's effect completes.
        
        Raises:
            RuntimeError: If environment or action manager not available
            RuntimeError: If no command received from transport layer (timeout after 10ms)
        """
        if self.env is None:
            raise RuntimeError("No environment set, cannot apply command")

        if self.action_manager is None:
            raise RuntimeError("Action manager not available, cannot apply command")

        # Get command instance (includes timestamp and metadata)
        command = self.get_joint_command(timeout_ms=10)
        if command is None:
            raise RuntimeError("No command received from transport layer")

        # Extract joint positions array from command
        command_array = command.joint_positions

        # Convert NumPy array to tensor (zero-copy when array is C-contiguous float32)
        # The joint_positions array from KrabbyDesiredJointPositions is already
        # a zero-copy view of the bytes and is C-contiguous float32
        command_tensor = torch.from_numpy(command_array).to(device=self.env.device, dtype=torch.float32)

        # Add batch dimension if needed (action_manager expects (num_envs, action_dim))
        if command_tensor.ndim == 1:
            command_tensor = command_tensor.unsqueeze(0)  # Shape: (1, ACTION_DIM)

        # Process actions through action manager (same as training)
        # This handles any preprocessing, clipping, etc.
        # Note: process_action (singular) matches the API used in env.step()
        self.action_manager.process_action(command_tensor)

        # Apply actions to set joint position targets (same as training)
        # Note: apply_action (singular) matches the API used in env.step()
        self.action_manager.apply_action()

