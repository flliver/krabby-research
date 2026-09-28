"""Mapper from hardware observations to Parkour model format.

This mapper converts raw hardware sensor data into the format expected by
the Parkour policy model. It uses zero-copy operations where possible to
minimize data copying. Layout and sizes come from robot + model definitions
(ObservationDimensions).
"""

import logging
from typing import Optional, Type

import numpy as np
import torch
from compute.parkour.model_definition import ObservationDimensions
from compute.parkour.utils.math import euler_xyz_from_quat, wrap_to_pi

from hal.client.data_structures.hardware import HardwareObservations
from hal.client.observation.types import NavigationCommand
from compute.parkour.parkour_types import ParkourObservation, TeacherObservation

logger = logging.getLogger(__name__)


class HWObservationsToParkourMapper:
    """Maps hardware observations to Parkour model format.

    Uses observation_dimensions (from robot + model definitions) for all
    array sizes and layout. Zero-copy where possible.
    """

    def __init__(self, observation_dimensions: ObservationDimensions):
        """Initialize the mapper with observation layout from definitions.

        Args:
            observation_dimensions: Layout from model_definition.get_observation_dimensions(robot_definition)
        """
        d = observation_dimensions
        self._dims = observation_dimensions
        self._history_buffer = np.zeros((d.num_hist, d.num_prop), dtype=np.float32)
        self._episode_step = 0
        self._previous_action = np.zeros(d.observation_joint_count, dtype=np.float32)
    
    def set_previous_action(
        self, action: dict[str, float], joint_names: tuple[str, ...]
    ) -> None:
        """Set the previous action for history tracking.

        Args:
            action: Previous joint command by name.
            joint_names: Order of joints (used to fill _previous_action).
        """
        n = min(self._dims.observation_joint_count, len(joint_names))
        for i in range(n):
            self._previous_action[i] = action.get(joint_names[i], 0.0)
    
    def map(self, hw_obs: HardwareObservations, nav_cmd: Optional[NavigationCommand] = None, observation_type: Type[ParkourObservation] = TeacherObservation) -> ParkourObservation:
        """Map hardware observations to model format.
        
        Args:
            hw_obs: Hardware observation data
            nav_cmd: Optional navigation command (vx, vy, yaw_rate) to include in observation
            observation_type: Type of observation to create (default: TeacherObservation)
            
        Returns:
            ParkourObservation in model format
            
        Raises:
            ValueError: If hardware observation is invalid
        """
        # Extract features from hardware data
        proprioceptive = self._extract_proprioceptive(hw_obs, nav_cmd)
        scan = self._extract_scan_features(hw_obs)
        priv_explicit = self._extract_priv_explicit(hw_obs)
        priv_latent = self._extract_priv_latent(hw_obs)
        history = self._extract_history(proprioceptive)
        
        if self._dims.num_vision == 0:
            vision: list = []
        else:
            # To implement vision on hardware we would need:
            # - Run the same vision encoder used at training on hw_obs (camera_rgb).
            # - Match training preprocessing (resize, normalization, etc.).
            # - Output one feature vector per camera with sizes matching observation_dimensions.vision_dims.
            # - Integrate the encoder (and any GPU usage) into the inference path.
            raise NotImplementedError(
                "vision features (num_vision > 0) not yet supported in HWObservationsToParkourMapper"
            )
        return observation_type.from_parts(
            self._dims,
            proprioceptive=proprioceptive,
            scan=scan,
            priv_explicit=priv_explicit,
            priv_latent=priv_latent,
            history=history,
            timestamp_ns=hw_obs.timestamp_ns,
            vision=vision,
        )
    
    def _extract_proprioceptive(self, hw_obs: HardwareObservations, nav_cmd: Optional[NavigationCommand] = None) -> np.ndarray:
        """Extract proprioceptive features from hardware observations.
        
        Matches training format from ExtremeParkourObservations:
        - [0:3]   base angular velocity (body frame) * 0.25
        - [3:5]   IMU roll, pitch (wrapped to [-pi, pi])
        - [5]     zero (placeholder)
        - [6]     delta_yaw (target yaw - current yaw)
        - [7]     delta_next_yaw (next target yaw - current yaw)
        - [8]     zero (placeholder for vy command)
        - [9]     vx command (forward velocity)
        - [10]    terrain type flag (1 if not flat, 0 if flat)
        - [11]    flat terrain flag (1 if flat, 0 if not flat)
        - [12:24] joint positions (relative to default) - 12 joints
        - [24:36] joint velocities * 0.05 - 12 joints
        - [36:48] previous action (last joint command) - 12 joints
        - [48:52] contact forces (4 feet, normalized to [-0.5, 0.5])
        
        Args:
            hw_obs: Hardware observations
            nav_cmd: Optional navigation command to include in proprioceptive features
            
        Returns:
            Proprioceptive features array of shape (num_prop,)
        """
        num_prop = self._dims.num_prop
        num_joints = self._dims.observation_joint_count
        proprioceptive = np.zeros(num_prop, dtype=np.float32)
        
        # [0:3] Base angular velocity (body frame) * 0.25
        proprioceptive[0:3] = hw_obs.base_ang_vel_b * 0.25
        
        # [3:5] IMU roll, pitch (derived from base quaternion)
        # HAL ``base_quat_w`` is (x, y, z, w); ``euler_xyz_from_quat`` expects (w, x, y, z).
        q = hw_obs.base_quat_w
        quat_wxyz = torch.tensor(
            [[float(q[3]), float(q[0]), float(q[1]), float(q[2])]],
            dtype=torch.float32,
        )
        roll, pitch, _ = euler_xyz_from_quat(quat_wxyz)
        # Wrap to [-pi, pi] (matching environment)
        roll_wrapped = wrap_to_pi(roll)
        pitch_wrapped = wrap_to_pi(pitch)
        proprioceptive[3] = roll_wrapped.item()
        proprioceptive[4] = pitch_wrapped.item()
        
        # [5] Zero (placeholder)
        proprioceptive[5] = 0.0
        
        # [6] delta_yaw (target yaw - current yaw)
        # Use delta_yaw from HardwareObservations (extracted from environment)
        proprioceptive[6] = float(hw_obs.delta_yaw)
        
        # [7] delta_next_yaw (next target yaw - current yaw)
        # Use delta_next_yaw from HardwareObservations (extracted from environment)
        proprioceptive[7] = float(hw_obs.delta_next_yaw)
        
        # [8] Zero (placeholder for vy command)
        proprioceptive[8] = 0.0
        
        # [9] vx command (forward velocity)
        if nav_cmd is not None:
            proprioceptive[9] = nav_cmd.vx
        else:
            proprioceptive[9] = 0.0
        
        # [10] Terrain type flag (1 if not flat, 0 if flat)
        # Use terrain_type_flag from HardwareObservations (extracted from environment)
        proprioceptive[10] = float(hw_obs.terrain_type_flag)
        
        # [11] Flat terrain flag (1 if flat, 0 if not flat)
        # Use flat_terrain_flag from HardwareObservations (extracted from environment)
        proprioceptive[11] = float(hw_obs.flat_terrain_flag)
        
        # [12:12+num_joints] Joint positions (relative to default)
        n = min(num_joints, len(hw_obs.joint_positions))
        proprioceptive[12 : 12 + n] = hw_obs.joint_positions[:n]

        # [24:24+num_joints] Joint velocities * 0.05 (HAL may already apply scaling)
        n = min(num_joints, len(hw_obs.joint_velocities))
        proprioceptive[24 : 24 + n] = hw_obs.joint_velocities[:n]

        # [36:36+num_joints] Previous action (last joint command)
        n = min(num_joints, len(hw_obs.previous_action))
        proprioceptive[36 : 36 + n] = hw_obs.previous_action[:n]

        # Contact forces: start at 12 + 3*num_joints, count = num_prop - 12 - 3*num_joints
        contact_count = num_prop - 12 - 3 * num_joints
        contact_start = 12 + 3 * num_joints
        n = min(max(0, contact_count), len(hw_obs.contact_forces))
        if contact_count > 0:
            proprioceptive[contact_start : contact_start + n] = hw_obs.contact_forces[:n]
        
        return proprioceptive
    
    def _extract_scan_features(self, hw_obs: HardwareObservations) -> np.ndarray:
        """Extract scan/depth features from hardware observations.

        Matches training format: 132 height measurements from ray scanner.
        Scan features must be provided by the HAL server:
        - Isaac: from observation manager measured_heights.
        - Jetson: from front depth map via HAL policy scan mapping when the camera is initialized.

        If scan_features is None (e.g. Jetson without ZED), returns zeros. Policy
        behavior will be degraded without real scan/terrain data.

        Args:
            hw_obs: Hardware observations

        Returns:
            Scan features array of shape (num_scan,) (front then optional side).
        """
        d = self._dims
        nf = d.num_scan_front
        ns = d.num_side_scan
        total = d.num_scan

        def front_block() -> np.ndarray:
            if hw_obs.scan_features is not None:
                scan_features = hw_obs.scan_features
                if len(scan_features) == nf:
                    return scan_features.astype(np.float32)
                if len(scan_features) > nf:
                    return scan_features[:nf].astype(np.float32)
                features = np.zeros(nf, dtype=np.float32)
                features[: len(scan_features)] = scan_features.astype(np.float32)
                return features
            return np.zeros(nf, dtype=np.float32)

        if ns == 0:
            out = front_block()
            if len(out) != total:
                raise ValueError(
                    f"internal: num_scan_front {nf} != num_scan total {total} with num_side_scan=0"
                )
            return out

        front = front_block()
        if hw_obs.side_scan_features is not None:
            side = hw_obs.side_scan_features
            if len(side) == ns:
                side_arr = side.astype(np.float32)
            elif len(side) > ns:
                side_arr = side[:ns].astype(np.float32)
            else:
                side_arr = np.zeros(ns, dtype=np.float32)
                side_arr[: len(side)] = side.astype(np.float32)
        else:
            side_arr = np.zeros(ns, dtype=np.float32)
        out = np.concatenate([front, side_arr])
        if len(out) != total:
            raise ValueError(
                f"scan concat length {len(out)} != num_scan {total} (front {nf} + side {ns})"
            )
        return out

    def _extract_priv_explicit(self, hw_obs: HardwareObservations) -> np.ndarray:
        """Extract privileged explicit features from hardware observations.
        
        Matches training format: 9 dims
        - [0:3] base_lin_vel * 2.0
        - [3:6] zero (placeholder)
        - [6:9] zero (placeholder)
        
        NOTE: These features are typically estimated by the estimator network
        during inference, so returning zeros is acceptable. The estimator will
        fill them in based on proprioceptive features.
        
        Args:
            hw_obs: Hardware observations
            
        Returns:
            Privileged explicit features array of shape (num_priv_explicit,)
        """
        return np.zeros(self._dims.num_priv_explicit, dtype=np.float32)
    
    def _extract_priv_latent(self, hw_obs: HardwareObservations) -> np.ndarray:
        """Extract privileged latent features from hardware observations.
        
        Matches training format: 29 dims
        - Body mass and center of mass
        - Friction coefficients
        - Joint stiffness and damping (normalized)
        
        In simulation, privileged latent is available from environment.
        On real hardware, it's not available and will be zeros.
        
        Args:
            hw_obs: Hardware observations
            
        Returns:
            Privileged latent features array of shape (num_priv_latent,)
        """
        n = self._dims.num_priv_latent
        if hw_obs.privileged_latent is not None:
            if hw_obs.privileged_latent.shape != (n,):
                raise ValueError(f"privileged_latent shape {hw_obs.privileged_latent.shape} != ({n},)")
            return hw_obs.privileged_latent.astype(np.float32)
        return np.zeros(n, dtype=np.float32)
    
    def _extract_history(self, proprioceptive: np.ndarray) -> np.ndarray:
        """Extract history features from previous proprioceptive observations.
        
        Matches training format: history_dim = num_hist * num_prop (from observation_dimensions).
        Stores last num_hist proprioceptive observations; on episode start fills buffer
        with current observation, otherwise shifts buffer and appends current.
        
        NOTE: The environment observation code zeros out delta_yaw (index 6) and delta_next_yaw
        (index 7) in obs_buf AFTER building the observation, so the history buffer in the returned
        observation has zeros at these positions. To match this behavior, we zero out positions
        6-7 in the proprioceptive observation before storing it in the history buffer.

        Args:
            proprioceptive: Current proprioceptive observation (num_prop,)
            
        Returns:
            History features array of shape (history_dim,)
        """
        # Zero out delta_yaw (index 6) and delta_next_yaw (index 7) before storing in history
        # This matches the environment behavior where obs_buf[:, 6:8] = 0 is applied after
        # building the observation, so the history buffer has zeros at these positions
        # NOTE: Indices 6 and 7 are in the fixed header section [0:12] of the proprioceptive
        # observation, which is constant across all robot types, so hardcoding is safe.
        proprioceptive_for_history = proprioceptive.copy()
        proprioceptive_for_history[6] = 0.0
        proprioceptive_for_history[7] = 0.0
        
        num_hist = self._dims.num_hist
        if self._episode_step <= 1:
            for i in range(num_hist):
                self._history_buffer[i] = proprioceptive_for_history.copy()
        else:
            self._history_buffer[:-1] = self._history_buffer[1:]
            self._history_buffer[-1] = proprioceptive_for_history.copy()
        self._episode_step += 1
        return self._history_buffer.flatten()
