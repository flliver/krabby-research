"""Mapper from Krabby hardware observations to Parkour model format.

This mapper converts raw hardware sensor data into the format expected by
the Parkour policy model. It uses zero-copy operations where possible to
minimize data copying.
"""

import logging
from typing import Optional, Type

import numpy as np

from hal.client.data_structures.hardware import KrabbyHardwareObservations
from hal.client.observation.types import NavigationCommand
from compute.parkour.types import (
    NUM_PROP,
    NUM_SCAN,
    NUM_PRIV_EXPLICIT,
    NUM_PRIV_LATENT,
    HISTORY_DIM,
    OBS_DIM,
    ParkourObservation,
    TeacherObservation,
)

logger = logging.getLogger(__name__)


class KrabbyHWObservationsToParkourMapper:
    """Maps Krabby hardware observations to Parkour model format.
    
    Uses zero-copy operations where possible to minimize data copying.
    Only copies when structural transformation is required.
    
    Zero-copy guarantees:
    - Large arrays (RGB, depth) are processed but may require copies for
      feature extraction (depends on preprocessing pipeline)
    - Joint positions can be views if source is compatible
    - Final observation array is constructed from parts (may require copy)
    
    Note: This is a placeholder implementation. The actual mapping logic
    will depend on:
    - How RGB/depth images are processed into features
    - How joint positions map to proprioceptive features
    - The exact feature extraction pipeline
    """
    
    def __init__(self):
        """Initialize the mapper."""
        pass
    
    def map(self, hw_obs: KrabbyHardwareObservations, nav_cmd: Optional[NavigationCommand] = None, observation_type: Type[ParkourObservation] = TeacherObservation) -> ParkourObservation:
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
        # This is a placeholder - actual implementation will depend on
        # the feature extraction pipeline
        
        # Proprioceptive features (53 dims) - from joint positions and other proprioceptive data
        # For now, pad/truncate joint positions to match NUM_PROP
        proprioceptive = self._extract_proprioceptive(hw_obs, nav_cmd)
        
        # Scan/depth features (132 dims) - from depth map and RGB cameras
        scan = self._extract_scan_features(hw_obs)
        
        # Privileged explicit features (9 dims) - from hardware sensors
        priv_explicit = self._extract_priv_explicit(hw_obs)
        
        # Privileged latent features (29 dims) - from hardware sensors
        priv_latent = self._extract_priv_latent(hw_obs)
        
        # History features (530 dims) - from previous observations
        # For now, use zeros (history should be maintained externally)
        history = np.zeros(HISTORY_DIM, dtype=np.float32)
        
        # Create observation from parts (use specified observation type)
        return observation_type.from_parts(
            proprioceptive=proprioceptive,
            scan=scan,
            priv_explicit=priv_explicit,
            priv_latent=priv_latent,
            history=history,
            timestamp_ns=hw_obs.timestamp_ns,
        )
    
    def _extract_proprioceptive(self, hw_obs: KrabbyHardwareObservations, nav_cmd: Optional[NavigationCommand] = None) -> np.ndarray:
        """Extract proprioceptive features from hardware observations.
        
        Args:
            hw_obs: Hardware observations
            nav_cmd: Optional navigation command to include in proprioceptive features
            
        Returns:
            Proprioceptive features array of shape (NUM_PROP,)
            
        Note:
            Navigation commands are included at positions 8-10:
            - Position 8: vx (forward velocity)
            - Position 9: vy (lateral velocity)
            - Position 10: yaw_rate (angular velocity)
        """
        # Placeholder: map 18 joint positions to 53 proprioceptive features
        # In reality, this would include:
        # - Joint positions (18)
        # - Joint velocities (18)
        # - Base pose (7: position + quaternion)
        # - Base velocities (6: linear + angular)
        # - Other proprioceptive sensors
        
        # For now, pad joint positions to NUM_PROP
        proprioceptive = np.zeros(NUM_PROP, dtype=np.float32)
        num_joints = min(18, NUM_PROP)
        proprioceptive[:num_joints] = hw_obs.joint_positions[:num_joints]
        
        # Include navigation commands at positions 8-10 (matching training format)
        if nav_cmd is not None:
            proprioceptive[8] = nav_cmd.vx
            proprioceptive[9] = nav_cmd.vy
            proprioceptive[10] = nav_cmd.yaw_rate
        
        return proprioceptive
    
    def _extract_scan_features(self, hw_obs: KrabbyHardwareObservations) -> np.ndarray:
        """Extract scan/depth features from hardware observations.
        
        Args:
            hw_obs: Hardware observations
            
        Returns:
            Scan features array of shape (NUM_SCAN,)
        """
        # Placeholder: extract depth features from depth map
        # In reality, this would involve:
        # - Processing depth map into features
        # - Possibly using RGB images for additional features
        # - Feature extraction pipeline (CNN, etc.)
        
        # For now, flatten depth map and take first NUM_SCAN elements
        depth_flat = hw_obs.depth_map.flatten()
        scan = np.zeros(NUM_SCAN, dtype=np.float32)
        num_features = min(len(depth_flat), NUM_SCAN)
        scan[:num_features] = depth_flat[:num_features]
        
        return scan
    
    def _extract_priv_explicit(self, hw_obs: KrabbyHardwareObservations) -> np.ndarray:
        """Extract privileged explicit features from hardware observations.
        
        Args:
            hw_obs: Hardware observations
            
        Returns:
            Privileged explicit features array of shape (NUM_PRIV_EXPLICIT,)
        """
        # Placeholder: extract privileged explicit features
        # These are features available during training but not at inference
        # Examples: ground truth contact forces, terrain properties, etc.
        
        return np.zeros(NUM_PRIV_EXPLICIT, dtype=np.float32)
    
    def _extract_priv_latent(self, hw_obs: KrabbyHardwareObservations) -> np.ndarray:
        """Extract privileged latent features from hardware observations.
        
        Args:
            hw_obs: Hardware observations
            
        Returns:
            Privileged latent features array of shape (NUM_PRIV_LATENT,)
        """
        # Placeholder: extract privileged latent features
        # These are latent representations available during training
        
        return np.zeros(NUM_PRIV_LATENT, dtype=np.float32)

