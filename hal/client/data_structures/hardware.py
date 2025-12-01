"""Hardware data structures for Krabby robot.

These structures represent raw hardware sensor data and desired joint positions.
They are designed for zero-copy operations where possible.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class KrabbyHardwareObservations:
    """Hardware observation data from Krabby robot.
    
    Contains all raw sensor data from the hardware:
    - Joint positions (18 DOF)
    - Camera data (2x RGB)
    - Depth map
    - Confidence map
    
    Note: This is a dummy structure for now. The real hardware spec will be
    published soon and this structure will be updated accordingly.
    
    Zero-copy guarantees:
    - Arrays are stored as numpy arrays (may be views or copies depending on source)
    - Large arrays (RGB, depth, confidence) should use views when possible
    - Scalar values (timestamp) are copied
    """
    
    joint_positions: np.ndarray  # Shape: (18,), dtype: float32
    rgb_camera_1: np.ndarray  # Shape: (H, W, 3), dtype: uint8 or float32
    rgb_camera_2: np.ndarray  # Shape: (H, W, 3), dtype: uint8 or float32
    depth_map: np.ndarray  # Shape: (H, W), dtype: float32
    confidence_map: np.ndarray  # Shape: (H, W), dtype: float32
    timestamp_ns: int
    
    def __post_init__(self) -> None:
        """Validate hardware observations."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        
        # Validate joint positions
        if self.joint_positions.shape != (18,):
            raise ValueError(
                f"joint_positions shape {self.joint_positions.shape} != (18,)"
            )
        if self.joint_positions.dtype != np.float32:
            # Convert to float32 if needed (creates copy)
            self.joint_positions = self.joint_positions.astype(np.float32)
        
        # Validate camera arrays (check shape consistency)
        if self.rgb_camera_1.shape != self.rgb_camera_2.shape:
            raise ValueError(
                f"Camera shapes must match: {self.rgb_camera_1.shape} != {self.rgb_camera_2.shape}"
            )
        if len(self.rgb_camera_1.shape) != 3 or self.rgb_camera_1.shape[2] != 3:
            raise ValueError(
                f"RGB camera must be (H, W, 3), got {self.rgb_camera_1.shape}"
            )
        
        # Validate depth and confidence maps
        if self.depth_map.shape != self.confidence_map.shape:
            raise ValueError(
                f"Depth and confidence shapes must match: {self.depth_map.shape} != {self.confidence_map.shape}"
            )
        if len(self.depth_map.shape) != 2:
            raise ValueError(f"Depth map must be 2D, got shape {self.depth_map.shape}")
        
        # Ensure depth and confidence are float32
        if self.depth_map.dtype != np.float32:
            self.depth_map = self.depth_map.astype(np.float32)
        if self.confidence_map.dtype != np.float32:
            self.confidence_map = self.confidence_map.astype(np.float32)
    
    @classmethod
    def create_dummy(
        cls,
        camera_height: int = 480,
        camera_width: int = 640,
        timestamp_ns: Optional[int] = None,
    ) -> "KrabbyHardwareObservations":
        """Create dummy hardware observations for testing.
        
        Args:
            camera_height: Height of camera images (default 480)
            camera_width: Width of camera images (default 640)
            timestamp_ns: Optional timestamp (defaults to current time)
        
        Returns:
            KrabbyHardwareObservations with dummy data
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        
        return cls(
            joint_positions=np.zeros(18, dtype=np.float32),
            rgb_camera_1=np.zeros((camera_height, camera_width, 3), dtype=np.uint8),
            rgb_camera_2=np.zeros((camera_height, camera_width, 3), dtype=np.uint8),
            depth_map=np.zeros((camera_height, camera_width), dtype=np.float32),
            confidence_map=np.ones((camera_height, camera_width), dtype=np.float32),
            timestamp_ns=timestamp_ns,
        )


@dataclass
class KrabbyDesiredJointPositions:
    """Desired joint positions for Krabby robot.
    
    Contains 18 target joint positions for hardware control.
    
    Zero-copy guarantees:
    - joint_positions array may be a view if source is compatible
    - timestamp is always copied (scalar)
    """
    
    joint_positions: np.ndarray  # Shape: (18,), dtype: float32
    timestamp_ns: int
    
    def __post_init__(self) -> None:
        """Validate desired joint positions."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        
        if self.joint_positions.shape != (18,):
            raise ValueError(
                f"joint_positions shape {self.joint_positions.shape} != (18,)"
            )
        if self.joint_positions.dtype != np.float32:
            # Convert to float32 if needed (creates copy)
            self.joint_positions = self.joint_positions.astype(np.float32)
    
    @classmethod
    def create_dummy(cls, timestamp_ns: Optional[int] = None) -> "KrabbyDesiredJointPositions":
        """Create dummy desired joint positions for testing.
        
        Args:
            timestamp_ns: Optional timestamp (defaults to current time)
        
        Returns:
            KrabbyDesiredJointPositions with dummy data
        """
        if timestamp_ns is None:
            timestamp_ns = time.time_ns()
        
        return cls(
            joint_positions=np.zeros(18, dtype=np.float32),
            timestamp_ns=timestamp_ns,
        )

