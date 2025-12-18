"""Hardware data structures for Krabby robot.

These structures represent raw hardware sensor data and desired joint positions.
They are designed for zero-copy operations where possible.
"""

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HardwareObservations:
    """Hardware observation data.
    
    Contains all raw sensor data from the hardware:
    - Joint positions (18 DOF)
    - Camera data (2x RGB)
    - Depth map
    - Confidence map
    - Camera resolution metadata (self-describing)
    
    Zero-copy guarantees:
    - Arrays are stored as numpy arrays (may be views or copies depending on source)
    - Large arrays (RGB, depth, confidence) should use views when possible
    - Scalar values (timestamp, camera dimensions) are copied
    """
    
    joint_positions: np.ndarray  # Shape: (18,), dtype: float32
    rgb_camera_1: np.ndarray  # Shape: (camera_height, camera_width, 3), dtype: uint8 or float32
    rgb_camera_2: np.ndarray  # Shape: (camera_height, camera_width, 3), dtype: uint8 or float32
    depth_map: np.ndarray  # Shape: (camera_height, camera_width), dtype: float32
    confidence_map: np.ndarray  # Shape: (camera_height, camera_width), dtype: float32
    camera_height: int  # Height of camera images
    camera_width: int  # Width of camera images
    timestamp_ns: int
    
    def __post_init__(self) -> None:
        """Validate hardware observations."""
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        
        if self.camera_height <= 0 or self.camera_width <= 0:
            raise ValueError(f"Camera dimensions must be positive, got {self.camera_height}x{self.camera_width}")
        
        # Validate joint positions
        if self.joint_positions.shape != (18,):
            raise ValueError(
                f"joint_positions shape {self.joint_positions.shape} != (18,)"
            )
        if self.joint_positions.dtype != np.float32:
            # Convert to float32 if needed (creates copy)
            self.joint_positions = self.joint_positions.astype(np.float32)
        
        # Validate camera arrays (check shape consistency with metadata)
        expected_shape_2d = (self.camera_height, self.camera_width)
        expected_shape_3d = (self.camera_height, self.camera_width, 3)
        
        if self.rgb_camera_1.shape != self.rgb_camera_2.shape:
            raise ValueError(
                f"Camera shapes must match: {self.rgb_camera_1.shape} != {self.rgb_camera_2.shape}"
            )
        if self.rgb_camera_1.shape != expected_shape_3d:
            raise ValueError(
                f"RGB camera must be {expected_shape_3d}, got {self.rgb_camera_1.shape}"
            )
        
        # Validate depth and confidence maps
        if self.depth_map.shape != self.confidence_map.shape:
            raise ValueError(
                f"Depth and confidence shapes must match: {self.depth_map.shape} != {self.confidence_map.shape}"
            )
        if self.depth_map.shape != expected_shape_2d:
            raise ValueError(
                f"Depth map must be {expected_shape_2d}, got {self.depth_map.shape}"
            )
        
        # Ensure depth and confidence are float32
        if self.depth_map.dtype != np.float32:
            self.depth_map = self.depth_map.astype(np.float32)
        if self.confidence_map.dtype != np.float32:
            self.confidence_map = self.confidence_map.astype(np.float32)
    
    def to_bytes(self) -> list[bytes]:
        """Serialize to bytes for ZMQ transport.
        
        Format: multipart message with metadata and arrays:
        - Part 0: metadata JSON (shapes, dtypes, timestamp)
        - Part 1: joint_positions bytes
        - Part 2: rgb_camera_1 bytes
        - Part 3: rgb_camera_2 bytes
        - Part 4: depth_map bytes
        - Part 5: confidence_map bytes
        
        Returns:
            List of bytes for multipart ZMQ message
        """
        # Ensure arrays are contiguous and correct dtype
        joint_pos = np.ascontiguousarray(self.joint_positions, dtype=np.float32)
        rgb1 = np.ascontiguousarray(self.rgb_camera_1, dtype=self.rgb_camera_1.dtype)
        rgb2 = np.ascontiguousarray(self.rgb_camera_2, dtype=self.rgb_camera_2.dtype)
        depth = np.ascontiguousarray(self.depth_map, dtype=np.float32)
        conf = np.ascontiguousarray(self.confidence_map, dtype=np.float32)
        
        # Create metadata
        metadata = {
            "joint_positions": {"shape": list(joint_pos.shape), "dtype": str(joint_pos.dtype)},
            "rgb_camera_1": {"shape": list(rgb1.shape), "dtype": str(rgb1.dtype)},
            "rgb_camera_2": {"shape": list(rgb2.shape), "dtype": str(rgb2.dtype)},
            "depth_map": {"shape": list(depth.shape), "dtype": str(depth.dtype)},
            "confidence_map": {"shape": list(conf.shape), "dtype": str(conf.dtype)},
            "camera_height": self.camera_height,
            "camera_width": self.camera_width,
            "timestamp_ns": self.timestamp_ns,
        }
        
        return [
            json.dumps(metadata).encode("utf-8"),
            joint_pos.tobytes(),
            rgb1.tobytes(),
            rgb2.tobytes(),
            depth.tobytes(),
            conf.tobytes(),
        ]
    
    @classmethod
    def from_bytes(cls, parts: list[bytes]) -> "HardwareObservations":
        """Deserialize from ZMQ multipart message.
        
        Args:
            parts: List of bytes from ZMQ multipart message
                Expected format: [metadata_json, joint_positions, rgb_camera_1, rgb_camera_2, depth_map, confidence_map]
            
        Returns:
            HardwareObservations instance
            
        Raises:
            ValueError: If message format is invalid (wrong number of parts, invalid JSON, etc.)
        """
        if len(parts) != 6:
            raise ValueError(f"Expected 6 parts (metadata + 5 arrays), got {len(parts)}")
        
        # Parse metadata - fail fast on invalid JSON
        try:
            metadata = json.loads(parts[0].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid metadata JSON: {e}") from e
        
        # Deserialize arrays - let numpy raise errors if shapes/dtypes are wrong
        try:
            joint_pos = np.frombuffer(parts[1], dtype=np.dtype(metadata["joint_positions"]["dtype"]))
            joint_pos = joint_pos.reshape(tuple(metadata["joint_positions"]["shape"])).astype(np.float32)
            
            rgb1 = np.frombuffer(parts[2], dtype=np.dtype(metadata["rgb_camera_1"]["dtype"]))
            rgb1 = rgb1.reshape(tuple(metadata["rgb_camera_1"]["shape"]))
            
            rgb2 = np.frombuffer(parts[3], dtype=np.dtype(metadata["rgb_camera_2"]["dtype"]))
            rgb2 = rgb2.reshape(tuple(metadata["rgb_camera_2"]["shape"]))
            
            depth = np.frombuffer(parts[4], dtype=np.dtype(metadata["depth_map"]["dtype"]))
            depth = depth.reshape(tuple(metadata["depth_map"]["shape"])).astype(np.float32)
            
            conf = np.frombuffer(parts[5], dtype=np.dtype(metadata["confidence_map"]["dtype"]))
            conf = conf.reshape(tuple(metadata["confidence_map"]["shape"])).astype(np.float32)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Error deserializing arrays: {e}") from e
        
        # Extract camera dimensions from metadata or infer from array shapes
        camera_height = metadata.get("camera_height")
        camera_width = metadata.get("camera_width")
        if camera_height is None or camera_width is None:
            # Fallback: extract from rgb_camera_1 shape if not in metadata
            if len(rgb1.shape) >= 2:
                camera_height = rgb1.shape[0]
                camera_width = rgb1.shape[1]
            else:
                raise ValueError("Cannot determine camera dimensions from metadata or array shapes")
        
        return cls(
            joint_positions=joint_pos,
            rgb_camera_1=rgb1,
            rgb_camera_2=rgb2,
            depth_map=depth,
            confidence_map=conf,
            camera_height=camera_height,
            camera_width=camera_width,
            timestamp_ns=metadata.get("timestamp_ns", 0),
        )
    


@dataclass
class JointCommand:
    """Joint command structure.
    
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
        
        # Runtime type validation: Validate values (check for NaN/Inf)
        if not np.isfinite(self.joint_positions).all():
            nan_count = np.isnan(self.joint_positions).sum()
            inf_count = np.isinf(self.joint_positions).sum()
            raise ValueError(f"Invalid joint_positions values: {nan_count} NaN, {inf_count} Inf")
    
    def to_bytes(self) -> list[bytes]:
        """Serialize to bytes for ZMQ transport.
        
        Format: multipart message with metadata and array:
        - Part 0: metadata JSON (shape, dtype, timestamp)
        - Part 1: joint_positions bytes
        
        Returns:
            List of bytes for multipart ZMQ message
        """
        # Ensure array is contiguous and correct dtype
        joint_pos = np.ascontiguousarray(self.joint_positions, dtype=np.float32)
        
        # Create metadata
        metadata = {
            "joint_positions": {"shape": list(joint_pos.shape), "dtype": str(joint_pos.dtype)},
            "timestamp_ns": self.timestamp_ns,
        }
        
        return [
            json.dumps(metadata).encode("utf-8"),
            joint_pos.tobytes(),
        ]
    
    @classmethod
    def from_bytes(cls, parts: list[bytes]) -> "JointCommand":
        """Deserialize from ZMQ multipart message.
        
        Args:
            parts: List of bytes from ZMQ multipart message
                Expected format: [metadata_json, joint_positions]
            
        Returns:
            JointCommand instance
            
        Raises:
            ValueError: If message format is invalid (wrong number of parts, invalid JSON, etc.)
        """
        if len(parts) != 2:
            raise ValueError(f"Expected 2 parts (metadata + array), got {len(parts)}")
        
        # Parse metadata - fail fast on invalid JSON
        try:
            metadata = json.loads(parts[0].decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Invalid metadata JSON: {e}") from e
        
        # Deserialize array - let numpy raise errors if shapes/dtypes are wrong
        try:
            joint_pos = np.frombuffer(parts[1], dtype=np.dtype(metadata["joint_positions"]["dtype"]))
            joint_pos = joint_pos.reshape(tuple(metadata["joint_positions"]["shape"])).astype(np.float32)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Error deserializing array: {e}") from e
        
        return cls(
            joint_positions=joint_pos,
            timestamp_ns=metadata.get("timestamp_ns", 0),
        )
    

