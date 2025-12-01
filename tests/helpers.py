"""Test helpers for creating dummy hardware data structures."""

import time
from typing import Optional

import numpy as np

from hal.client.data_structures.hardware import (
    KrabbyDesiredJointPositions,
    KrabbyHardwareObservations,
)


def create_dummy_hw_obs(
    camera_height: int = 480,
    camera_width: int = 640,
    timestamp_ns: Optional[int] = None,
) -> KrabbyHardwareObservations:
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
    
    return KrabbyHardwareObservations(
        joint_positions=np.zeros(18, dtype=np.float32),
        rgb_camera_1=np.zeros((camera_height, camera_width, 3), dtype=np.uint8),
        rgb_camera_2=np.zeros((camera_height, camera_width, 3), dtype=np.uint8),
        depth_map=np.zeros((camera_height, camera_width), dtype=np.float32),
        confidence_map=np.ones((camera_height, camera_width), dtype=np.float32),
        timestamp_ns=timestamp_ns,
    )


def create_dummy_joint_positions(
    timestamp_ns: Optional[int] = None,
) -> KrabbyDesiredJointPositions:
    """Create dummy desired joint positions for testing.
    
    Args:
        timestamp_ns: Optional timestamp (defaults to current time)
    
    Returns:
        KrabbyDesiredJointPositions with dummy data
    """
    if timestamp_ns is None:
        timestamp_ns = time.time_ns()
    
    return KrabbyDesiredJointPositions(
        joint_positions=np.zeros(18, dtype=np.float32),
        timestamp_ns=timestamp_ns,
    )

