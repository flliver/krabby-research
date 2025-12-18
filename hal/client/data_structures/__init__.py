"""Hardware data structures for Krabby robot.

These structures represent raw hardware sensor data and desired joint positions.
They are designed for zero-copy operations where possible.
"""

from hal.client.data_structures.hardware import (
    HardwareObservations,
    JointCommand,
)

__all__ = [
    "HardwareObservations",
    "JointCommand",
]

