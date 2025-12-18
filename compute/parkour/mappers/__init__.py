"""Mappers for converting between hardware and Parkour model data formats.

These mappers use zero-copy operations where possible to minimize data copying.
They handle structural transformation between hardware sensor data and Parkour
model observation formats.
"""

from compute.parkour.mappers.hardware_to_model import HWObservationsToParkourMapper
from compute.parkour.mappers.model_to_hardware import ParkourLocomotionToHWMapper

__all__ = [
    "HWObservationsToParkourMapper",
    "ParkourLocomotionToHWMapper",
]

