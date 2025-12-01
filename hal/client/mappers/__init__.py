"""Mappers for converting between hardware and model data formats.

These mappers use zero-copy operations where possible to minimize data copying.
They handle structural transformation between hardware sensor data and model
observation formats.
"""

from hal.mappers.hardware_to_model import KrabbyHWObservationsToParkourMapper
from hal.mappers.model_to_hardware import ParkourLocomotionToKrabbyHWMapper

__all__ = [
    "KrabbyHWObservationsToParkourMapper",
    "ParkourLocomotionToKrabbyHWMapper",
]

