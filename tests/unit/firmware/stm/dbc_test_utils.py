"""Helpers for krabby.dbc decode assertions in STM unit tests."""

from __future__ import annotations

import struct


def dbc_float64(value: object) -> float:
    """cantools may decode IEEE float64 signals as unsigned integer bit patterns."""
    if isinstance(value, int):
        return struct.unpack("<d", struct.pack("<Q", value & ((1 << 64) - 1)))[0]
    return float(value)