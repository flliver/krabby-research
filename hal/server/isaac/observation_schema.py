"""Summarize ``HardwareObservations`` field names and array shapes for regression tests."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np

from hal.client.data_structures.hardware import HardwareObservations, RgbdCatalogObservation

__all__ = ["hardware_observation_schema"]


def _array_schema(arr: np.ndarray) -> tuple[tuple[int, ...], str]:
    return (tuple(int(x) for x in arr.shape), str(arr.dtype))


def _rgbd_entry(entry: RgbdCatalogObservation) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rgb": _array_schema(entry.rgb),
        "depth": _array_schema(entry.depth),
    }
    if entry.scan_features is not None:
        out["scan_features"] = _array_schema(entry.scan_features)
    else:
        out["scan_features"] = None
    return out


def hardware_observation_schema(obs: HardwareObservations) -> dict[str, Any]:
    """JSON-serializable map of field name → shape/dtype (or scalar type)."""
    schema: dict[str, Any] = {}
    for field in fields(obs):
        value = getattr(obs, field.name)
        if value is None:
            schema[field.name] = None
        elif isinstance(value, np.ndarray):
            schema[field.name] = _array_schema(value)
        elif isinstance(value, dict):
            schema[field.name] = {
                str(key): _rgbd_entry(item) if isinstance(item, RgbdCatalogObservation) else item
                for key, item in value.items()
            }
        elif isinstance(value, (int, float, bool)):
            schema[field.name] = type(value).__name__
        else:
            schema[field.name] = type(value).__name__
    return schema
