"""Depth quantization for H.265 recording (8-bit mono).

Maps metric depth into ``[d_min, d_max]`` linearly onto ``0..254``; ``255`` is the
invalid sentinel (NaN / Inf / out of band). Over a 1.35 m ToF band that is about
5.3 mm per step (``(d_max - d_min) / 254``), which is inside typical ToF noise past
half a meter. Prefer 8-bit over 10-bit so appsrc→encode stays on the simple GRAY8
path used elsewhere in HAL.
"""

from __future__ import annotations

import numpy as np


_VALID_MAX_U8 = 254
_INVALID_U8 = 255


def depth_quant_step_m(d_min: float, d_max: float) -> float:
    """Meters per code step for 8-bit depth (valid codes 0..254)."""
    if d_max <= d_min:
        raise ValueError("d_max must be greater than d_min")
    return float(d_max - d_min) / float(_VALID_MAX_U8)


def float32_depth_to_gray8(
    depth_m: np.ndarray,
    d_min: float,
    d_max: float,
) -> np.ndarray:
    """Map metric depth (meters, float32) to **GRAY8** uint8.

    In-range depth maps linearly to ``0..254``; ``255`` = invalid.
    """
    if d_max <= d_min:
        raise ValueError("d_max must be greater than d_min")
    d = np.asarray(depth_m, dtype=np.float32)
    out = np.full(d.shape, _INVALID_U8, dtype=np.uint8)
    valid = np.isfinite(d) & (d >= d_min) & (d <= d_max)
    span = float(d_max - d_min)
    scaled = (d[valid].astype(np.float64) - d_min) / span * float(_VALID_MAX_U8)
    u = np.rint(scaled).astype(np.int64)
    u = np.clip(u, 0, _VALID_MAX_U8).astype(np.uint8)
    out[valid] = u
    return out


def gray8_to_float32_depth(
    gray: np.ndarray,
    d_min: float,
    d_max: float,
) -> np.ndarray:
    """Inverse of :func:`float32_depth_to_gray8` (invalid → NaN)."""
    if d_max <= d_min:
        raise ValueError("d_max must be greater than d_min")
    g = np.asarray(gray)
    out = np.full(g.shape, np.nan, dtype=np.float32)
    valid = g != _INVALID_U8
    span = float(d_max - d_min)
    out[valid] = (d_min + (g[valid].astype(np.float64) / float(_VALID_MAX_U8)) * span).astype(
        np.float32
    )
    return out


def depth_quant_range_for_catalog_id(catalog_id: str) -> tuple[float, float]:
    """Return ``(d_min, d_max)`` meters for a catalog camera id.

    Prefers Jetson catalog ``gst_depth_quant_range_m`` when that module is importable;
    otherwise uses Isaac front / ToF preview bands.
    """
    try:
        from hal.server.jetson.sensor_backend_jetson import JETSON_SENSOR_CATALOG_BY_ID
    except ModuleNotFoundError:
        pass
    else:
        entry = JETSON_SENSOR_CATALOG_BY_ID.get(catalog_id)
        if entry is not None and entry.gst_depth_quant_range_m is not None:
            lo, hi = entry.gst_depth_quant_range_m
            return float(lo), float(hi)

    from hal.server.isaac.sim_depth_ranges import (
        ISAAC_FRONT_DEPTH_PREVIEW_M,
        ISAAC_TOF_DEPTH_PREVIEW_M,
    )

    if catalog_id == "front_rgbd" or catalog_id.startswith("front"):
        return ISAAC_FRONT_DEPTH_PREVIEW_M
    return ISAAC_TOF_DEPTH_PREVIEW_M
