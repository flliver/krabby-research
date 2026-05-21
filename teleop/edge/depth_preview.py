"""Metric depth (float32 meters) → uint8 RGB preview for WebRTC (rgb24)."""

from __future__ import annotations

import math

import numpy as np


def depth_meters_to_rgb24_u8(
    depth_m: np.ndarray,
    *,
    depth_range_m: tuple[float, float],
) -> np.ndarray:
    """Map (H, W) float depth in **meters** to (H, W, 3) uint8 **RGB** (identical channels = gray).

    ``depth_range_m`` is ``(d_min, d_max)`` in meters (same band as GRAY16 / sensor catalog); must
    satisfy ``d_max > d_min`` with finite endpoints. Finite depth is linearly mapped and clipped to
    0..255; non-finite values stay black.

    Output is always ``(H, W, 3)`` at the **native depth grid** — no resize.
    """
    lo, hi = float(depth_range_m[0]), float(depth_range_m[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        raise ValueError(
            f"depth_range_m must be finite (d_min, d_max) with d_max > d_min, got {depth_range_m!r}"
        )

    d = np.asarray(depth_m, dtype=np.float32)
    if d.ndim != 2:
        raise ValueError(f"depth_m must be 2D (H, W), got shape {d.shape}")
    h, w = d.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    finite = np.isfinite(d)
    if not np.any(finite):
        return out

    x = np.zeros_like(d, dtype=np.float32)
    x[finite] = (d[finite] - lo) / (hi - lo)
    np.clip(x, 0.0, 1.0, out=x)
    u8 = (x * 255.0).astype(np.uint8)
    out[:, :, 0] = u8
    out[:, :, 1] = u8
    out[:, :, 2] = u8
    return out
