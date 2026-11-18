"""Unit tests for 8-bit depth quantization."""

from __future__ import annotations

import numpy as np
import pytest

from data_collection.depth_quant import (
    depth_quant_step_m,
    float32_depth_to_gray8,
    gray8_to_float32_depth,
)
from data_collection.tools.decode_h265 import check_depth_roundtrip


def test_float32_depth_to_gray8_roundtrip_within_step() -> None:
    lo, hi = 0.15, 1.5
    depth = np.linspace(lo, hi, 64, dtype=np.float32).reshape(8, 8)
    err = check_depth_roundtrip(depth, d_min=lo, d_max=hi)
    assert err <= depth_quant_step_m(lo, hi) + 1e-6


def test_invalid_depth_maps_to_nan_on_decode() -> None:
    lo, hi = 0.2, 25.0
    depth = np.array([[np.nan, 0.0, 100.0], [1.0, 5.0, np.inf]], dtype=np.float32)
    gray = float32_depth_to_gray8(depth, lo, hi)
    assert gray[0, 0] == 255
    assert gray[0, 1] == 255  # below band
    assert gray[0, 2] == 255  # above band
    assert gray[1, 0] != 255
    back = gray8_to_float32_depth(gray, lo, hi)
    assert np.isnan(back[0, 0])
    assert np.isnan(back[0, 1])
    assert np.isfinite(back[1, 0])


def test_quant_step_rejects_bad_range() -> None:
    with pytest.raises(ValueError):
        depth_quant_step_m(1.0, 1.0)
