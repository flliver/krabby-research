"""Tests for teleop.edge.depth_preview."""

from __future__ import annotations

import numpy as np
import pytest

from teleop.edge.depth_preview import depth_meters_to_rgb24_u8


def test_depth_preview_shape_and_gray() -> None:
    d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rgb = depth_meters_to_rgb24_u8(d, depth_range_m=(1.0, 4.0))
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[:, :, 0] == rgb[:, :, 1])
    assert np.all(rgb[:, :, 1] == rgb[:, :, 2])
    assert int(rgb[0, 0, 0]) == 0
    assert int(rgb[1, 1, 0]) == 255


def test_depth_preview_nan_is_black() -> None:
    d = np.ones((4, 4), dtype=np.float32) * 1.5
    d[1:3, 1:3] = np.nan
    rgb = depth_meters_to_rgb24_u8(d, depth_range_m=(1.0, 2.0))
    assert np.all(rgb[2, 2, :] == 0)
    assert int(rgb[0, 0, 0]) > 0


def test_depth_preview_requires_valid_range() -> None:
    d = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="depth_range_m"):
        depth_meters_to_rgb24_u8(d, depth_range_m=(3.0, 1.0))
    with pytest.raises(ValueError, match="depth_range_m"):
        depth_meters_to_rgb24_u8(d, depth_range_m=(1.0, 1.0))


def test_depth_preview_requires_2d() -> None:
    with pytest.raises(ValueError, match="2D"):
        depth_meters_to_rgb24_u8(
            np.zeros((2, 2, 2), dtype=np.float32),
            depth_range_m=(0.0, 1.0),
        )
