"""Isaac sim depth bands stay aligned with Jetson sensor catalog."""

from __future__ import annotations

from hal.server.isaac.sim_depth_ranges import (
    ISAAC_FRONT_DEPTH_CLIP_M,
    ISAAC_FRONT_DEPTH_POSINF_CLIP_M,
    ISAAC_FRONT_DEPTH_PREVIEW_M,
    ISAAC_FRONT_RGB_CLIP_M,
    ISAAC_SIDE_DEPTH_CLIP_M,
    ISAAC_SIDE_DEPTH_POSINF_CLIP_M,
    ISAAC_SIDE_DEPTH_PREVIEW_M,
    ISAAC_SIDE_RGB_CLIP_M,
)
from hal.server.jetson.sensor_backend_jetson import JETSON_SENSOR_CATALOG_BY_ID


def test_front_depth_bands_match_zed_catalog() -> None:
    zed = JETSON_SENSOR_CATALOG_BY_ID["front_rgbd"]
    assert zed.gst_depth_quant_range_m == ISAAC_FRONT_DEPTH_PREVIEW_M
    assert ISAAC_FRONT_DEPTH_CLIP_M == ISAAC_FRONT_DEPTH_PREVIEW_M
    assert ISAAC_FRONT_DEPTH_POSINF_CLIP_M == ISAAC_FRONT_DEPTH_CLIP_M[1]
    assert ISAAC_FRONT_RGB_CLIP_M == ISAAC_FRONT_DEPTH_CLIP_M


def test_side_depth_bands_match_maixsense_catalog() -> None:
    for cid in ("side_right_rgbd", "side_left_rgbd"):
        row = JETSON_SENSOR_CATALOG_BY_ID[cid]
        assert row.gst_depth_quant_range_m == ISAAC_SIDE_DEPTH_PREVIEW_M
    assert ISAAC_SIDE_DEPTH_CLIP_M == ISAAC_SIDE_DEPTH_PREVIEW_M
    assert ISAAC_SIDE_DEPTH_POSINF_CLIP_M == ISAAC_SIDE_DEPTH_CLIP_M[1]
    assert ISAAC_SIDE_RGB_CLIP_M == (0.15, 4.0)
    assert ISAAC_SIDE_RGB_CLIP_M != ISAAC_SIDE_DEPTH_CLIP_M
