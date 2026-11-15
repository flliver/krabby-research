"""Isaac sim depth bands stay aligned with Jetson sensor catalog / SEN0581 range."""

from __future__ import annotations

from hal.server.isaac.sim_depth_ranges import (
    ISAAC_FRONT_DEPTH_CLIP_M,
    ISAAC_FRONT_DEPTH_POSINF_CLIP_M,
    ISAAC_FRONT_DEPTH_PREVIEW_M,
    ISAAC_FRONT_RGB_CLIP_M,
    ISAAC_TOF_DEPTH_CLIP_M,
    ISAAC_TOF_DEPTH_POSINF_CLIP_M,
    ISAAC_TOF_DEPTH_PREVIEW_M,
)
from hal.server.jetson.sensor_backend_jetson import JETSON_SENSOR_CATALOG_BY_ID


def test_front_depth_bands_match_zed_catalog() -> None:
    zed = JETSON_SENSOR_CATALOG_BY_ID["front_rgbd"]
    assert zed.gst_depth_quant_range_m == ISAAC_FRONT_DEPTH_PREVIEW_M
    assert ISAAC_FRONT_DEPTH_CLIP_M == ISAAC_FRONT_DEPTH_PREVIEW_M
    assert ISAAC_FRONT_DEPTH_POSINF_CLIP_M == ISAAC_FRONT_DEPTH_CLIP_M[1]
    assert ISAAC_FRONT_RGB_CLIP_M == ISAAC_FRONT_DEPTH_CLIP_M


def test_tof_depth_bands_match_sen0581_range() -> None:
    assert ISAAC_TOF_DEPTH_CLIP_M == (0.15, 1.5)
    assert ISAAC_TOF_DEPTH_PREVIEW_M == ISAAC_TOF_DEPTH_CLIP_M
    assert ISAAC_TOF_DEPTH_POSINF_CLIP_M == ISAAC_TOF_DEPTH_CLIP_M[1]
    for cid in ("belly_right_tof", "belly_left_tof", "rear_tof"):
        row = JETSON_SENSOR_CATALOG_BY_ID[cid]
        assert row.gst_depth_quant_range_m == ISAAC_TOF_DEPTH_PREVIEW_M
