"""Unit tests for hal.server.rgbd_recording_catalog."""

from __future__ import annotations

from data_collection.serialization import catalog_camera_topic

from hal.server.rgbd_recording_catalog import hal_rgbd_catalog_ids_for_recording


def test_hal_rgbd_catalog_ids_match_jetson_registry() -> None:
    ids = hal_rgbd_catalog_ids_for_recording()
    assert "front_rgbd" in ids
    assert "side_right_rgbd" in ids
    assert "side_left_rgbd" in ids
    assert "front_rgbd_gray16_depth" not in ids


def test_catalog_topics_use_full_catalog_id() -> None:
    assert catalog_camera_topic("front_rgbd", "rgb") == "/camera/front_rgbd/rgb"
    assert catalog_camera_topic("front_rgbd", "depth") == "/camera/front_rgbd/depth"
