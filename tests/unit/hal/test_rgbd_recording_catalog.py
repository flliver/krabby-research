"""Unit tests for hal.server.rgbd_recording_catalog."""

from __future__ import annotations

import pytest

from data_collection.serialization import catalog_camera_topic
from hal.server.rgbd_recording_catalog import (
    catalog_channels_from_hal_catalog,
    hal_rgbd_catalog_ids_for_recording,
    hal_teleop_bootstrap_catalog_ids,
    rgbd_ids_from_hal_catalog,
    teleop_bootstrap_ids_from_catalog_channels,
)


def test_hal_rgbd_catalog_ids_match_jetson_registry() -> None:
    jetson = pytest.importorskip("hal.server.jetson.sensor_backend_jetson")
    ids = hal_rgbd_catalog_ids_for_recording()
    assert ids == rgbd_ids_from_hal_catalog(jetson.JETSON_SENSOR_CATALOG)
    assert "front_rgbd" in ids
    assert "belly_right_tof" in ids
    assert "belly_left_tof" in ids
    assert "rear_tof" in ids
    assert "front_rgbd_gray16_depth" not in ids


def test_hal_rgbd_catalog_ids_match_isaac_registry_when_jetson_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "hal.server.jetson.sensor_backend_jetson" or (
            fromlist and "sensor_backend_jetson" in fromlist
        ):
            raise ModuleNotFoundError(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import)
    from hal.server.isaac.rgbd_recording_catalog import isaac_hal_rgbd_catalog_ids_for_recording

    assert hal_rgbd_catalog_ids_for_recording() == isaac_hal_rgbd_catalog_ids_for_recording()


def test_catalog_topics_use_full_catalog_id() -> None:
    assert catalog_camera_topic("front_rgbd", "rgb") == "/camera/front_rgbd/rgb"
    assert catalog_camera_topic("front_rgbd", "depth") == "/camera/front_rgbd/depth"


def test_hal_teleop_bootstrap_includes_rgb_and_gray16_depth() -> None:
    jetson = pytest.importorskip("hal.server.jetson.sensor_backend_jetson")
    bootstrap = hal_teleop_bootstrap_catalog_ids()
    expected = teleop_bootstrap_ids_from_catalog_channels(
        catalog_channels_from_hal_catalog(jetson.JETSON_SENSOR_CATALOG)
    )
    assert bootstrap == expected
    assert bootstrap == (
        "front_rgbd",
        "front_rgbd_gray16_depth",
        "belly_right_tof_gray16_depth",
        "belly_left_tof_gray16_depth",
        "rear_tof_gray16_depth",
    )
    assert "belly_right_tof" not in bootstrap
    assert "belly_left_tof" not in bootstrap
    assert "rear_tof" not in bootstrap
