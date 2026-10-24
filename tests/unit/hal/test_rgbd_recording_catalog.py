"""Unit tests for hal.server.rgbd_recording_catalog."""

from __future__ import annotations

import pytest

from data_collection.serialization import catalog_camera_topic
from hal.server.rgbd_recording_catalog import (
    hal_rgbd_catalog_ids_for_recording,
    rgbd_ids_from_hal_catalog,
)


def test_hal_rgbd_catalog_ids_match_jetson_registry() -> None:
    jetson = pytest.importorskip("hal.server.jetson.sensor_backend_jetson")
    ids = hal_rgbd_catalog_ids_for_recording()
    assert ids == rgbd_ids_from_hal_catalog(jetson.JETSON_SENSOR_CATALOG)
    assert "front_rgbd" in ids
    assert "side_right_rgbd" in ids
    assert "side_left_rgbd" in ids
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
