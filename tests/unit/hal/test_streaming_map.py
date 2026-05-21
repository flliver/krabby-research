"""Tests for ``hal.server.streaming_map``."""

from __future__ import annotations

import pytest

from hal.server.isaac.sensor_backend_isaac import IsaacSensorInterface
from hal.server.streaming_map import build_encoded_pipeline_for_sensor_id, sensor_by_id

from tests.unit.hal.isaac_sensor_fixtures import ISAAC_CONFIGURED_SENSORS_FIXTURE


@pytest.fixture()
def isaac_iface() -> IsaacSensorInterface:
    return IsaacSensorInterface(configured_sensors=ISAAC_CONFIGURED_SENSORS_FIXTURE)


def test_sensor_by_id(isaac_iface: IsaacSensorInterface) -> None:
    s = sensor_by_id(isaac_iface, "front_rgbd")
    assert s.id == "front_rgbd"


def test_sensor_by_id_missing(isaac_iface: IsaacSensorInterface) -> None:
    with pytest.raises(KeyError, match="not in list_sensors"):
        sensor_by_id(isaac_iface, "does_not_exist")


def test_build_h264_pipeline_front_rgbd(isaac_iface: IsaacSensorInterface) -> None:
    p = build_encoded_pipeline_for_sensor_id(
        isaac_iface, "front_rgbd", encoding="h264", output_element="fakesink"
    )
    assert "appsrc name=src" in p
    assert "fakesink" in p


def test_build_h264_radar(isaac_iface: IsaacSensorInterface) -> None:
    p = build_encoded_pipeline_for_sensor_id(isaac_iface, "radar_front", output_element="fakesink")
    assert "fakesink" in p
