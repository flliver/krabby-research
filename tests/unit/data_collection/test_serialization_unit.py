"""Unit tests for data_collection.serialization."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rosbags.rosbag2")

from rosbags.typesys import get_typestore
from rosbags.typesys.stores import Stores

from data_collection.config import TopicEnable
from data_collection.serialization import (
    catalog_camera_topic,
    observation_to_writes,
    serialize_joint_state,
)
from hal.client.data_structures.hardware import RgbdCatalogObservation
from tests.helpers import create_dummy_hw_obs


def _ts():
    return get_typestore(Stores.LATEST)


def test_observation_to_writes_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        observation_to_writes(_ts(), object(), TopicEnable())


def test_serialize_joint_state_renames_when_wrong_name_count() -> None:
    ts = _ts()
    pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    vel = np.zeros(3, dtype=np.float32)
    raw = serialize_joint_state(ts, 100, "base", ("only_one",), pos, vel)
    msg = ts.deserialize_cdr(raw, "sensor_msgs/msg/JointState")
    assert list(msg.name) == ["joint_0", "joint_1", "joint_2"]


def test_rgbd_catalog_writes_rgb_and_depth_topics() -> None:
    ts = _ts()
    topics = TopicEnable(joints_state=False, joints_command=False, imu=False)
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    dep = np.ones((2, 3), dtype=np.float32)
    obs = create_dummy_hw_obs(camera_height=2, camera_width=3)
    obs.rgbd_by_catalog_id = {
        "side_rgbd": RgbdCatalogObservation(rgb=rgb, depth=dep),
        "front_rgbd": RgbdCatalogObservation(rgb=rgb, depth=dep),
    }
    rows = observation_to_writes(ts, obs, topics, ())
    topics_found = {r[0] for r in rows}
    assert topics_found == {
        catalog_camera_topic("front_rgbd", "rgb"),
        catalog_camera_topic("front_rgbd", "depth"),
        catalog_camera_topic("side_rgbd", "rgb"),
        catalog_camera_topic("side_rgbd", "depth"),
    }


def test_catalog_mono_rgb_uses_mono8() -> None:
    ts = _ts()
    topics = TopicEnable(joints_state=False, joints_command=False, imu=False)
    gray = np.zeros((4, 5), dtype=np.uint8)
    obs = create_dummy_hw_obs(camera_height=4, camera_width=5)
    obs.rgbd_by_catalog_id = {
        "radar_front": RgbdCatalogObservation(rgb=gray, depth=np.ones((4, 5), dtype=np.float32)),
    }
    rows = observation_to_writes(ts, obs, topics, ())
    rgb_row = next(r for r in rows if r[0] == catalog_camera_topic("radar_front", "rgb"))
    img = ts.deserialize_cdr(rgb_row[2], "sensor_msgs/msg/Image")
    assert img.encoding == "mono8"


def test_split_stamp_via_header_timestamp() -> None:
    ts = _ts()
    obs = create_dummy_hw_obs()
    obs.timestamp_ns = 3_500_000_123
    topics = TopicEnable(joints_state=False, joints_command=False, imu=True)
    rows = observation_to_writes(ts, obs, topics, ())
    imu_row = next(r for r in rows if r[0] == "/imu")
    msg = ts.deserialize_cdr(imu_row[2], "sensor_msgs/msg/Imu")
    assert msg.header.stamp.sec == 3
    assert msg.header.stamp.nanosec == 500_000_123
