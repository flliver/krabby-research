"""Write a minimal mcap bag from dummy observations."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

pytest.importorskip("rosbags.rosbag2")

from data_collection.config import DataCollectorConfig, HalEndpoints, TopicEnable
from data_collection.rotating_bag import RotatingMcapWriter
from data_collection.serialization import catalog_camera_topic, observation_to_writes
from hal.client.data_structures.hardware import HardwareObservations, RgbdCatalogObservation
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_typestore
from rosbags.typesys.stores import Stores


def _minimal_obs_with_camera() -> HardwareObservations:
    h, w = 4, 6
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    dep = np.ones((h, w), dtype=np.float32) * 0.5
    return HardwareObservations(
        joint_positions=np.linspace(-0.1, 0.1, 12, dtype=np.float32),
        camera_height=h,
        camera_width=w,
        timestamp_ns=1_500_000_000,
        base_ang_vel_b=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        base_lin_vel_b=np.zeros(3, dtype=np.float32),
        base_quat_w=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.linspace(0.0, 0.2, 12, dtype=np.float32),
        rgbd_by_catalog_id={
            "front_rgbd": RgbdCatalogObservation(rgb=rgb, depth=dep),
        },
    )


def test_rotating_writer_single_message_roundtrip(tmp_path):
    cfg = DataCollectorConfig(
        hal=HalEndpoints("inproc://a", "inproc://b"),
        output_dir=tmp_path / "out",
        topics=TopicEnable(),
    )
    specs = [
        ("/joints/state", "sensor_msgs/msg/JointState"),
        ("/joints/command", "sensor_msgs/msg/JointState"),
        ("/imu", "sensor_msgs/msg/Imu"),
    ]
    writer = RotatingMcapWriter(
        cfg.output_dir,
        rotation_max_bytes=10_000_000,
        rotation_max_minutes=60.0,
        max_disk_usage_fraction=0.99,
        topic_msgtypes=specs,
    )
    obs = _minimal_obs_with_camera()
    ts = writer.typestore
    rows = observation_to_writes(ts, obs, cfg.topics, ())
    writer.write_messages(rows, obs.timestamp_ns)
    writer.close()

    bag_dirs = [d for d in cfg.output_dir.iterdir() if d.is_dir() and (d / "metadata.yaml").is_file()]
    assert len(bag_dirs) == 1
    bag = bag_dirs[0]

    ts2 = get_typestore(Stores.LATEST)
    counts: dict[str, int] = {}
    rgb_topic = catalog_camera_topic("front_rgbd", "rgb")
    depth_topic = catalog_camera_topic("front_rgbd", "depth")
    with AnyReader([bag]) as reader:
        for conn, _ts, raw in reader.messages():
            counts[conn.topic] = counts.get(conn.topic, 0) + 1
            if conn.topic == rgb_topic:
                msg = ts2.deserialize_cdr(raw, "sensor_msgs/msg/Image")
                assert msg.width == 6 and msg.height == 4

    assert counts.get(rgb_topic) == 1
    assert counts.get(depth_topic) == 1
    assert counts.get("/joints/state") == 1
    assert counts.get("/joints/command") == 1
    assert counts.get("/imu") == 1

    shutil.rmtree(cfg.output_dir, ignore_errors=True)
