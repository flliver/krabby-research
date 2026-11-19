"""Write a minimal mcap bag from dummy observations."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

pytest.importorskip("rosbags.rosbag2")

from data_collection.config import DataCollectorConfig, HalEndpoints, TopicEnable
from data_collection.rotating_bag import RotatingMcapWriter
from data_collection.serialization import (
    BASE_TWIST_MSGTYPE,
    BASE_TWIST_TOPIC,
    IMU_MSGTYPE,
    IMU_TOPIC,
    JOINTS_COMMAND_TOPIC,
    JOINTS_STATE_TOPIC,
    JOINT_STATE_MSGTYPE,
    catalog_camera_topic,
    observation_to_writes,
)
from hal.client.data_structures.hardware import HardwareObservations, SensorObservation
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
        base_lin_vel_b=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        base_quat_w=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        joint_velocities=np.zeros(12, dtype=np.float32),
        contact_forces=np.zeros(5, dtype=np.float32),
        previous_action=np.linspace(0.0, 0.2, 12, dtype=np.float32),
        rgbd_by_catalog_id={
            "front_rgbd": SensorObservation(channels={"rgb": rgb, "depth": dep}),
        },
    )


def test_rotating_writer_single_message_roundtrip(tmp_path):
    cfg = DataCollectorConfig(
        hal=HalEndpoints("inproc://a", "inproc://b"),
        output_dir=tmp_path / "out",
        topics=TopicEnable(),
        camera_encode="raw",
    )
    specs = [
        (JOINTS_STATE_TOPIC, JOINT_STATE_MSGTYPE),
        (JOINTS_COMMAND_TOPIC, JOINT_STATE_MSGTYPE),
        (IMU_TOPIC, IMU_MSGTYPE),
        (BASE_TWIST_TOPIC, BASE_TWIST_MSGTYPE),
        (catalog_camera_topic("front_rgbd", "rgb"), "sensor_msgs/msg/Image"),
        (catalog_camera_topic("front_rgbd", "depth"), "sensor_msgs/msg/Image"),
    ]
    writer = RotatingMcapWriter(
        cfg.output_dir,
        rotation_max_bytes=10_000_000,
        rotation_max_minutes=60.0,
        min_free_fraction=0.01,
        topic_msgtypes=specs,
    )
    obs = _minimal_obs_with_camera()
    ts = writer.typestore
    rows = observation_to_writes(ts, obs, cfg.topics, (), camera_encode="raw")
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
    assert counts.get(JOINTS_STATE_TOPIC) == 1
    assert counts.get(JOINTS_COMMAND_TOPIC) == 1
    assert counts.get(IMU_TOPIC) == 1
    assert counts.get(BASE_TWIST_TOPIC) == 1
    with AnyReader([bag]) as reader:
        for conn, _ts, raw in reader.messages():
            if conn.topic == BASE_TWIST_TOPIC:
                twist = ts2.deserialize_cdr(raw, BASE_TWIST_MSGTYPE)
                assert twist.twist.linear.x == pytest.approx(0.2)
                break

    shutil.rmtree(cfg.output_dir, ignore_errors=True)


def test_bag_mixes_raw_and_h265_topics(tmp_path):
    """Write raw Image topics and synthetic CompressedImage H.265 side-by-side."""
    from data_collection.serialization import (
        COMPRESSED_IMAGE_MSGTYPE,
        hevc_format_string,
        serialize_compressed_image,
    )
    from hal.client.data_structures.hardware import SensorObservation

    cfg = DataCollectorConfig(
        hal=HalEndpoints("inproc://a", "inproc://b"),
        output_dir=tmp_path / "mixed",
        topics=TopicEnable(
            joints_state=False, joints_command=False, imu=False, base_twist=False
        ),
        camera_encode="raw",
    )
    raw_depth = catalog_camera_topic("front_rgbd", "depth")
    h265_depth = catalog_camera_topic("belly_left_tof", "depth_h265")
    specs = [
        (raw_depth, "sensor_msgs/msg/Image"),
        (h265_depth, COMPRESSED_IMAGE_MSGTYPE),
    ]
    writer = RotatingMcapWriter(
        cfg.output_dir,
        rotation_max_bytes=10_000_000,
        rotation_max_minutes=60.0,
        min_free_fraction=0.01,
        topic_msgtypes=specs,
    )
    ts = writer.typestore
    dep = np.ones((4, 4), dtype=np.float32) * 0.5
    obs = _minimal_obs_with_camera()
    obs.rgbd_by_catalog_id = {
        "front_rgbd": SensorObservation(
            channels={"depth": dep, "rgb": np.zeros((4, 4, 3), dtype=np.uint8)}
        ),
    }
    rows = observation_to_writes(ts, obs, cfg.topics, (), camera_encode="raw")
    depth_rows = [r for r in rows if r[0] == raw_depth]
    fake_hevc = b"\x00\x00\x00\x01\x40\x01" + b"\x00" * 32
    h265_row = (
        h265_depth,
        COMPRESSED_IMAGE_MSGTYPE,
        serialize_compressed_image(
            ts,
            obs.timestamp_ns,
            h265_depth,
            fake_hevc,
            format=hevc_format_string(depth_range_m=(0.15, 1.5)),
        ),
    )
    writer.write_messages(depth_rows + [h265_row], obs.timestamp_ns)
    writer.close()

    bag = next(d for d in cfg.output_dir.iterdir() if (d / "metadata.yaml").is_file())
    topics_seen: set[str] = set()
    with AnyReader([bag]) as reader:
        for conn, _ts, raw in reader.messages():
            topics_seen.add(conn.topic)
            if conn.topic == h265_depth:
                msg = ts.deserialize_cdr(raw, COMPRESSED_IMAGE_MSGTYPE)
                assert msg.format.startswith("hevc")
                assert "range=" in msg.format
    assert raw_depth in topics_seen
    assert h265_depth in topics_seen
    shutil.rmtree(cfg.output_dir, ignore_errors=True)
