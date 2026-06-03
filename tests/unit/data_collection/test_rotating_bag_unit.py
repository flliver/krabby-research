"""Unit tests for data_collection.rotating_bag."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("rosbags.rosbag2")

import numpy as np
from rosbags.typesys import get_typestore
from rosbags.typesys.stores import Stores

from data_collection.rotating_bag import (
    RotatingMcapWriter,
    _bag_dirs,
    _total_bag_bytes,
    enforce_disk_quota,
)
from data_collection.serialization import catalog_camera_topic


def test_bag_dirs_empty_nonexistent(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    assert _bag_dirs(missing) == []


def test_bag_dirs_orders_by_metadata_mtime(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    old = root / "old_bag"
    new = root / "new_bag"
    old.mkdir()
    new.mkdir()
    (old / "metadata.yaml").write_text("rosbag2_bagfile_information:\n  version: 9\n", encoding="utf-8")
    (new / "metadata.yaml").write_text("rosbag2_bagfile_information:\n  version: 9\n", encoding="utf-8")
    import os
    import time

    os.utime(old / "metadata.yaml", (time.time() - 100, time.time() - 100))
    dirs = _bag_dirs(root)
    assert len(dirs) == 2
    assert dirs[0].name == "old_bag"
    assert dirs[1].name == "new_bag"


def test_total_bag_bytes_counts_files(tmp_path: Path) -> None:
    root = tmp_path / "out"
    b = root / "b0"
    b.mkdir(parents=True)
    (b / "metadata.yaml").write_text("x", encoding="utf-8")
    (b / "data.mcap").write_bytes(b"0123456789")
    assert _total_bag_bytes(root) == 11


def test_enforce_disk_quota_deletes_oldest(tmp_path: Path) -> None:
    root = tmp_path / "out"
    root.mkdir()
    for name, age in (("a", 200), ("b", 100)):
        d = root / name
        d.mkdir()
        (d / "metadata.yaml").write_text("rosbag2_bagfile_information:\n  version: 9\n", encoding="utf-8")
        (d / "f.bin").write_bytes(b"x" * 50)
        import os
        import time

        t = time.time() - age
        os.utime(d / "metadata.yaml", (t, t))
    enforce_disk_quota(root, max_bytes=80)
    assert _total_bag_bytes(root) <= 80


def test_enforce_disk_quota_no_dirs_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    root = tmp_path / "out"
    root.mkdir()
    with caplog.at_level(logging.ERROR):
        with patch("data_collection.rotating_bag._total_bag_bytes", return_value=1_000_000), patch(
            "data_collection.rotating_bag._bag_dirs", return_value=[]
        ):
            enforce_disk_quota(root, max_bytes=0)
    assert "no bag directories" in caplog.text


def _tiny_rgb_cdr(ts) -> bytes:
    Header = ts.types["std_msgs/msg/Header"]
    Stamp = ts.types["builtin_interfaces/msg/Time"]
    Img = ts.types["sensor_msgs/msg/Image"]
    h = Header(stamp=Stamp(sec=0, nanosec=0), frame_id="c")
    msg = Img(header=h, height=1, width=1, encoding="rgb8", is_bigendian=0, step=3, data=np.zeros(3, dtype=np.uint8))
    return ts.serialize_cdr(msg, "sensor_msgs/msg/Image")


def test_rotating_writer_skips_unknown_connection(tmp_path: Path) -> None:
    ts = get_typestore(Stores.LATEST)
    payload = _tiny_rgb_cdr(ts)
    rgb_topic = catalog_camera_topic("front_rgbd", "rgb")
    specs = [(rgb_topic, "sensor_msgs/msg/Image")]
    w = RotatingMcapWriter(
        tmp_path / "bags",
        rotation_max_bytes=10_000_000,
        rotation_max_minutes=60.0,
        max_disk_usage_fraction=0.99,
        topic_msgtypes=specs,
    )
    w.write_messages(
        [
            (rgb_topic, "sensor_msgs/msg/Image", payload),
            ("/unknown/topic", "sensor_msgs/msg/Image", payload),
        ],
        1,
    )
    w.close()


def test_rotating_writer_rotates_when_max_minutes_zero(tmp_path: Path) -> None:
    ts = get_typestore(Stores.LATEST)
    payload = _tiny_rgb_cdr(ts)
    root = tmp_path / "bags"
    rgb_topic = catalog_camera_topic("front_rgbd", "rgb")
    specs = [(rgb_topic, "sensor_msgs/msg/Image")]
    w = RotatingMcapWriter(
        root,
        rotation_max_bytes=10_000_000,
        rotation_max_minutes=0.0,
        max_disk_usage_fraction=0.99,
        topic_msgtypes=specs,
    )
    w.write_messages([(rgb_topic, "sensor_msgs/msg/Image", payload)], 100)
    w.write_messages([(rgb_topic, "sensor_msgs/msg/Image", payload)], 200)
    w.close()
    bag_dirs = [d for d in root.iterdir() if d.is_dir() and (d / "metadata.yaml").is_file()]
    assert len(bag_dirs) >= 2


def test_close_segment_oserror_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    rgb_topic = catalog_camera_topic("front_rgbd", "rgb")
    specs = [(rgb_topic, "sensor_msgs/msg/Image")]
    w = RotatingMcapWriter(
        tmp_path / "bags",
        rotation_max_bytes=10_000_000,
        rotation_max_minutes=60.0,
        max_disk_usage_fraction=0.99,
        topic_msgtypes=specs,
    )
    w.ensure_started()
    mock_writer = MagicMock()
    mock_writer.path = w._writer.path
    mock_writer.close.side_effect = OSError("fail close")
    w._writer = mock_writer
    with caplog.at_level(logging.ERROR):
        w._close_segment()
    assert "Error closing bag" in caplog.text
    w.close()


def test_close_idempotent(tmp_path: Path) -> None:
    w = RotatingMcapWriter(
        tmp_path / "bags",
        rotation_max_bytes=10_000_000,
        rotation_max_minutes=60.0,
        max_disk_usage_fraction=0.99,
        topic_msgtypes=[(catalog_camera_topic("front_rgbd", "rgb"), "sensor_msgs/msg/Image")],
    )
    w.close()
    w.close()
