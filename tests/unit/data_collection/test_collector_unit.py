"""Unit tests for data_collection.collector (topic catalog + mocked run loop)."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import zmq

from data_collection.collector import HalDataCollector, _topic_msgtype_catalog
from data_collection.config import DataCollectorConfig, HalEndpoints, TopicEnable
from tests.helpers import create_dummy_hw_obs

pytest.importorskip("rosbags.rosbag2")
from rosbags.typesys import get_typestore
from rosbags.typesys.stores import Stores


def _cfg(tmp: Path) -> DataCollectorConfig:
    return DataCollectorConfig(
        hal=HalEndpoints("inproc://a", "inproc://b"),
        output_dir=tmp / "out",
        topics=TopicEnable(),
    )


def test_topic_msgtype_catalog_respects_flags() -> None:
    cfg = DataCollectorConfig(
        hal=HalEndpoints("x", "y"),
        output_dir=Path("/tmp/z"),
        topics=TopicEnable(joints_state=False, joints_command=False, imu=False),
    )
    assert _topic_msgtype_catalog(cfg) == []


def test_topic_msgtype_catalog_includes_proprioception() -> None:
    cfg = DataCollectorConfig(
        hal=HalEndpoints("x", "y"),
        output_dir=Path("/tmp/z"),
    )
    pairs = _topic_msgtype_catalog(cfg)
    topics = {p[0] for p in pairs}
    assert "/joints/state" in topics
    assert "/imu" in topics


def test_hal_data_collector_run_no_bag_exits(tmp_path: Path) -> None:
    ctx = zmq.Context()
    cfg = _cfg(tmp_path)
    coll = HalDataCollector(cfg, ctx)
    coll._bag = None
    stop = threading.Event()
    coll.run(stop)
    ctx.term()


def test_hal_data_collector_run_writes_via_mock(tmp_path: Path) -> None:
    ctx = zmq.Context()
    cfg = _cfg(tmp_path)
    cfg.rates.images_hz = 1000.0
    cfg.rates.joints_imu_hz = 1000.0
    coll = HalDataCollector(cfg, ctx)

    ts = get_typestore(Stores.LATEST)
    mock_bag = MagicMock()
    mock_bag.typestore = ts
    coll._bag = mock_bag

    stop_evt = threading.Event()
    polls = {"n": 0}

    def poll_side_effect(timeout_ms: int = 10):
        polls["n"] += 1
        if polls["n"] > 4:
            stop_evt.set()
            return None
        o = create_dummy_hw_obs(camera_height=8, camera_width=8)
        o.timestamp_ns = 9_000_000_000 + polls["n"]
        return o
    coll._client = MagicMock()
    coll._client.poll.side_effect = poll_side_effect

    coll.run(stop_evt)

    assert mock_bag.write_messages.called
    ctx.term()


def test_hal_data_collector_close_bag_oserror_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    ctx = zmq.Context()
    cfg = _cfg(tmp_path)
    coll = HalDataCollector(cfg, ctx)
    mock_bag = MagicMock()
    mock_bag.close.side_effect = OSError("disk")
    coll._bag = mock_bag
    with caplog.at_level(logging.ERROR):
        coll.close()
    assert "Error closing bag" in caplog.text
    ctx.term()


def test_hal_data_collector_run_oserror_stops_loop(tmp_path: Path) -> None:
    ctx = zmq.Context()
    cfg = _cfg(tmp_path)
    cfg.rates.images_hz = 1000.0
    cfg.rates.joints_imu_hz = 1000.0
    coll = HalDataCollector(cfg, ctx)

    ts = get_typestore(Stores.LATEST)
    mock_bag = MagicMock()
    mock_bag.typestore = ts
    mock_bag.write_messages.side_effect = OSError("disk full")
    coll._bag = mock_bag

    obs = create_dummy_hw_obs(camera_height=8, camera_width=8)
    obs.timestamp_ns = 1_000_000_000
    coll._client = MagicMock()
    coll._client.poll.return_value = obs

    stop = threading.Event()
    coll.run(stop)
    assert mock_bag.write_messages.called
    ctx.term()


def test_start_collector_thread_stops_and_closes(tmp_path: Path) -> None:
    from data_collection.collector import start_collector_thread

    ctx = zmq.Context()
    cfg = _cfg(tmp_path)
    stop = threading.Event()
    coll, th = start_collector_thread(cfg, ctx, stop)
    coll._bag = None
    th.start()
    th.join(timeout=2.0)
    assert not th.is_alive()
    ctx.term()
