"""Second `HalClient` recording loop."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import zmq

from data_collection.config import DataCollectorConfig
from data_collection.serialization import IMAGE_MSGTYPE, is_catalog_camera_topic, observation_to_writes
from hal.client.client import HalClient
from hal.client.config import HalClientConfig

logger = logging.getLogger(__name__)


def _topic_msgtype_catalog(cfg: DataCollectorConfig) -> list[tuple[str, str]]:
    """Static (topic, msgtype) pairs registered at bag open.

    Catalog camera topics are added on first write (see ``RotatingMcapWriter``).
    """
    t = cfg.topics
    pairs: list[tuple[str, str]] = []
    js = "sensor_msgs/msg/JointState"
    if t.joints_state:
        pairs.append(("/joints/state", js))
    if t.joints_command:
        pairs.append(("/joints/command", js))
    if t.imu:
        pairs.append(("/imu", "sensor_msgs/msg/Imu"))
    return pairs


class HalDataCollector:
    """Polls a dedicated `HalClient` and writes observations to rotating mcap bags."""

    def __init__(self, cfg: DataCollectorConfig, zmq_context: zmq.Context) -> None:
        self._cfg = cfg
        self._ctx = zmq_context
        hcfg = HalClientConfig(
            observation_endpoint=cfg.hal.observation_endpoint,
            command_endpoint=cfg.hal.command_endpoint,
        )
        self._client = HalClient(hcfg, context=zmq_context)
        self._topic_specs = _topic_msgtype_catalog(cfg)
        self._bag = None
        try:
            from data_collection.rotating_bag import RotatingMcapWriter

            self._bag = RotatingMcapWriter(
                cfg.output_dir,
                rotation_max_bytes=cfg.rotation_max_bytes,
                rotation_max_minutes=cfg.rotation_max_minutes,
                max_disk_usage_fraction=cfg.max_disk_usage_fraction,
                topic_msgtypes=self._topic_specs,
            )
        except (ImportError, RuntimeError) as e:
            logger.warning("Recording disabled (rosbags / writer unavailable): %s", e)

    def initialize(self) -> None:
        self._client.initialize()
        logger.info(
            "HalDataCollector initialized (observation=%s)",
            self._cfg.hal.observation_endpoint,
        )

    def close(self) -> None:
        if self._bag is not None:
            try:
                self._bag.close()
            except OSError as e:
                logger.error("Error closing bag writer: %s", e, exc_info=True)
        self._client.close()

    def run(self, stop_event: threading.Event) -> None:
        """Blocking loop until ``stop_event`` is set."""
        if self._bag is None:
            logger.error("HalDataCollector.run: no bag writer; exiting thread")
            return
        period_img = 1.0 / max(self._cfg.rates.images_hz, 1e-6)
        period_state = 1.0 / max(self._cfg.rates.joints_imu_hz, 1e-6)
        next_img = 0.0
        next_state = 0.0
        last_obs_ts: Optional[int] = None
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_img and now < next_state:
                remaining = min(next_img - now, next_state - now)
                stop_event.wait(timeout=max(0.0, min(remaining, 0.05)))
                continue
            obs = self._client.poll(timeout_ms=self._cfg.polling_timeout_ms)
            if obs is None:
                continue
            if last_obs_ts is not None and obs.timestamp_ns == last_obs_ts:
                stop_event.wait(timeout=0.002)
                continue
            last_obs_ts = obs.timestamp_ns
            now = time.monotonic()
            rows = observation_to_writes(
                self._bag.typestore,
                obs,
                self._cfg.topics,
                self._cfg.joint_names,
            )
            try:
                if now >= next_img:
                    img_rows = [
                        r
                        for r in rows
                        if r[1] == IMAGE_MSGTYPE and is_catalog_camera_topic(r[0])
                    ]
                    if img_rows:
                        self._bag.write_messages(img_rows, obs.timestamp_ns)
                    next_img = now + period_img
                if now >= next_state:
                    st_rows = [r for r in rows if r[0] in ("/joints/state", "/joints/command", "/imu")]
                    if st_rows:
                        self._bag.write_messages(st_rows, obs.timestamp_ns)
                    next_state = now + period_state
            except OSError:
                logger.error("HalDataCollector stopping due to write failure")
                break


def start_collector_thread(
    cfg: DataCollectorConfig,
    zmq_context: zmq.Context,
    stop_event: threading.Event,
) -> tuple[HalDataCollector, threading.Thread]:
    """Create collector, spawn daemon thread. Call ``collector.initialize()`` before start."""
    collector = HalDataCollector(cfg, zmq_context)

    def _target() -> None:
        try:
            collector.run(stop_event)
        finally:
            collector.close()

    th = threading.Thread(target=_target, name="HalDataCollector", daemon=True)
    return collector, th
