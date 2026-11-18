"""Second `HalClient` recording loop."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import zmq

from data_collection.config import DataCollectorConfig
from data_collection.serialization import (
    BASE_TWIST_MSGTYPE,
    BASE_TWIST_TOPIC,
    IMU_MSGTYPE,
    IMU_TOPIC,
    JOINTS_COMMAND_TOPIC,
    JOINTS_STATE_TOPIC,
    JOINT_STATE_MSGTYPE,
    catalog_camera_topic_msgtypes,
    is_catalog_camera_msgtype,
    is_catalog_camera_topic,
    observation_to_writes,
)
from hal.client.client import HalClient
from hal.client.config import HalClientConfig
from hal.server.rgbd_recording_catalog import (
    hal_rgbd_catalog_channels_for_recording,
)

_STATE_TOPICS = frozenset(
    {JOINTS_STATE_TOPIC, JOINTS_COMMAND_TOPIC, IMU_TOPIC, BASE_TWIST_TOPIC}
)

logger = logging.getLogger(__name__)


def _topic_msgtype_catalog(cfg: DataCollectorConfig) -> list[tuple[str, str]]:
    """Static (topic, msgtype) pairs registered at bag open.

    Catalog camera channels come from ``hal_rgbd_catalog_channels_for_recording()``;
    per-observation writes still only emit keys present in ``rgbd_by_catalog_id``.
    """
    t = cfg.topics
    pairs: list[tuple[str, str]] = list(
        catalog_camera_topic_msgtypes(
            hal_rgbd_catalog_channels_for_recording(),
            camera_encode=cfg.camera_encode,
        )
    )
    if t.joints_state:
        pairs.append((JOINTS_STATE_TOPIC, JOINT_STATE_MSGTYPE))
    if t.joints_command:
        pairs.append((JOINTS_COMMAND_TOPIC, JOINT_STATE_MSGTYPE))
    if t.imu:
        pairs.append((IMU_TOPIC, IMU_MSGTYPE))
    if t.base_twist:
        pairs.append((BASE_TWIST_TOPIC, BASE_TWIST_MSGTYPE))
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
        self._h265 = None
        if cfg.camera_encode == "h265":
            try:
                from data_collection.h265_encode import H265FrameEncoder

                self._h265 = H265FrameEncoder()
            except Exception as e:
                logger.warning("H.265 encoder unavailable; recording disabled: %s", e)
        try:
            from data_collection.rotating_bag import RotatingMcapWriter

            self._bag = RotatingMcapWriter(
                cfg.output_dir,
                rotation_max_bytes=cfg.rotation_max_bytes,
                rotation_max_minutes=cfg.rotation_max_minutes,
                min_free_fraction=cfg.min_free_fraction,
                topic_msgtypes=self._topic_specs,
            )
        except (ImportError, RuntimeError) as e:
            logger.warning("Recording disabled (rosbags / writer unavailable): %s", e)

    def initialize(self) -> None:
        self._client.initialize()
        logger.info(
            "HalDataCollector initialized (observation=%s camera_encode=%s)",
            self._cfg.hal.observation_endpoint,
            self._cfg.camera_encode,
        )

    def close(self) -> None:
        if self._h265 is not None:
            try:
                self._h265.close()
            except Exception as e:
                logger.error("Error closing H.265 encoder: %s", e, exc_info=True)
            self._h265 = None
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
        if self._cfg.camera_encode == "h265" and self._h265 is None:
            logger.error("HalDataCollector.run: H.265 encoder missing; exiting thread")
            return

        rates = self._cfg.rates
        # Per-catalog next-fire times; state uses a single period.
        next_img_by_id: dict[str, float] = {}
        period_state = 1.0 / max(rates.joints_imu_hz, 1e-6)
        next_state = 0.0
        last_obs_ts: Optional[int] = None
        while not stop_event.is_set():
            now = time.monotonic()
            # Sleep until the soonest due image or state sample.
            due_times = [next_state]
            due_times.extend(next_img_by_id.values())
            # Also wake periodically so new catalog ids get scheduled.
            soonest = min(due_times) if due_times else now
            if now < soonest:
                remaining = soonest - now
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

            rgbd = obs.rgbd_by_catalog_id or {}
            due_ids = [
                cid
                for cid in rgbd.keys()
                if now >= next_img_by_id.get(cid, 0.0)
            ]
            write_images = bool(due_ids)
            write_state = now >= next_state

            if not write_images and not write_state:
                continue

            # Restrict image serialization to due catalog ids only.
            original_rgbd = obs.rgbd_by_catalog_id
            if write_images and due_ids:
                obs.rgbd_by_catalog_id = {cid: rgbd[cid] for cid in due_ids}
            elif not write_images:
                obs.rgbd_by_catalog_id = {}
            try:
                try:
                    rows = observation_to_writes(
                        self._bag.typestore,
                        obs,
                        self._cfg.topics,
                        self._cfg.joint_names,
                        camera_encode=self._cfg.camera_encode,
                        h265_encoder=self._h265,
                        images_hz_for=rates.images_hz_for,
                    )
                except Exception as e:
                    # Keep the collector alive across transient encode failures.
                    logger.error("HalDataCollector serialize failed: %s", e, exc_info=True)
                    if write_images:
                        for cid in due_ids:
                            period = 1.0 / max(rates.images_hz_for(cid), 1e-6)
                            next_img_by_id[cid] = now + period
                    if write_state:
                        next_state = now + period_state
                    continue
            finally:
                obs.rgbd_by_catalog_id = original_rgbd

            try:
                if write_images:
                    img_rows = [
                        r
                        for r in rows
                        if is_catalog_camera_msgtype(r[1]) and is_catalog_camera_topic(r[0])
                    ]
                    if img_rows:
                        self._bag.write_messages(img_rows, obs.timestamp_ns)
                    for cid in due_ids:
                        period = 1.0 / max(rates.images_hz_for(cid), 1e-6)
                        next_img_by_id[cid] = now + period
                if write_state:
                    st_rows = [r for r in rows if r[0] in _STATE_TOPICS]
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
