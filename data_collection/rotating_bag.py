"""Rosbag2 mcap writer with rotation and output-directory disk quota."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from rosbags.rosbag2 import Writer, StoragePlugin
    from rosbags.typesys.stores import Stores
    from rosbags.typesys import get_typestore
    from rosbags.interfaces import Connection
except ImportError:
    Writer = None  # type: ignore[misc, assignment]
    StoragePlugin = None  # type: ignore[misc, assignment]
    get_typestore = None  # type: ignore[misc, assignment]
    Stores = None  # type: ignore[misc, assignment]
    Connection = None  # type: ignore[misc, assignment]


def _bag_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    out: list[Path] = []
    for child in output_dir.iterdir():
        if child.is_dir() and (child / "metadata.yaml").is_file():
            out.append(child)
    return sorted(out, key=lambda p: (p / "metadata.yaml").stat().st_mtime)


def _total_bag_bytes(output_dir: Path) -> int:
    total = 0
    for bag in _bag_dirs(output_dir):
        for f in bag.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total


def enforce_disk_quota(output_dir: Path, max_bytes: int) -> None:
    """Delete oldest bag directories (by ``metadata.yaml`` mtime) until under ``max_bytes``."""
    while _total_bag_bytes(output_dir) > max_bytes:
        dirs = _bag_dirs(output_dir)
        if not dirs:
            logger.error(
                "Disk quota exceeded but no bag directories to delete under %s (max_bytes=%s)",
                output_dir,
                max_bytes,
            )
            return
        victim = dirs[0]
        logger.warning("Deleting oldest bag to respect disk quota: %s", victim)
        shutil.rmtree(victim, ignore_errors=True)


class RotatingMcapWriter:
    """Single rosbag2 folder per segment; mcap storage; size and time rotation."""

    def __init__(
        self,
        output_dir: Path,
        *,
        rotation_max_bytes: int,
        rotation_max_minutes: float,
        max_disk_usage_fraction: float,
        topic_msgtypes: list[tuple[str, str]],
    ) -> None:
        if Writer is None or get_typestore is None:
            raise ImportError(
                "rosbags is required for RotatingMcapWriter; add it to images/locomotion/requirements.txt "
                "or images/isaacsim/requirements.txt"
            )
        self._output_dir = Path(output_dir)
        self._rotation_max_bytes = int(rotation_max_bytes)
        self._rotation_max_minutes = float(rotation_max_minutes)
        self._max_disk_fraction = float(max_disk_usage_fraction)
        self._topic_msgtypes = list(topic_msgtypes)
        self._typestore = get_typestore(Stores.LATEST)
        self._writer: Optional[Writer] = None
        self._connections: dict[tuple[str, str], Connection] = {}
        self._segment_started_monotonic: float = 0.0
        self._seq = 0

    @property
    def typestore(self):
        return self._typestore

    def _max_collector_bytes(self) -> int:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(self._output_dir)
        return int(self._max_disk_fraction * usage.total)

    def _current_segment_bytes(self) -> int:
        if self._writer is None:
            return 0
        total = 0
        for f in self._writer.path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total

    def _open_segment(self) -> None:
        if self._writer is not None:
            self._close_segment()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        enforce_disk_quota(self._output_dir, self._max_collector_bytes())
        seg_name = f"krabby_{self._seq:06d}_{time.strftime('%Y%m%d_%H%M%S')}"
        self._seq += 1
        path = self._output_dir / seg_name
        if path.exists():
            raise FileExistsError(path)
        self._writer = Writer(path, version=9, storage_plugin=StoragePlugin.MCAP)
        self._writer.open()
        self._connections.clear()
        for topic, msgtype in self._topic_msgtypes:
            conn = self._writer.add_connection(topic, msgtype, typestore=self._typestore)
            self._connections[(topic, msgtype)] = conn
        self._segment_started_monotonic = time.monotonic()
        logger.info("Opened new mcap segment %s", path)

    def _close_segment(self) -> None:
        if self._writer is None:
            return
        path = self._writer.path
        try:
            self._writer.close()
        except OSError as e:
            logger.error("Error closing bag at %s: %s", path, e, exc_info=True)
        self._writer = None
        self._connections.clear()
        logger.info("Closed bag segment %s", path)

    def ensure_started(self) -> None:
        if self._writer is None:
            self._open_segment()

    def _connection_for(self, topic: str, msgtype: str) -> Connection:
        assert self._writer is not None
        key = (topic, msgtype)
        conn = self._connections.get(key)
        if conn is not None:
            return conn
        conn = self._writer.add_connection(topic, msgtype, typestore=self._typestore)
        self._connections[key] = conn
        return conn

    def write_messages(self, rows: list[tuple[str, str, bytes]], stamp_ns: int) -> None:
        """Write rows (topic, msg_type, data). Opens first segment lazily."""
        self.ensure_started()
        assert self._writer is not None
        for topic, msgtype, data in rows:
            try:
                conn = self._connection_for(topic, msgtype)
                self._writer.write(conn, int(stamp_ns), data)
            except OSError as e:
                logger.error("Bag write failed (%s): %s", topic, e, exc_info=True)
                raise
        self._maybe_rotate()

    def _maybe_rotate(self) -> None:
        if self._writer is None:
            return
        too_big = self._current_segment_bytes() >= self._rotation_max_bytes
        too_old = (time.monotonic() - self._segment_started_monotonic) >= (
            self._rotation_max_minutes * 60.0
        )
        if too_big or too_old:
            logger.info("Rotating bag (too_big=%s too_old=%s)", too_big, too_old)
            self._close_segment()
            enforce_disk_quota(self._output_dir, self._max_collector_bytes())
            self._open_segment()

    def close(self) -> None:
        self._close_segment()
