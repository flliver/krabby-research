"""HAL data collector configuration (dataclasses).

Defaults for production are assembled in ``data_collection/collector_settings.py``.
``from_dict`` / ``load_config`` remain for tests and optional YAML overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


JointsCommandSource = Literal["previous_action"]
"""`/joints/command` is recorded from `HardwareObservations.previous_action` (last
command applied / echoed in the observation pipeline), not from a separate tap on `put_joint_command`."""

CameraEncodeMode = Literal["h265", "raw"]
"""How catalog camera channels are written to the bag."""


@dataclass
class HalEndpoints:
    """ZMQ endpoint strings (same as `HalClientConfig`).

    For in-proc deployment these must match the primary client and server bind URLs exactly.
    """

    observation_endpoint: str
    command_endpoint: str


@dataclass
class RecordingRates:
    """Target maximum record cadence (wall clock). Latest-only HAL semantics may yield fewer samples.

    Image cadence is per catalog id: ``images_hz_by_catalog_id[id]`` if present, else
    ``default_images_hz``.
    """

    default_images_hz: float = 10.0
    images_hz_by_catalog_id: dict[str, float] = field(default_factory=dict)
    joints_imu_hz: float = 50.0

    def images_hz_for(self, catalog_id: str) -> float:
        return float(self.images_hz_by_catalog_id.get(catalog_id, self.default_images_hz))


@dataclass
class TopicEnable:
    """Which non-catalog ROS topics to write when data is available.

    RGB-D catalog streams are recorded from ``HardwareObservations.rgbd_by_catalog_id``.
    With ``camera_encode="h265"`` they land as ``/camera/{catalog_id}/{channel}_h265``
    (CompressedImage); with ``"raw"`` as ``/camera/{catalog_id}/{channel}`` (Image).
    """

    joints_state: bool = True
    joints_command: bool = True
    imu: bool = True
    base_twist: bool = True


@dataclass
class DataCollectorConfig:
    hal: HalEndpoints
    output_dir: Path
    min_free_fraction: float = 0.10
    rotation_max_bytes: int = 262_144_000  # 250 MiB
    rotation_max_minutes: float = 5.0
    rates: RecordingRates = field(default_factory=RecordingRates)
    topics: TopicEnable = field(default_factory=TopicEnable)
    joint_names: tuple[str, ...] = ()
    joints_command_source: JointsCommandSource = "previous_action"
    polling_timeout_ms: int = 10
    camera_encode: CameraEncodeMode = "h265"

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> DataCollectorConfig:
        hal_raw = raw["hal"]
        hal = HalEndpoints(
            observation_endpoint=str(hal_raw["observation_endpoint"]),
            command_endpoint=str(hal_raw["command_endpoint"]),
        )
        rates_raw = raw.get("rates") or {}
        by_id_raw = rates_raw.get("images_hz_by_catalog_id") or {}
        if not isinstance(by_id_raw, dict):
            raise ValueError("rates.images_hz_by_catalog_id must be a mapping")
        # Legacy YAML key ``images_hz`` maps to default_images_hz.
        default_hz = rates_raw.get("default_images_hz", rates_raw.get("images_hz", 10.0))
        rates = RecordingRates(
            default_images_hz=float(default_hz),
            images_hz_by_catalog_id={str(k): float(v) for k, v in by_id_raw.items()},
            joints_imu_hz=float(rates_raw.get("joints_imu_hz", 50.0)),
        )
        topics_raw = raw.get("topics") or {}
        topics = TopicEnable(
            joints_state=bool(topics_raw.get("joints_state", True)),
            joints_command=bool(topics_raw.get("joints_command", True)),
            imu=bool(topics_raw.get("imu", True)),
            base_twist=bool(topics_raw.get("base_twist", True)),
        )
        jsrc = raw.get("joints_command_source", "previous_action")
        if jsrc != "previous_action":
            raise ValueError(
                f"joints_command_source={jsrc!r} unsupported; only 'previous_action' is implemented "
                "(see docs/DATA_COLLECTOR.md)."
            )
        enc = str(raw.get("camera_encode", "h265"))
        if enc not in ("h265", "raw"):
            raise ValueError(f"camera_encode={enc!r} unsupported; use 'h265' or 'raw'")
        joint_names_raw = raw.get("joint_names") or []
        joint_names = tuple(str(x) for x in joint_names_raw)
        return DataCollectorConfig(
            hal=hal,
            output_dir=Path(raw["output_dir"]).expanduser(),
            min_free_fraction=float(raw.get("min_free_fraction", 0.10)),
            rotation_max_bytes=int(raw.get("rotation_max_bytes", 262_144_000)),
            rotation_max_minutes=float(raw.get("rotation_max_minutes", 5.0)),
            rates=rates,
            topics=topics,
            joint_names=joint_names,
            joints_command_source="previous_action",
            polling_timeout_ms=int(raw.get("polling_timeout_ms", 10)),
            camera_encode=enc,  # type: ignore[arg-type]
        )


def load_config(path: Path | str) -> DataCollectorConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {p} must be a YAML mapping")
    return DataCollectorConfig.from_dict(raw)
