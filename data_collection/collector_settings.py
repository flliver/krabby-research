"""Default HAL data collector parameters.

Edit values here (same idea as ``JETSON_SENSOR_CATALOG`` in
``hal/server/jetson/sensor_backend_jetson.py``): one Python module as the source of truth,
not a separate YAML layer.

``build_data_collector_config`` wires HAL transport strings from the server entrypoint;
everything else comes from the constants below.
"""

from __future__ import annotations

from pathlib import Path

from data_collection.config import (
    DataCollectorConfig,
    HalEndpoints,
    RecordingRates,
    TopicEnable,
)

# Writable bag root (override at runtime with ``--data-collector-output-dir``).
DEFAULT_OUTPUT_DIR = Path("/data/krabby_bags")

MIN_FREE_FRACTION: float = 0.10
ROTATION_MAX_BYTES: int = 262_144_000  # 250 MiB
ROTATION_MAX_MINUTES: float = 5.0

# Front RGB-D at 10 Hz; SEN0581 ToF depth rows at 20 Hz (sensor native fps).
RECORDING_RATES = RecordingRates(
    default_images_hz=10.0,
    images_hz_by_catalog_id={
        "front_rgbd": 10.0,
        "belly_left_tof": 20.0,
        "belly_right_tof": 20.0,
        "rear_tof": 20.0,
    },
    joints_imu_hz=50.0,
)

TOPIC_ENABLE = TopicEnable()

JOINT_NAMES: tuple[str, ...] = ()
POLLING_TIMEOUT_MS: int = 10
CAMERA_ENCODE = "h265"


def build_data_collector_config(
    *,
    observation_endpoint: str,
    command_endpoint: str,
    output_dir: Path | str | None = None,
) -> DataCollectorConfig:
    """Assemble ``DataCollectorConfig`` from this module and HAL endpoint strings."""
    od = Path(output_dir).expanduser() if output_dir is not None else DEFAULT_OUTPUT_DIR
    return DataCollectorConfig(
        hal=HalEndpoints(
            observation_endpoint=observation_endpoint,
            command_endpoint=command_endpoint,
        ),
        output_dir=od,
        min_free_fraction=MIN_FREE_FRACTION,
        rotation_max_bytes=ROTATION_MAX_BYTES,
        rotation_max_minutes=ROTATION_MAX_MINUTES,
        rates=RECORDING_RATES,
        topics=TOPIC_ENABLE,
        joint_names=JOINT_NAMES,
        joints_command_source="previous_action",
        polling_timeout_ms=POLLING_TIMEOUT_MS,
        camera_encode=CAMERA_ENCODE,  # type: ignore[arg-type]
    )
