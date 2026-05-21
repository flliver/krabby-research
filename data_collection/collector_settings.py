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

MAX_DISK_USAGE_FRACTION: float = 0.5
ROTATION_MAX_BYTES: int = 1_073_741_824
ROTATION_MAX_MINUTES: float = 30.0

RECORDING_RATES = RecordingRates(images_hz=10.0, joints_imu_hz=50.0)

TOPIC_ENABLE = TopicEnable()

JOINT_NAMES: tuple[str, ...] = ()
POLLING_TIMEOUT_MS: int = 10


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
        max_disk_usage_fraction=MAX_DISK_USAGE_FRACTION,
        rotation_max_bytes=ROTATION_MAX_BYTES,
        rotation_max_minutes=ROTATION_MAX_MINUTES,
        rates=RECORDING_RATES,
        topics=TOPIC_ENABLE,
        joint_names=JOINT_NAMES,
        joints_command_source="previous_action",
        polling_timeout_ms=POLLING_TIMEOUT_MS,
    )
