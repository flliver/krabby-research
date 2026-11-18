"""Defaults assembled from ``collector_settings`` (Python source, not YAML)."""

from __future__ import annotations

from pathlib import Path

from data_collection.collector_settings import (
    DEFAULT_OUTPUT_DIR,
    build_data_collector_config,
)


def test_build_data_collector_config_uses_defaults() -> None:
    cfg = build_data_collector_config(
        observation_endpoint="inproc://hal_observation",
        command_endpoint="inproc://hal_commands",
    )
    assert cfg.hal.observation_endpoint == "inproc://hal_observation"
    assert cfg.hal.command_endpoint == "inproc://hal_commands"
    assert cfg.output_dir == DEFAULT_OUTPUT_DIR
    assert cfg.rates.default_images_hz == 10.0
    assert cfg.rates.images_hz_for("belly_right_tof") == 20.0
    assert cfg.rates.images_hz_for("front_rgbd") == 10.0
    assert cfg.rotation_max_minutes == 5.0
    assert cfg.camera_encode == "h265"
    assert cfg.joints_command_source == "previous_action"


def test_build_data_collector_config_output_dir_override(tmp_path: Path) -> None:
    out = tmp_path / "bags"
    cfg = build_data_collector_config(
        observation_endpoint="inproc://a",
        command_endpoint="inproc://b",
        output_dir=out,
    )
    assert cfg.output_dir == out
