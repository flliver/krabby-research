"""Unit tests for data_collection.config (no rosbags)."""

from __future__ import annotations

import textwrap
from dataclasses import fields
from pathlib import Path

import pytest

from data_collection.config import DataCollectorConfig, TopicEnable, load_config

_CANONICAL_COLLECTOR_YAML = (
    Path(__file__).resolve().parents[3] / "data_collection" / "config" / "collector.yaml"
)


def test_load_canonical_collector_yaml() -> None:
    assert _CANONICAL_COLLECTOR_YAML.is_file(), _CANONICAL_COLLECTOR_YAML
    cfg = load_config(_CANONICAL_COLLECTOR_YAML)
    assert cfg.hal.observation_endpoint == "inproc://hal_observation"
    assert cfg.output_dir == Path("/data/krabby_bags")
    assert cfg.rates.images_hz == 10.0
    assert cfg.rates.joints_imu_hz == 50.0
    assert cfg.topics.joints_state is True
    assert cfg.joints_command_source == "previous_action"


def test_load_config_minimal(tmp_path: Path) -> None:
    p = tmp_path / "collector.yaml"
    out = tmp_path / "bags"
    p.write_text(
        textwrap.dedent(
            f"""
            hal:
              observation_endpoint: inproc://hal_observation
              command_endpoint: inproc://hal_commands
            output_dir: {out}
            rates:
              images_hz: 5.0
              joints_imu_hz: 20.0
            joint_names: ["j0", "j1"]
            """
        ).strip(),
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert cfg.hal.observation_endpoint == "inproc://hal_observation"
    assert cfg.hal.command_endpoint == "inproc://hal_commands"
    assert cfg.output_dir == out
    assert cfg.rates.images_hz == 5.0
    assert cfg.rates.joints_imu_hz == 20.0
    assert cfg.joint_names == ("j0", "j1")
    assert cfg.joints_command_source == "previous_action"


def test_load_config_root_not_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(p)


def test_from_dict_joints_command_source_unsupported() -> None:
    raw = {
        "hal": {
            "observation_endpoint": "inproc://a",
            "command_endpoint": "inproc://b",
        },
        "output_dir": "/tmp/x",
        "joints_command_source": "echo_from_server",
    }
    with pytest.raises(ValueError, match="unsupported"):
        DataCollectorConfig.from_dict(raw)


def test_from_dict_missing_hal_key() -> None:
    with pytest.raises(KeyError):
        DataCollectorConfig.from_dict({"output_dir": "/tmp/x"})


def test_from_dict_topics_all_false() -> None:
    raw = {
        "hal": {
            "observation_endpoint": "inproc://a",
            "command_endpoint": "inproc://b",
        },
        "output_dir": "/tmp/x",
        "topics": {f.name: False for f in fields(TopicEnable)},
    }
    cfg = DataCollectorConfig.from_dict(raw)
    assert cfg.topics.joints_state is False
    assert cfg.topics.imu is False
