"""~/.config/krabby-fleet/config.toml loading."""
from __future__ import annotations

from pathlib import Path

import pytest

from krabby_fleet_cli._config import load_config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_load_config_happy_path(tmp_path):
    path = _write(
        tmp_path,
        """
        [fleet]
        service_url = "https://fleet.example/api/"

        [cognito]
        user_pool_id = "us-east-1_TEST"
        client_id = "test-client"

        [ssh]
        default_user = "pilot"
        """,
    )
    config = load_config(path)
    assert config.service_url == "https://fleet.example/api"  # trailing slash stripped
    assert config.cognito_user_pool_id == "us-east-1_TEST"
    assert config.cognito_client_id == "test-client"
    assert config.default_ssh_user == "pilot"


def test_load_config_default_ssh_user(tmp_path):
    path = _write(
        tmp_path,
        """
        [fleet]
        service_url = "https://fleet.example/api"

        [cognito]
        user_pool_id = "us-east-1_TEST"
        client_id = "test-client"
        """,
    )
    config = load_config(path)
    assert config.default_ssh_user == "operator"


def test_load_config_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        load_config(tmp_path / "does-not-exist.toml")


def test_load_config_missing_section_exits(tmp_path):
    path = _write(tmp_path, '[fleet]\nservice_url = "https://fleet.example/api"\n')
    with pytest.raises(SystemExit):
        load_config(path)


def test_load_config_missing_key_exits(tmp_path):
    path = _write(
        tmp_path,
        """
        [fleet]
        service_url = "https://fleet.example/api"

        [cognito]
        user_pool_id = "us-east-1_TEST"
        """,
    )
    with pytest.raises(SystemExit):
        load_config(path)
