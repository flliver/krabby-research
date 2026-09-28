"""Tests for krabby-fleet teleop URL helper + portal_base_url."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from krabby_fleet_cli._config import Config, load_config, portal_base_url
from krabby_fleet_cli.teleop import cmd_teleop, teleop_url


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_portal_base_url_strips_api_suffix():
    cfg = Config(
        service_url="https://fleet.example/api",
        cognito_user_pool_id="pool",
        cognito_client_id="client",
    )
    assert portal_base_url(cfg) == "https://fleet.example"


def test_portal_base_url_override():
    cfg = Config(
        service_url="https://fleet.example/api",
        cognito_user_pool_id="pool",
        cognito_client_id="client",
        portal_url="https://portal.example",
    )
    assert portal_base_url(cfg) == "https://portal.example"


def test_load_config_portal_url(tmp_path):
    path = _write(
        tmp_path,
        """
        [fleet]
        service_url = "https://fleet.example/api"
        portal_url = "https://ui.example"

        [cognito]
        user_pool_id = "us-east-1_TEST"
        client_id = "test-client"
        """,
    )
    config = load_config(path)
    assert config.portal_url == "https://ui.example"
    assert portal_base_url(config) == "https://ui.example"


def test_teleop_url():
    assert (
        teleop_url("bench-krabby-ci", portal_url="https://fleet.example")
        == "https://fleet.example/devices/bench-krabby-ci/teleop"
    )


def test_cmd_teleop_opens_browser(tmp_path):
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
    with patch("krabby_fleet_cli.teleop.load_config", return_value=load_config(path)), patch(
        "krabby_fleet_cli.teleop.webbrowser.open", return_value=True
    ) as open_browser:
        cmd_teleop("bench-krabby-ci")
    open_browser.assert_called_once_with(
        "https://fleet.example/devices/bench-krabby-ci/teleop"
    )
