"""HTTP calls to krabby-fleet-service: mocked requests, no real network access."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from krabby_fleet_cli._api import close_ssh_tunnel, list_devices, open_ssh_tunnel
from krabby_fleet_cli._config import Config

_CONFIG = Config(
    service_url="https://fleet.example/api",
    cognito_user_pool_id="us-east-1_TEST",
    cognito_client_id="test-client",
)


def test_list_devices_happy_path():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = [
        {"thingName": "bench-krabby-ci", "connected": True, "connectivityTimestamp": 123},
    ]
    with patch("krabby_fleet_cli._api.requests.get", return_value=fake_resp) as get:
        result = list_devices(_CONFIG, "access-token")

    assert result == [{"thingName": "bench-krabby-ci", "connected": True, "connectivityTimestamp": 123}]
    args, kwargs = get.call_args
    assert args[0] == "https://fleet.example/api/devices"
    assert kwargs["headers"] == {"Authorization": "Bearer access-token"}


def test_open_ssh_tunnel_happy_path():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {
        "tunnelId": "abc123",
        "sourceAccessToken": "src-token",
        "region": "us-east-1",
    }
    with patch("krabby_fleet_cli._api.requests.post", return_value=fake_resp) as post:
        result = open_ssh_tunnel(_CONFIG, "bench-krabby-ci", "access-token")

    assert result == {"tunnelId": "abc123", "sourceAccessToken": "src-token", "region": "us-east-1"}
    args, kwargs = post.call_args
    assert args[0] == "https://fleet.example/api/devices/bench-krabby-ci/ssh-tunnel"
    assert kwargs["headers"] == {"Authorization": "Bearer access-token"}


def test_close_ssh_tunnel_calls_delete():
    fake_resp = MagicMock(status_code=204)
    with patch("krabby_fleet_cli._api.requests.delete", return_value=fake_resp) as delete:
        close_ssh_tunnel(_CONFIG, "bench-krabby-ci", "abc123", "access-token")

    args, kwargs = delete.call_args
    assert args[0] == "https://fleet.example/api/devices/bench-krabby-ci/ssh-tunnel/abc123"
    assert kwargs["headers"] == {"Authorization": "Bearer access-token"}


def test_close_ssh_tunnel_404_does_not_raise():
    fake_resp = MagicMock(status_code=404, text="not found")
    with patch("krabby_fleet_cli._api.requests.delete", return_value=fake_resp):
        close_ssh_tunnel(_CONFIG, "bench-krabby-ci", "abc123", "access-token")  # no exception
