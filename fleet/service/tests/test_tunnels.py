"""POST/DELETE ssh-tunnel routes: mocked iotsecuretunneling client, no real AWS calls.

These tests override the require_operator dependency directly (bypassing
real JWT verification) so they stay focused on the tunnel logic and the AWS
calls it makes, rather than re-deriving auth-token coverage.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from krabby_fleet_service._auth import require_operator
from krabby_fleet_service._config import Settings, get_settings
from krabby_fleet_service.app import app

_SETTINGS = Settings(
    aws_region="us-east-2",
    cognito_user_pool_id="us-east-2_TESTPOOL",
    cognito_app_client_id="test-client-id",
    iot_ats_endpoint="example-ats.iot.us-east-2.amazonaws.com",
)
_FAKE_CLAIMS = {"sub": "test-operator", "cognito:groups": ["operator"]}


@pytest.fixture
def authed_client():
    app.dependency_overrides[get_settings] = lambda: _SETTINGS
    app.dependency_overrides[require_operator] = lambda: _FAKE_CLAIMS
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_create_ssh_tunnel_calls_open_tunnel(authed_client):
    fake_client = MagicMock()
    fake_client.open_tunnel.return_value = {
        "tunnelId": "abc123",
        "sourceAccessToken": "src-token",
    }
    with patch("krabby_fleet_service._tunnels._client", return_value=fake_client):
        resp = authed_client.post("/devices/bench-krabby-ci/ssh-tunnel")

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"tunnelId": "abc123", "sourceAccessToken": "src-token", "region": "us-east-2"}

    fake_client.open_tunnel.assert_called_once()
    _, kwargs = fake_client.open_tunnel.call_args
    assert kwargs["destinationConfig"] == {"thingName": "bench-krabby-ci", "services": ["SSH"]}


def test_delete_ssh_tunnel_calls_close_tunnel(authed_client):
    fake_client = MagicMock()
    with patch("krabby_fleet_service._tunnels._client", return_value=fake_client):
        resp = authed_client.delete("/devices/bench-krabby-ci/ssh-tunnel/abc123")

    assert resp.status_code == 204
    fake_client.close_tunnel.assert_called_once_with(tunnelId="abc123", delete=True)


def test_healthz_needs_no_auth():
    app.dependency_overrides.clear()
    resp = TestClient(app).get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body.get("mqttConnected") is True
