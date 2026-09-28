"""Unit tests for TURN REST API credential minting and ICE server list."""
from __future__ import annotations

import base64
import hashlib
import hmac

import pytest
from fastapi.testclient import TestClient

from krabby_fleet_service._auth import require_operator
from krabby_fleet_service._config import Settings, get_settings
from krabby_fleet_service._ice import GOOGLE_STUN_URL, build_ice_servers, mint_turn_credentials
from krabby_fleet_service.app import app

_SETTINGS = Settings(
    aws_region="us-east-1",
    cognito_user_pool_id="us-east-1_TESTPOOL",
    cognito_app_client_id="test-client-id",
    iot_ats_endpoint="example-ats.iot.us-east-1.amazonaws.com",
    turn_auth_secret="test-turn-secret",
    turn_host="fleet.example.com",
    turn_ttl_secs=3600,
)
_FAKE_CLAIMS = {"sub": "operator-sub-1", "cognito:groups": ["operator"]}


class TestMintTurnCredentials:
    def test_username_is_expiry_colon_id(self) -> None:
        username, credential, ttl = mint_turn_credentials(
            "secret", user_id="alice", ttl_secs=3600, now=1_700_000_000
        )
        assert ttl == 3600
        assert username == "1700003600:alice"
        expected = base64.b64encode(
            hmac.new(b"secret", username.encode(), hashlib.sha1).digest()
        ).decode("ascii")
        assert credential == expected

    def test_ttl_too_short_rejected(self) -> None:
        with pytest.raises(ValueError):
            mint_turn_credentials("secret", ttl_secs=30)


class TestBuildIceServers:
    def test_includes_stun_and_turn(self) -> None:
        body = build_ice_servers(_SETTINGS, user_id="op", now=1_700_000_000)
        assert body["version"] == 1
        assert body["ttlSeconds"] == 3600
        assert body["iceServers"][0] == {"urls": GOOGLE_STUN_URL}
        turn = body["iceServers"][1]
        assert turn["username"] == "1700003600:op"
        assert "turn:fleet.example.com:3478?transport=udp" in turn["urls"]
        assert "turn:fleet.example.com:3478?transport=tcp" in turn["urls"]
        assert turn["credential"]

    def test_stun_only_without_secret(self) -> None:
        settings = Settings(
            aws_region="us-east-1",
            cognito_user_pool_id="us-east-1_TESTPOOL",
            cognito_app_client_id="test-client-id",
            iot_ats_endpoint="example-ats.iot.us-east-1.amazonaws.com",
            turn_host="fleet.example.com",
            turn_auth_secret="",
        )
        body = build_ice_servers(settings)
        assert len(body["iceServers"]) == 1
        assert body["iceServers"][0]["urls"] == GOOGLE_STUN_URL


@pytest.fixture
def authed_client():
    app.dependency_overrides[get_settings] = lambda: _SETTINGS
    app.dependency_overrides[require_operator] = lambda: _FAKE_CLAIMS
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_get_ice_servers_authed(authed_client):
    resp = authed_client.get("/teleop/ice-servers")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == 1
    assert body["iceServers"][0]["urls"] == GOOGLE_STUN_URL
    assert body["iceServers"][1]["username"].endswith(":operator-sub-1")
