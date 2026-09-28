"""require_operator: valid operator JWT succeeds (happy path).

Tokens are signed locally with a throwaway RSA keypair; JWKS is patched.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from krabby_fleet_service._auth import require_operator
from krabby_fleet_service._config import Settings, get_settings

_SETTINGS = Settings(
    aws_region="us-east-1",
    cognito_user_pool_id="us-east-1_TESTPOOL",
    cognito_app_client_id="test-client-id",
    iot_ats_endpoint="example-ats.iot.us-east-1.amazonaws.com",
)


@pytest.fixture(scope="module")
def keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def _token(keypair, **overrides) -> str:
    private_key, _ = keypair
    now = int(time.time())
    claims = {
        "iss": _SETTINGS.cognito_issuer,
        "token_use": "access",
        "client_id": _SETTINGS.cognito_app_client_id,
        "cognito:groups": ["operator"],
        "iat": now,
        "exp": now + 3600,
        **overrides,
    }
    return jwt.encode(claims, private_key, algorithm="RS256")


@pytest.fixture
def app(keypair):
    _, public_key = keypair
    test_app = FastAPI()

    @test_app.get("/protected")
    def protected(claims: dict = Depends(require_operator)):
        return {"ok": True}

    test_app.dependency_overrides[get_settings] = lambda: _SETTINGS

    fake_jwks_client = MagicMock()
    fake_jwks_client.get_signing_key_from_jwt.return_value = MagicMock(key=public_key)
    with patch("krabby_fleet_service._auth._jwks_client", return_value=fake_jwks_client):
        yield test_app

    test_app.dependency_overrides.clear()


def test_valid_operator_token_succeeds(app, keypair):
    resp = TestClient(app).get(
        "/protected", headers={"Authorization": f"Bearer {_token(keypair)}"}
    )
    assert resp.status_code == 200
