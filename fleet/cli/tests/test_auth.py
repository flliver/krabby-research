"""Session cache + expiry/refresh/login branching -- no real Cognito network access.

`_interactive_login` and `_refresh` (the only functions that actually touch
pycognito/Cognito) are patched directly so these tests only exercise the
cache and branching logic in `get_access_token`.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import jwt

import krabby_fleet_cli._auth as auth
from krabby_fleet_cli._config import Config

_CONFIG = Config(
    service_url="https://fleet.example/api",
    cognito_user_pool_id="us-east-1_TEST",
    cognito_client_id="test-client",
)


def _fake_token(exp_offset: int) -> str:
    return jwt.encode(
        {"exp": int(time.time()) + exp_offset}, "unused-secret-padded-to-32-bytes!", algorithm="HS256"
    )


def test_session_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "SESSION_PATH", tmp_path / "session.json")
    auth._save_session("access", "id", "refresh")
    session = auth._load_session()
    assert session == {"access_token": "access", "id_token": "id", "refresh_token": "refresh"}


def test_load_session_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "SESSION_PATH", tmp_path / "session.json")
    assert auth._load_session() is None


def test_get_access_token_uses_valid_cached_token(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "SESSION_PATH", tmp_path / "session.json")
    token = _fake_token(exp_offset=3600)
    auth._save_session(token, "id", "refresh")

    with patch.object(auth, "_interactive_login") as login, patch.object(auth, "_refresh") as refresh:
        result = auth.get_access_token(_CONFIG)

    assert result == token
    login.assert_not_called()
    refresh.assert_not_called()


def test_get_access_token_refreshes_expired_token(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "SESSION_PATH", tmp_path / "session.json")
    expired = _fake_token(exp_offset=-10)
    auth._save_session(expired, "id", "refresh-token")

    new_token = _fake_token(exp_offset=3600)
    with patch.object(auth, "_refresh", return_value=(new_token, "new-id")) as refresh, \
         patch.object(auth, "_interactive_login") as login:
        result = auth.get_access_token(_CONFIG)

    assert result == new_token
    refresh.assert_called_once_with(_CONFIG, "refresh-token")
    login.assert_not_called()
    assert auth._load_session()["access_token"] == new_token


def test_get_access_token_falls_back_to_login_when_no_session(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "SESSION_PATH", tmp_path / "session.json")
    new_token = _fake_token(exp_offset=3600)

    with patch.object(auth, "_interactive_login", return_value=(new_token, "id", "refresh")) as login:
        result = auth.get_access_token(_CONFIG)

    assert result == new_token
    login.assert_called_once_with(_CONFIG)
    assert auth._load_session()["refresh_token"] == "refresh"


def test_get_access_token_falls_back_to_login_when_refresh_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "SESSION_PATH", tmp_path / "session.json")
    expired = _fake_token(exp_offset=-10)
    auth._save_session(expired, "id", "stale-refresh-token")

    new_token = _fake_token(exp_offset=3600)
    with patch.object(auth, "_refresh", return_value=None) as refresh, \
         patch.object(auth, "_interactive_login", return_value=(new_token, "id", "new-refresh")) as login:
        result = auth.get_access_token(_CONFIG)

    assert result == new_token
    refresh.assert_called_once()
    login.assert_called_once_with(_CONFIG)
