"""Cognito SRP authentication and a local, short-lived session cache.

Uses SRP via `pycognito` (implements the SRP-6a math Cognito's
`USER_SRP_AUTH` flow requires -- boto3's `cognito-idp` client only exposes
the raw API calls, not the client-side SRP computation). Caches tokens at
`~/.config/krabby-fleet/session.json`, refreshing with the stored refresh
token before falling back to an interactive username/password prompt.
"""
from __future__ import annotations

import getpass
import json
import sys
import time
from pathlib import Path
from typing import Optional

import jwt

from krabby_fleet_cli._config import Config

SESSION_PATH = Path.home() / ".config" / "krabby-fleet" / "session.json"

# Refresh a bit before actual expiry so an API call never races token expiration.
_EXPIRY_SKEW_SECS = 60


def _token_expiry(access_token: str) -> int:
    claims = jwt.decode(access_token, options={"verify_signature": False})
    return int(claims["exp"])


def _load_session() -> Optional[dict]:
    if not SESSION_PATH.exists():
        return None
    try:
        return json.loads(SESSION_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_session(access_token: str, id_token: str, refresh_token: str) -> None:
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(
        json.dumps(
            {
                "access_token": access_token,
                "id_token": id_token,
                "refresh_token": refresh_token,
            }
        )
    )
    SESSION_PATH.chmod(0o600)


def _interactive_login(config: Config) -> tuple[str, str, str]:
    from pycognito import Cognito

    username = input("Cognito username: ")
    password = getpass.getpass("Cognito password: ")
    user = Cognito(config.cognito_user_pool_id, config.cognito_client_id, username=username)
    try:
        user.authenticate(password=password)
    except Exception as exc:  # noqa: BLE001 - pycognito raises varied boto3/botocore errors
        print(f"error: login failed: {exc}", file=sys.stderr)
        sys.exit(1)
    return user.access_token, user.id_token, user.refresh_token


def _refresh(config: Config, refresh_token: str) -> Optional[tuple[str, str]]:
    from pycognito import Cognito

    user = Cognito(
        config.cognito_user_pool_id,
        config.cognito_client_id,
        refresh_token=refresh_token,
    )
    try:
        user.renew_access_token()
    except Exception:  # noqa: BLE001 - stale/revoked refresh token falls back to interactive login
        return None
    return user.access_token, user.id_token


def get_access_token(config: Config) -> str:
    """Returns a valid Cognito access token, refreshing or prompting as needed."""
    session = _load_session()
    if session:
        try:
            if _token_expiry(session["access_token"]) - _EXPIRY_SKEW_SECS > int(time.time()):
                return session["access_token"]
        except (jwt.PyJWTError, KeyError):
            pass

        refreshed = _refresh(config, session.get("refresh_token", ""))
        if refreshed:
            access_token, id_token = refreshed
            _save_session(access_token, id_token, session["refresh_token"])
            return access_token

    access_token, id_token, refresh_token = _interactive_login(config)
    _save_session(access_token, id_token, refresh_token)
    return access_token
