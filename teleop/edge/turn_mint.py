"""Short-lived coturn REST credentials (same HMAC scheme as fleet ``_ice.py``)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from typing import Any

DEFAULT_TURN_TTL_SECS = 3600


def mint_turn_credentials(
    secret: str,
    *,
    user_id: str = "robot",
    ttl_secs: int = DEFAULT_TURN_TTL_SECS,
    now: int | None = None,
) -> tuple[str, str, int]:
    if ttl_secs < 60:
        raise ValueError("ttl_secs must be >= 60")
    expiry = int(now if now is not None else time.time()) + ttl_secs
    username = f"{expiry}:{user_id}"
    digest = hmac.new(
        secret.encode("utf-8"),
        username.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    credential = base64.b64encode(digest).decode("ascii")
    return username, credential, ttl_secs


def append_env_turn_servers(ice: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append fleet coturn when ``KRABBY_TELEOP_TURN_HOST`` + ``KRABBY_TELEOP_TURN_AUTH_SECRET`` are set."""
    host = os.environ.get("KRABBY_TELEOP_TURN_HOST", "").strip()
    secret = os.environ.get("KRABBY_TELEOP_TURN_AUTH_SECRET", "").strip()
    if not host or not secret:
        return ice
    ttl = int(os.environ.get("KRABBY_TELEOP_TURN_TTL_SECS", str(DEFAULT_TURN_TTL_SECS)))
    username, credential, _ = mint_turn_credentials(secret, ttl_secs=ttl)
    out = list(ice)
    out.append(
        {
            "urls": [
                f"turn:{host}:3478?transport=udp",
                f"turn:{host}:3478?transport=tcp",
            ],
            "username": username,
            "credential": credential,
        }
    )
    return out
