"""Short-lived TURN credentials + ICE server list for browser teleop peers.

Matches the existing teleop stack's ``GET /api/teleop-config`` shape
(``{"version":1,"iceServers":[...]}``) so the portal can swap the URL
without changing WebRTC bootstrap code.

TURN uses coturn's REST API auth (``use-auth-secret``): username is
``{expiry_unix}:{id}``, credential is ``base64(hmac-sha1(secret, username))``.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Any

from krabby_fleet_service._config import Settings

GOOGLE_STUN_URL = "stun:stun.l.google.com:19302"
DEFAULT_TURN_TTL_SECS = 3600


def mint_turn_credentials(
    secret: str,
    *,
    user_id: str = "operator",
    ttl_secs: int = DEFAULT_TURN_TTL_SECS,
    now: int | None = None,
) -> tuple[str, str, int]:
    """Return ``(username, credential, ttl_secs)`` for coturn REST API auth."""
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


def build_ice_servers(
    settings: Settings,
    *,
    user_id: str = "operator",
    now: int | None = None,
) -> dict[str, Any]:
    """Assemble the JSON body for ``GET /teleop/ice-servers``."""
    ice: list[dict[str, Any]] = [{"urls": GOOGLE_STUN_URL}]
    ttl = settings.turn_ttl_secs
    host = (settings.turn_host or "").strip()
    secret = (settings.turn_auth_secret or "").strip()

    if host and secret:
        username, credential, ttl = mint_turn_credentials(
            secret, user_id=user_id, ttl_secs=ttl, now=now
        )
        ice.append(
            {
                "urls": [
                    f"turn:{host}:3478?transport=udp",
                    f"turn:{host}:3478?transport=tcp",
                ],
                "username": username,
                "credential": credential,
            }
        )

    return {"version": 1, "iceServers": ice, "ttlSeconds": ttl}
