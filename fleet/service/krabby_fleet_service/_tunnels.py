"""Secure Tunneling open/close -- a thin proxy over boto3's iotsecuretunneling client.

Prefers reusing an already-OPEN tunnel for the thing (RotateTunnelAccessToken,
no Secure Tunneling charge) over OpenTunnel ($1 per call). The destination
access token never passes through this service or its caller: AWS delivers it
straight to the device over MQTT (`$aws/things/{thingName}/tunnels/notify`,
handled by `krabby agent`). Only the short-lived *source* token -- meaningless
without also holding valid Cognito-authenticated access to this endpoint --
goes back to the operator.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import HTTPException

from krabby_fleet_service._config import aws_region

# Matches the plan's documented default max tunnel lifetime.
_MAX_LIFETIME_MINUTES = 720
_DEST_SERVICES = ["SSH"]

logger = logging.getLogger(__name__)


def _client() -> Any:
    import boto3

    return boto3.client("iotsecuretunneling", region_name=aws_region())


def _destination_config(thing_name: str) -> dict[str, Any]:
    return {"thingName": thing_name, "services": list(_DEST_SERVICES)}


def _find_open_tunnel_id(client: Any, thing_name: str) -> Optional[str]:
    """Newest OPEN tunnel for thing, if any. Closes older OPEN duplicates."""
    response = client.list_tunnels(thingName=thing_name, maxResults=100)
    open_ids = [
        summary["tunnelId"]
        for summary in response.get("tunnelSummaries") or []
        if summary.get("status") == "OPEN" and summary.get("tunnelId")
    ]
    if not open_ids:
        return None
    keep, extras = open_ids[0], open_ids[1:]
    for tunnel_id in extras:
        try:
            client.close_tunnel(tunnelId=tunnel_id, delete=True)
            logger.info("closed duplicate OPEN tunnel %s for %s", tunnel_id, thing_name)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            logger.warning("failed to close duplicate tunnel %s", tunnel_id, exc_info=True)
    return keep


def _rotate_source_token(client: Any, thing_name: str, tunnel_id: str) -> str:
    """Rotate source+destination tokens; return new sourceAccessToken.

    DESTINATION (and ALL) rotation re-notifies the device over MQTT so the
    agent can spawn destination localproxy with a fresh one-shot token.
    """
    response = client.rotate_tunnel_access_token(
        tunnelId=tunnel_id,
        clientMode="ALL",
        destinationConfig=_destination_config(thing_name),
    )
    token = response.get("sourceAccessToken")
    if not token:
        raise RuntimeError(f"RotateTunnelAccessToken returned no sourceAccessToken for {tunnel_id}")
    return token


def open_ssh_tunnel(thing_name: str) -> dict[str, Any]:
    client = _client()
    dest = _destination_config(thing_name)

    existing_id = _find_open_tunnel_id(client, thing_name)
    if existing_id is not None:
        try:
            source_token = _rotate_source_token(client, thing_name, existing_id)
            logger.info("reused OPEN tunnel %s for %s", existing_id, thing_name)
            return {
                "tunnelId": existing_id,
                "sourceAccessToken": source_token,
                "region": aws_region(),
            }
        except Exception:  # noqa: BLE001 — fall through to OpenTunnel
            logger.warning(
                "RotateTunnelAccessToken failed for %s; opening a new tunnel",
                existing_id,
                exc_info=True,
            )
            try:
                client.close_tunnel(tunnelId=existing_id, delete=True)
            except Exception:  # noqa: BLE001 — best-effort; OpenTunnel still proceeds
                logger.warning("failed to close unusable tunnel %s", existing_id, exc_info=True)

    response = client.open_tunnel(
        description=f"krabby-fleet ssh: {thing_name}",
        destinationConfig=dest,
        timeoutConfig={"maxLifetimeTimeoutMinutes": _MAX_LIFETIME_MINUTES},
    )
    return {
        "tunnelId": response["tunnelId"],
        "sourceAccessToken": response["sourceAccessToken"],
        "region": aws_region(),
    }


def close_ssh_tunnel(tunnel_id: str) -> None:
    """Force-close. `delete=True` also removes the tunnel record, rather than
    leaving it around in a closed-but-listable state.

    Normal SSH sessions should leave the tunnel OPEN so the next open can
    RotateTunnelAccessToken instead of paying for another OpenTunnel.
    """
    client = _client()
    try:
        client.close_tunnel(tunnelId=tunnel_id, delete=True)
    except client.exceptions.ResourceNotFoundException as exc:
        raise HTTPException(status_code=404, detail="tunnel not found") from exc
