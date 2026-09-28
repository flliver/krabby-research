"""Secure Tunneling open/close -- a thin proxy over boto3's iotsecuretunneling client.

The destination access token never passes through this service or its
caller: AWS delivers it straight to the device over MQTT
(`$aws/things/{thingName}/tunnels/notify`, handled by `krabby agent`). Only
the short-lived *source* token -- meaningless without also holding valid
Cognito-authenticated access to this endpoint -- goes back to the operator.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException

from krabby_fleet_service._config import aws_region

# Matches the plan's documented default max tunnel lifetime.
_MAX_LIFETIME_MINUTES = 720


def _client() -> Any:
    import boto3

    return boto3.client("iotsecuretunneling", region_name=aws_region())


def open_ssh_tunnel(thing_name: str) -> dict[str, Any]:
    response = _client().open_tunnel(
        description=f"krabby-fleet ssh: {thing_name}",
        destinationConfig={"thingName": thing_name, "services": ["SSH"]},
        timeoutConfig={"maxLifetimeTimeoutMinutes": _MAX_LIFETIME_MINUTES},
    )
    return {
        "tunnelId": response["tunnelId"],
        "sourceAccessToken": response["sourceAccessToken"],
        "region": aws_region(),
    }


def close_ssh_tunnel(tunnel_id: str) -> None:
    """Force-close. `delete=True` also removes the tunnel record, rather than
    leaving it around in a closed-but-listable state."""
    client = _client()
    try:
        client.close_tunnel(tunnelId=tunnel_id, delete=True)
    except client.exceptions.ResourceNotFoundException as exc:
        raise HTTPException(status_code=404, detail="tunnel not found") from exc
