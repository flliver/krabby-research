"""HTTP calls to krabby-fleet-service's Secure Tunneling endpoints."""
from __future__ import annotations

import sys
from typing import Any

import requests

from krabby_fleet_cli._config import Config


def list_devices(config: Config, access_token: str) -> list[dict[str, Any]]:
    resp = requests.get(
        f"{config.service_url}/devices",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if resp.status_code == 401:
        print("error: not authenticated (401) -- session may have expired", file=sys.stderr)
        sys.exit(1)
    if resp.status_code == 403:
        print(
            "error: not authorized (403) -- your account isn't in the operator group",
            file=sys.stderr,
        )
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()


def open_ssh_tunnel(config: Config, thing_name: str, access_token: str) -> dict[str, Any]:
    resp = requests.post(
        f"{config.service_url}/devices/{thing_name}/ssh-tunnel",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if resp.status_code == 401:
        print("error: not authenticated (401) -- session may have expired", file=sys.stderr)
        sys.exit(1)
    if resp.status_code == 403:
        print(
            "error: not authorized (403) -- your account isn't in the operator group",
            file=sys.stderr,
        )
        sys.exit(1)
    resp.raise_for_status()
    return resp.json()


def close_ssh_tunnel(config: Config, thing_name: str, tunnel_id: str, access_token: str) -> None:
    resp = requests.delete(
        f"{config.service_url}/devices/{thing_name}/ssh-tunnel/{tunnel_id}",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=15,
    )
    if resp.status_code not in (204, 404):
        print(f"warning: tunnel close returned {resp.status_code}: {resp.text}", file=sys.stderr)
