"""krabby-fleet ssh <robot>: open a tunnel, proxy through it, run ssh, close on exit."""
from __future__ import annotations

import subprocess
from typing import Optional

from krabby_fleet_cli._api import close_ssh_tunnel, open_ssh_tunnel
from krabby_fleet_cli._auth import get_access_token
from krabby_fleet_cli._config import load_config
from krabby_fleet_cli._localproxy import free_local_port, spawn_source_proxy, wait_until_ready


def cmd_ssh(thing_name: str, user: Optional[str] = None) -> None:
    config = load_config()
    access_token = get_access_token(config)

    print(f"Opening SSH tunnel to {thing_name} ...")
    tunnel = open_ssh_tunnel(config, thing_name, access_token)
    tunnel_id = tunnel["tunnelId"]

    # Everything from here on runs inside try/finally: once the tunnel
    # exists, an interrupt or error anywhere below must still close it.
    proxy_proc = None
    try:
        local_port = free_local_port()
        proxy_proc = spawn_source_proxy(tunnel["sourceAccessToken"], tunnel["region"], local_port)
        wait_until_ready(local_port, proxy_proc)

        ssh_user = user or config.default_ssh_user
        print(f"[ok]  tunnel {tunnel_id} open -- connecting as {ssh_user}@localhost:{local_port}")
        subprocess.run(
            [
                "ssh",
                # Each session gets a fresh tunnel on a fresh local port, so
                # there's no stable "localhost:<port>" host identity to
                # check against known_hosts -- the Secure Tunnel's own
                # short-lived, Cognito-gated access token is the actual
                # security boundary here, not the SSH host key.
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-p", str(local_port),
                f"{ssh_user}@localhost",
            ]
        )
    finally:
        if proxy_proc is not None and proxy_proc.poll() is None:
            proxy_proc.terminate()
        close_ssh_tunnel(config, thing_name, tunnel_id, access_token)
        print("[ok]  tunnel closed")
