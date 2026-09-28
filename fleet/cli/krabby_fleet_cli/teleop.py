"""krabby-fleet teleop <robot>: open the portal teleop view in a browser."""
from __future__ import annotations

import webbrowser

from krabby_fleet_cli._config import load_config, portal_base_url


def teleop_url(thing_name: str, *, portal_url: str) -> str:
    """Same URL the portal \"Open teleop\" button uses."""
    base = portal_url.rstrip("/")
    return f"{base}/devices/{thing_name}/teleop"


def cmd_teleop(thing_name: str) -> None:
    config = load_config()
    url = teleop_url(thing_name, portal_url=portal_base_url(config))
    print(f"Opening teleop for {thing_name} ...")
    print(f"  {url}")
    if not webbrowser.open(url):
        print("Could not open a browser; open the URL above manually.")
