"""Map browser-style ``iceServers`` dicts to aiortc ``RTCConfiguration``."""

from __future__ import annotations

from typing import Any

from aiortc import RTCConfiguration, RTCIceServer


def rtc_configuration_from_ice_servers(
    servers: list[dict[str, Any]] | None,
) -> RTCConfiguration | None:
    """Build aiortc configuration from WebRTC ``iceServers`` entries."""
    if not servers:
        return None
    ice: list[RTCIceServer] = []
    for entry in servers[:32]:
        if not isinstance(entry, dict) or "urls" not in entry:
            continue
        urls = entry["urls"]
        if isinstance(urls, str):
            url_list = [urls]
        elif isinstance(urls, list):
            url_list = [str(u) for u in urls if u]
        else:
            continue
        if not url_list:
            continue
        kwargs: dict[str, Any] = {"urls": url_list}
        if entry.get("username") is not None:
            kwargs["username"] = str(entry["username"])
        if entry.get("credential") is not None:
            kwargs["credential"] = str(entry["credential"])
        ice.append(RTCIceServer(**kwargs))
    if not ice:
        return None
    return RTCConfiguration(iceServers=ice)
