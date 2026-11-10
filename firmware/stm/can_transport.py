"""SocketCAN-FD transport helpers."""

from __future__ import annotations


def default_interfaces() -> tuple[str, ...]:
    import os

    raw = os.environ.get("KRABBY_CAN_INTERFACES", "can0")
    return tuple(iface.strip() for iface in raw.split(",") if iface.strip())
