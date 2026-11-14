"""SocketCAN-FD transport helpers."""

from __future__ import annotations

from typing import Any

ARBITRATION_BITRATE = 1_000_000
DATA_BITRATE = 5_000_000

ROBOT_INTERFACES = ("can0", "can1")
VIRTUAL_INTERFACE = "vcan0"


def default_interfaces() -> tuple[str, ...]:
    import os

    raw = os.environ.get("KRABBY_CAN_INTERFACES", "can0")
    return tuple(iface.strip() for iface in raw.split(",") if iface.strip())


def interfaces_for_mode(*, virtual: bool) -> tuple[str, ...]:
    if virtual:
        import os

        return (os.environ.get("KRABBY_CAN_IFACE", VIRTUAL_INTERFACE),)
    return ROBOT_INTERFACES


def open_socketcan_fd(channel: str, **extra: Any):
    """Open a CAN-FD SocketCAN interface (1 Mbit/s arbitration, 5 Mbit/s data)."""
    import can

    return can.interface.Bus(
        channel=channel,
        interface="socketcan",
        fd=True,
        bitrate=ARBITRATION_BITRATE,
        data_bitrate=DATA_BITRATE,
        **extra,
    )


def send_fd_frame(bus, arbitration_id: int, data: bytes) -> None:
    import can

    message = can.Message(
        arbitration_id=arbitration_id,
        data=data,
        is_fd=True,
        bitrate_switch=True,
    )
    bus.send(message)