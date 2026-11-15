"""SocketCAN-FD transport helpers."""

from __future__ import annotations

from typing import Any

ARBITRATION_BITRATE = 1_000_000
DATA_BITRATE = 5_000_000

ROBOT_INTERFACES = ("can0", "can1")
VIRTUAL_INTERFACE = "vcan0"


class InProcessVirtualCanFdBus:
    """Virtual CAN-FD with separate TX/RX handles (python-can does not loop back on one Bus)."""

    def __init__(self, tx, rx) -> None:
        self._tx = tx
        self._rx = rx

    @classmethod
    def open(cls, channel: str):
        import can

        kwargs = {
            "interface": "virtual",
            "channel": channel,
            "fd": True,
            "bitrate": ARBITRATION_BITRATE,
            "data_bitrate": DATA_BITRATE,
        }
        return cls(can.interface.Bus(**kwargs), can.interface.Bus(**kwargs))

    @property
    def channel(self) -> str:
        return self._tx.channel

    @property
    def interface(self) -> str:
        return self._tx.interface

    @property
    def is_fd(self) -> bool:
        return bool(getattr(self._tx, "is_fd", False))

    def send(self, message) -> None:
        self._tx.send(message)

    def recv(self, timeout: float | None = None):
        if timeout is None:
            return self._rx.recv()
        return self._rx.recv(timeout=timeout)

    def shutdown(self) -> None:
        self._tx.shutdown()
        self._rx.shutdown()


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