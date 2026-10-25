"""Byte relay between pyserial and host ``Serial`` USB shim buffers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from firmware.sim.host_bridge import HostActuatorBridge


def relay_usb_serial_once(
    bridge: HostActuatorBridge,
    *,
    read_client: Callable[[int], bytes],
    write_client: Callable[[bytes], None],
    max_read: int = 4096,
) -> None:
    """Move one round of bytes host ``Serial`` <-> client transport."""
    chunk = read_client(max_read)
    if chunk:
        bridge.serial_usb_rx_write(chunk)
    tx = bytearray()
    while bridge.serial_usb_tx_available() > 0:
        byte = bridge.serial_usb_tx_read()
        if byte < 0:
            break
        tx.append(byte)
    if tx:
        write_client(bytes(tx))
