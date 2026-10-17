"""Open the simulated MCU USB link for pyserial (PTY on Unix, TCP elsewhere)."""

from __future__ import annotations

import os
from typing import Protocol

from firmware.sim.host_bridge import HostActuatorBridge
from firmware.sim.serial_pty_bridge import SimMcuPtyBridge
from firmware.sim.serial_tcp_bridge import SimMcuTcpBridge


class SimMcuUsbBridge(Protocol):
    def start(self) -> str: ...
    def close(self) -> None: ...


def open_sim_mcu_usb_bridge(bridge: HostActuatorBridge) -> tuple[SimMcuUsbBridge, str]:
    """Return ``(handle, port_spec)`` where ``port_spec`` is a device path or ``socket://`` URL."""
    if hasattr(os, "openpty"):
        pty = SimMcuPtyBridge(bridge)
        return pty, pty.start()
    tcp = SimMcuTcpBridge(bridge)
    return tcp, tcp.start()
