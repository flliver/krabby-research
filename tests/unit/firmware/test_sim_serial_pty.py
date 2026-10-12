"""pyserial on a PTY bridges to host ``Serial`` RX/TX buffers."""

from __future__ import annotations

import os
import time

import pytest

pytest.importorskip("serial")

from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.serial_pty_bridge import SimMcuPtyBridge


@pytest.fixture
def host_bridge():
    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        pytest.skip(str(exc))
    bridge.reset()
    return bridge


def _drain_sim_usb_rx(bridge, max_bytes: int = 64) -> bytes:
    out = bytearray()
    for _ in range(max_bytes):
        if bridge.serial_usb_rx_available() <= 0:
            break
        byte = bridge.serial_usb_rx_read()
        if byte < 0:
            break
        out.append(byte)
    return bytes(out)


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_pyserial_open_and_reads_sim_tx(host_bridge):
    import serial

    with SimMcuPtyBridge(host_bridge) as pty:
        with serial.Serial(pty.slave_path, 250000, timeout=1.0) as ser:
            assert ser.is_open
            host_bridge.serial_usb_println("Krabby Ready")
            assert ser.readline().rstrip(b"\r\n") == b"Krabby Ready"


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="PTY requires Unix")
def test_pyserial_write_reaches_sim_usb_rx(host_bridge):
    import serial

    with SimMcuPtyBridge(host_bridge) as pty:
        with serial.Serial(pty.slave_path, 250000, timeout=1.0) as ser:
            ser.write(b"H\n")
            ser.flush()
            deadline = time.monotonic() + 1.0
            rx = b""
            while time.monotonic() < deadline:
                rx = _drain_sim_usb_rx(host_bridge)
                if rx == b"H\n":
                    break
                time.sleep(0.01)
            assert rx == b"H\n"
