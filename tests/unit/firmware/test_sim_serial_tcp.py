"""pyserial ``socket://`` URL bridges to host ``Serial`` RX/TX buffers."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("serial")

from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge
from firmware.sim.serial_tcp_bridge import SimMcuTcpBridge


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


def test_pyserial_socket_url_reads_sim_tx(host_bridge):
    import serial

    with SimMcuTcpBridge(host_bridge) as tcp:
        with serial.serial_for_url(tcp.port_url, baudrate=250000, timeout=1.0) as ser:
            ser.dtr = False
            assert ser.is_open
            host_bridge.serial_usb_println("Krabby Ready")
            assert ser.readline().rstrip(b"\r\n") == b"Krabby Ready"


def test_pyserial_socket_url_write_reaches_sim_usb_rx(host_bridge):
    import serial

    with SimMcuTcpBridge(host_bridge) as tcp:
        with serial.serial_for_url(tcp.port_url, baudrate=250000, timeout=1.0) as ser:
            ser.dtr = False
            ser.write(b"H\n")
            ser.flush()
            deadline = time.monotonic() + 2.0
            rx = b""
            while time.monotonic() < deadline:
                rx = _drain_sim_usb_rx(host_bridge)
                if rx == b"H\n":
                    break
                time.sleep(0.01)
            assert rx == b"H\n"
