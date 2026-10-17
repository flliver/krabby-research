"""Bridge host pyserial to the simulated USB ``Serial`` port via a POSIX PTY."""

from __future__ import annotations

import os
import select
import sys
import threading
from typing import TYPE_CHECKING

from firmware.sim.serial_usb_relay import relay_usb_serial_once

if TYPE_CHECKING:
    from firmware.sim.host_bridge import HostActuatorBridge


class SimMcuPtyBridge:
    """Expose ``Serial`` (USB) as a PTY slave path pyserial can open."""

    def __init__(self, bridge: HostActuatorBridge) -> None:
        if not hasattr(os, "openpty"):
            raise RuntimeError(
                "PTY bridge requires Unix (Linux/macOS). "
                "On Windows use firmware.sim.serial_tcp_bridge.SimMcuTcpBridge "
                "or firmware/sim/VIRTUAL_SERIAL.md."
            )
        self._bridge = bridge
        self._master_fd: int | None = None
        self._slave_fd: int | None = None
        self.slave_path: str = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> str:
        """Create the PTY, start the relay thread, return the slave device path."""
        master, slave = os.openpty()
        self._master_fd = master
        self._slave_fd = slave
        self.slave_path = os.ttyname(slave)
        self._stop.clear()
        self._thread = threading.Thread(target=self._relay_loop, name="sim-mcu-pty", daemon=True)
        self._thread.start()
        return self.slave_path

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for fd in (self._master_fd, self._slave_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._master_fd = None
        self._slave_fd = None
        self.slave_path = ""

    def _relay_loop(self) -> None:
        assert self._master_fd is not None
        master = self._master_fd
        bridge = self._bridge
        while not self._stop.is_set():
            readable, _, _ = select.select([master], [], [], 0.05)
            if master in readable:
                try:
                    relay_usb_serial_once(
                        bridge,
                        read_client=lambda n: os.read(master, n),
                        write_client=lambda data: os.write(master, data),
                    )
                except OSError:
                    break
            else:
                relay_usb_serial_once(
                    bridge,
                    read_client=lambda _n: b"",
                    write_client=lambda data: os.write(master, data),
                )

    def __enter__(self) -> SimMcuPtyBridge:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def main() -> int:
    from firmware.sim.host_bridge import HostLibraryNotBuiltError, load_host_bridge

    try:
        bridge = load_host_bridge()
    except HostLibraryNotBuiltError as exc:
        print(exc, file=sys.stderr)
        return 1
    bridge.reset()
    with SimMcuPtyBridge(bridge) as pty:
        print(pty.slave_path, flush=True)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
