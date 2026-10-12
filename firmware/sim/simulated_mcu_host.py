"""Run host firmware + PTY so ``KrabbyMCUSDK`` can use a serial port path."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from firmware.sim.serial_pty_bridge import SimMcuPtyBridge

if TYPE_CHECKING:
    from firmware.sim.robot_core import RobotCore


class SimulatedMcuHost:
    """PTY slave path for pyserial plus a background Isaac/firmware tick."""

    def __init__(self, core: RobotCore) -> None:
        self._core = core
        self._pty = SimMcuPtyBridge(core.bridge)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.slave_path: str = ""

    def _tick_loop(self) -> None:
        while not self._stop.is_set():
            self._core.step(process_usb_serial=True)
            time.sleep(0.001)

    def start(self) -> str:
        self._core.reset()
        self._core.bridge.firmware_setup()
        self.slave_path = self._pty.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._tick_loop,
            name="simulated-mcu-host",
            daemon=True,
        )
        self._thread.start()
        return self.slave_path

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._pty.close()
        self.slave_path = ""

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()
