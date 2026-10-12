"""Isaac ``RobotCore`` / ``ThreeBoardRobotCore`` + PTY for KrabbyMCUSDK and the GUI."""

from __future__ import annotations

import inspect
import threading
import time
from typing import Any

from firmware.sim.serial_pty_bridge import SimMcuPtyBridge


class IsaacMcuHost:
    """Background physics/firmware loop; USB telemetry includes Task-2 current sense."""

    def __init__(self, core: Any) -> None:
        self._core = core
        self._pty = SimMcuPtyBridge(core.bridge)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.slave_path: str = ""

    def _step_once(self) -> None:
        step = self._core.step
        params = inspect.signature(step).parameters
        if "process_usb_serial" in params:
            step(process_usb_serial=True)
        else:
            step()

    def _tick_loop(self) -> None:
        while not self._stop.is_set():
            self._step_once()
            time.sleep(0.001)

    def start(self) -> str:
        self._core.reset()
        self._core.bridge.firmware_setup()
        self.slave_path = self._pty.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._tick_loop,
            name="isaac-mcu-host",
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
