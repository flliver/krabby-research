"""PTY MCU host in single-leader 18-joint mode (no three-board election)."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from firmware.sim.serial_pty_bridge import SimMcuPtyBridge

if TYPE_CHECKING:
    from firmware.sim.host_bridge import HostActuatorBridge


class SingleLeaderEighteenMcuHost:
    """Early SDK path: one emulator, three role-prefixed telemetry lines on USB."""

    def __init__(self, bridge: HostActuatorBridge) -> None:
        self._bridge = bridge
        self._pty = SimMcuPtyBridge(bridge)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.slave_path: str = ""

    def _tick_loop(self) -> None:
        while not self._stop.is_set():
            self._bridge.advance_millis(10)
            self._bridge.firmware_loop_once()
            time.sleep(0.005)

    def start(self) -> str:
        self._bridge.enable_single_leader_eighteen_joint(True)
        self._bridge.firmware_setup()
        self.slave_path = self._pty.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._tick_loop,
            name="single-leader-mcu-host",
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
