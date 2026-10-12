"""Three-board SYNC election with the leader FRONT USB port exposed as a PTY."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from firmware.sim.serial_pty_bridge import SimMcuPtyBridge
from firmware.sim.three_board_boot import (
    LEADER_BOARD_ID,
    boot_three_board_emulators,
)

if TYPE_CHECKING:
    from firmware.sim.host_bridge import HostActuatorBridge


class ThreeBoardMcuHost:
    """Runs role election for three emulators; SDK connects to FRONT USB only."""

    def __init__(self, bridge: HostActuatorBridge) -> None:
        self._bridge = bridge
        self._pty = SimMcuPtyBridge(bridge)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.slave_path: str = ""
        self.roles_by_board: dict[str, int] = {}

    def _tick_loop(self) -> None:
        while not self._stop.is_set():
            self._bridge.advance_millis(10)
            self._bridge.firmware_loop_once()
            time.sleep(0.005)

    def start(self) -> str:
        self.roles_by_board = boot_three_board_emulators(self._bridge)
        self._bridge.firmware_setup()
        self.slave_path = self._pty.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._tick_loop,
            name="three-board-mcu-host",
            daemon=True,
        )
        self._thread.start()
        return self.slave_path

    @property
    def leader_board_id(self) -> int:
        return LEADER_BOARD_ID

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
