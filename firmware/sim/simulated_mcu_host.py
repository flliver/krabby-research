"""Front-board Isaac PTY host (alias for :class:`IsaacMcuHost`)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from firmware.sim.isaac_mcu_host import IsaacMcuHost

if TYPE_CHECKING:
    from firmware.sim.robot_core import RobotCore


class SimulatedMcuHost(IsaacMcuHost):
    """PTY slave path for pyserial plus a background Isaac/firmware tick."""

    def __init__(self, core: RobotCore) -> None:
        super().__init__(core)
