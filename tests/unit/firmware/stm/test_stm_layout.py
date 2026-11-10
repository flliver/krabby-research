"""STM tree layout and package surface."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
STM = REPO_ROOT / "firmware" / "stm"

REQUIRED_PATHS = (
    STM / "core" / "board.h",
    STM / "board" / "board_hw.c",
    STM / "sim" / "board_sim.c",
    STM / "cli" / "__init__.py",
    STM / "codec.py",
    STM / "mcu_can.py",
    STM / "scripts",
)


@pytest.mark.parametrize("path", REQUIRED_PATHS, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_stm_scaffold_paths_exist(path: Path) -> None:
    assert path.exists(), path


def test_firmware_stm_package_imports() -> None:
    from firmware.stm import __version__ as v
    from firmware.stm import ids

    assert v
    assert len(ids.MCU_JOINTS) == 18
