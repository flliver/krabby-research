#!/usr/bin/env python3
"""Print default MCU serial endpoint (same rules as firmware.mcu_port.default_target)."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from firmware.mcu_port import McuSerialDevice, default_target  # noqa: E402


def main() -> None:
    target = default_target()
    if not isinstance(target, McuSerialDevice):
        print(target, file=sys.stderr)
        sys.exit(
            "detect_mcu_port prints a local device path; "
            "KRABBY_MCU_PORT is set to a pyserial URL — use that URL directly."
        )
    print(target.path)


if __name__ == "__main__":
    main()
