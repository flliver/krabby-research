#!/usr/bin/env python3
"""Print MCU serial device path (same rules as firmware.mcu_port.default_port)."""
from __future__ import annotations

import os
import sys

# Repo root = parent of firmware/ (this file lives in firmware/scripts/)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from firmware.mcu_port import default_port  # noqa: E402


if __name__ == "__main__":
    try:
        print(default_port())
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
