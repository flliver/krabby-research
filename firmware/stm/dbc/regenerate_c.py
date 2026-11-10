#!/usr/bin/env python3
"""Regenerate committed C bindings from krabby_leg.dbc (cantools)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DBC_DIR = Path(__file__).resolve().parent
DBC_PATH = DBC_DIR / "krabby_leg.dbc"
OUT_DIR = DBC_DIR / "generated"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter with cantools installed (default: current interpreter)",
    )
    args = parser.parse_args()

    if not DBC_PATH.is_file():
        print(f"missing {DBC_PATH}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "-m",
        "cantools",
        "generate_c_source",
        str(DBC_PATH),
        "--database-name",
        "krabby_leg",
        "-o",
        str(OUT_DIR),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
