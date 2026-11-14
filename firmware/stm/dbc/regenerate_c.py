#!/usr/bin/env python3
"""Regenerate committed C bindings from STM DBC files (cantools)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DBC_DIR = Path(__file__).resolve().parent
OUT_DIR = DBC_DIR / "generated"

# (path, cantools --database-name stem for generated C/H)
DATABASES = (("krabby.dbc", "krabby"),)


def _generate(python: str, dbc_name: str, db_name: str) -> None:
    dbc_path = DBC_DIR / dbc_name
    if not dbc_path.is_file():
        print(f"skip missing {dbc_path}", file=sys.stderr)
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        python,
        "-m",
        "cantools",
        "generate_c_source",
        str(dbc_path),
        "--database-name",
        db_name,
        "-o",
        str(OUT_DIR),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter with cantools installed (default: current interpreter)",
    )
    args = parser.parse_args()

    for dbc_name, db_name in DATABASES:
        _generate(args.python, dbc_name, db_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
