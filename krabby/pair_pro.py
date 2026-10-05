"""krabby pair-pro — pair a Nintendo Switch Pro Controller over Bluetooth."""
from __future__ import annotations

import subprocess
import sys
from importlib import resources


def cmd_pair_pro() -> None:
    """Run the packaged pairing script (needs bluetoothctl/btmon; typically sudo)."""
    ref = resources.files("krabby").joinpath("scripts/pair_pro_controller.sh")
    with resources.as_file(ref) as path:
        if not path.is_file():
            print(f"[err] pairing script missing: {path}", file=sys.stderr)
            sys.exit(1)
        # Keep the script alive for the full run (zip wheels extract under as_file).
        raise SystemExit(subprocess.call(["bash", str(path)]))
