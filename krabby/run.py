"""krabby run — start the locomotion container."""
from __future__ import annotations

import subprocess
import sys
from typing import Optional

from krabby._docker import run_cmd
from krabby._state import installed_image, resolve_image_ref


_GAMEPAD_ONLY_ENTRYPOINT = "krabby-hal-server-gamepad-only"
_GAMEPAD_ONLY_ARGS: list[str] = []


def cmd_run(image_ref: Optional[str] = None, extra_args: Optional[list[str]] = None, entrypoint: Optional[str] = None, extra_mounts: Optional[list[str]] = None, gamepad_only: bool = False) -> None:
    if image_ref is None:
        image_ref = installed_image()
    ref = resolve_image_ref(image_ref)
    if gamepad_only:
        entrypoint = _GAMEPAD_ONLY_ENTRYPOINT
        extra_args = _GAMEPAD_ONLY_ARGS + (extra_args or [])
    elif not extra_args and not entrypoint:
        print(
            "error: krabby run requires either --gamepad-only or a --checkpoint argument.\n"
            "\n"
            "  Gamepad control (no checkpoint needed):\n"
            "    krabby run --gamepad-only\n"
            "\n"
            "  Inference mode:\n"
            "    krabby run -- --checkpoint /path/to/checkpoint.pt",
            file=sys.stderr,
        )
        sys.exit(1)
    cmd = run_cmd(ref, extra_args or [], entrypoint=entrypoint, extra_mounts=extra_mounts)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
