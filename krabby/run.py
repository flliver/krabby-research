"""krabby run — start the locomotion stack.

Default (`krabby run`, no args): launches the full gamepad stack — HAL server +
krabby-uno client/controller — in one container, so a paired gamepad drives the
robot with a single command. `--gamepad-only` is an explicit alias for the same.

Inference: `krabby run -- --checkpoint /path/to/ckpt.pt` (or `--entrypoint ...`)
runs the HAL server with a policy checkpoint instead.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Optional

from krabby._docker import gamepad_cmd, run_cmd
from krabby._state import installed_image, resolve_image_ref


def cmd_run(
    image_ref: Optional[str] = None,
    extra_args: Optional[list[str]] = None,
    entrypoint: Optional[str] = None,
    extra_mounts: Optional[list[str]] = None,
    gamepad_only: bool = False,
) -> None:
    if image_ref is None:
        image_ref = installed_image()
    ref = resolve_image_ref(image_ref)
    extra_args = list(extra_args or [])
    # argparse REMAINDER keeps the `--` separator: `krabby run -- --checkpoint x`
    # parses to ["--", "--checkpoint", "x"]. Strip one leading `--` so it doesn't
    # reach the container, where the HAL server / krabby-uno argparse would reject it.
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    # Inference / custom path is selected by a policy checkpoint or an explicit
    # entrypoint. Everything else — including gamepad client args like --device-id —
    # runs the combined gamepad stack. --gamepad-only forces the gamepad path.
    inference = not gamepad_only and (entrypoint is not None or "--checkpoint" in extra_args)

    if inference:
        cmd = run_cmd(ref, extra_args, entrypoint=entrypoint, extra_mounts=extra_mounts)
    else:
        cmd = gamepad_cmd(ref, extra_args, extra_mounts=extra_mounts)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
