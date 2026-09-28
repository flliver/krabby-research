"""krabby run — start the locomotion stack.

Default for **fleet-enrolled** hosts (``/etc/krabby/iot/``): HAL in portal or
inference mode with ``--teleop-ip 127.0.0.1`` so fleet portal teleop always works.

Default for **non-enrolled** hosts: gamepad stack (HAL + krabby-uno).

Inference: ``krabby run -- --checkpoint /path/to/ckpt.pt`` (enrolled hosts still
get fleet teleop flags merged from ``/etc/krabby/locomotion.json``).
"""
from __future__ import annotations

import subprocess
import sys
from typing import Optional

from krabby._docker import fleet_hal_cmd, gamepad_cmd, run_cmd
from krabby._locomotion_config import build_hal_argv, fleet_enrolled, fleet_volume_mounts, load_config
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
    inference_explicit = not gamepad_only and (
        entrypoint is not None or "--checkpoint" in extra_args
    )
    enrolled = fleet_enrolled()

    if gamepad_only:
        cmd = gamepad_cmd(ref, extra_args, extra_mounts=extra_mounts)
    # Fleet-enrolled (`/etc/krabby/iot/`): HAL + `--teleop-ip` from locomotion.json.
    elif enrolled:
        cfg = load_config()
        hal_argv = build_hal_argv(extra_args)
        flat = fleet_volume_mounts(cfg)
        cmd = fleet_hal_cmd(ref, hal_argv, flat_mounts=flat, extra_mounts=extra_mounts)
    elif inference_explicit:
        cmd = run_cmd(ref, extra_args, entrypoint=entrypoint, extra_mounts=extra_mounts)
    else:
        cmd = gamepad_cmd(ref, extra_args, extra_mounts=extra_mounts)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)
