"""krabby run — start the locomotion container."""
from __future__ import annotations

import subprocess
import sys
from typing import Optional

from krabby._docker import run_cmd
from krabby._state import installed_image, resolve_image_ref


def cmd_run(image_ref: Optional[str] = None, extra_args: Optional[list[str]] = None, entrypoint: Optional[str] = None) -> None:
    if image_ref is None:
        image_ref = installed_image()
    ref = resolve_image_ref(image_ref)
    cmd = run_cmd(ref, extra_args or [], entrypoint=entrypoint)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
