"""krabby firmware — pass-through to krabby-firmware inside the locomotion container."""
from __future__ import annotations

import subprocess
import sys
from typing import Optional

from krabby._docker import firmware_cmd
from krabby._state import installed_image, resolve_image_ref


def cmd_firmware(firmware_args: list[str], image_ref: Optional[str] = None) -> None:
    if image_ref is None:
        image_ref = installed_image()
    ref = resolve_image_ref(image_ref)
    cmd = firmware_cmd(ref, firmware_args)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
