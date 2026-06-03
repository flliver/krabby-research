"""krabby firmware — pass-through to krabby-firmware inside the locomotion container.

Distinct from the standalone `krabby-firmware` host CLI (the `firmware` package's
flash tool): this wrapper runs that same CLI *inside the installed image*, so a kit
owner who only `pip install krabby` can flash MCUs without installing flash tools
(avrdude/arduino-cli) on the host. Args are forwarded verbatim, and a TTY is
allocated when the host shell is interactive so the menu and paged output behave
the same as running `krabby-firmware` directly.
"""
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
    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    cmd = firmware_cmd(ref, firmware_args, interactive=interactive)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
