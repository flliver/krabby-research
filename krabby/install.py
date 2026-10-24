"""krabby install — pull the locomotion image and set up the host."""
from __future__ import annotations

import sys
from typing import Optional

from krabby._docker import pull
from krabby._host import run_host_setup
from krabby._state import resolve_image_ref, save_state


def cmd_install(
    image_ref: Optional[str] = None,
    channel: Optional[str] = None,
    launch_on_startup: bool = True,
) -> None:
    try:
        ref = resolve_image_ref(image_ref, channel=channel)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"Installing Krabby locomotion stack from {ref} ...")
    run_host_setup(launch_on_startup=launch_on_startup)
    digest = pull(ref)
    save_state(ref, digest)
    print(f"\n[ok]  Installed {ref}")
    if digest:
        print(f"      digest: {digest}")
