"""krabby update — pull a newer image and record the new digest."""
from __future__ import annotations

import sys
from typing import Optional

from krabby._docker import pull
from krabby._state import installed_image, resolve_image_ref, save_state


def cmd_update(
    image_ref: Optional[str] = None, channel: Optional[str] = None
) -> None:
    if image_ref is None and channel is None:
        image_ref = installed_image()
    try:
        ref = resolve_image_ref(image_ref, channel=channel)
    except ValueError as exc:
        sys.exit(str(exc))
    print(f"Updating Krabby locomotion stack: {ref} ...")
    digest = pull(ref)
    save_state(ref, digest)
    print(f"\n[ok]  Updated {ref}")
    if digest:
        print(f"      digest: {digest}")
