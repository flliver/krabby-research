"""Persistent install state: image ref and digest written after a successful pull."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

ECR_REPO = "public.ecr.aws/t7t7b3i3/krabby-locomotion"
# Kit owners track the stable release line by default. CI moves the
# `release-latest` tag to the newest release/* build (see publish-locomotion.yml),
# mirroring `krabby-firmware update`'s "latest non-mainline release" behavior.
DEFAULT_TAG = "release-latest"


def _state_home() -> Path:
    """Home directory that holds the install-state file.

    `krabby install` needs sudo (udev rules, dialout), but `run`/`update`/`firmware`
    run unprivileged. Resolve to the invoking user's home via SUDO_USER under sudo so
    the state written by `sudo krabby install` is the same file the user's later
    commands read — not root's home. Mirrors _host._ensure_dialout's SUDO_USER use.
    """
    sudo_user = os.environ.get("SUDO_USER")
    if sudo_user:
        try:
            import pwd
            return Path(pwd.getpwnam(sudo_user).pw_dir)
        except (KeyError, ImportError):
            pass
    return Path.home()


STATE_PATH = _state_home() / ".config" / "krabby" / "state.json"


def resolve_image_ref(ref: Optional[str] = None) -> str:
    """Return a fully-qualified image URI.

    - None → ECR_REPO:DEFAULT_TAG
    - bare tag (no '/' or ':' prefix)  → ECR_REPO:<tag>
    - already-qualified URI            → returned as-is
    """
    if ref is None:
        return f"{ECR_REPO}:{DEFAULT_TAG}"
    if "/" not in ref and ":" not in ref:
        return f"{ECR_REPO}:{ref}"
    return ref


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_state(image_ref: str, digest: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps({"image_ref": image_ref, "digest": digest}, indent=2))


def installed_image() -> Optional[str]:
    return load_state().get("image_ref")
