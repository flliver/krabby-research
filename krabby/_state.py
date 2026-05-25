"""Persistent install state: image ref and digest written after a successful pull."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

ECR_REPO = "public.ecr.aws/t7t7b3i3/krabby-locomotion"
DEFAULT_TAG = "mainline-latest"
STATE_PATH = Path.home() / ".config" / "krabby" / "state.json"


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
