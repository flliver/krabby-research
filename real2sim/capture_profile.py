#!/usr/bin/env python3
"""STO-SCN-091 — camera profile at ingest (EXIF/capture-mode -> camera model).

Decide the camera model a solver should use from *known capture metadata*, never
from pixel inference (STO-SCN-096 conclusion #3; HUG-SCN-004). The camera model is
a property of the camera + capture mode, not the scene.

Key fact (T-001): the distinguishing input — fisheye vs dewarped — is NOT carried
in EXIF on DJI footage, and extracted video frames usually carry no EXIF at all.
So `mode` MUST be declared per scene. EXIF (make/model) is best-effort
corroboration only. Unknown {make, model, mode} -> fail loud, never guess.

This module is pure + importable + standalone-testable. It does NOT write the v4
store; the ingest-time emission of the resolved profile into scene metadata is a
separate, store-writer-mediated step (HUG-SCN-005 #11).

CLI:
    python capture_profile.py --make DJI --model "DJI Action 3" --mode fisheye
    python capture_profile.py --image path/to/frame.jpg --mode dewarped
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

DEFAULT_REGISTRY = Path(__file__).with_name("capture_profiles.json")


class ProfileError(ValueError):
    """Raised when a capture profile cannot be resolved (unknown camera/mode, or
    a required input — notably `mode` — is missing). Fail loud, never guess."""


def _norm(s: str | None) -> str:
    return (s or "").strip().lower()


def load_registry(path: str | os.PathLike | None = None) -> list[dict]:
    p = Path(path) if path else DEFAULT_REGISTRY
    data = json.loads(Path(p).read_text())
    return data.get("profiles", [])


def read_exif(image_path: str | os.PathLike) -> dict:
    """Best-effort EXIF read for corroboration. Returns {} when nothing is
    available (the common case for ffmpeg-extracted frames). Tries Pillow first,
    then `exiftool` if on PATH. Never raises — EXIF is corroboration, not the
    decision input."""
    p = Path(image_path)
    if not p.exists():
        return {}
    # 1) Pillow (if installed)
    try:
        from PIL import Image, ExifTags  # type: ignore

        img = Image.open(p)
        raw = getattr(img, "_getexif", lambda: None)() or {}
        tagmap = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
        make, model = tagmap.get("Make"), tagmap.get("Model")
        if make or model:
            return {"make": make, "model": model, "_source": "pillow"}
    except Exception:
        pass
    # 2) exiftool (if on PATH)
    try:
        out = subprocess.run(
            ["exiftool", "-j", "-Make", "-Model", str(p)],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            rec = (json.loads(out.stdout) or [{}])[0]
            if rec.get("Make") or rec.get("Model"):
                return {"make": rec.get("Make"), "model": rec.get("Model"),
                        "_source": "exiftool"}
    except Exception:
        pass
    return {}


def resolve(make: str | None, model: str | None, mode: str | None,
            registry: list[dict] | None = None) -> dict:
    """Resolve a capture profile from camera identity + capture mode.

    Raises ProfileError (fail loud) when `mode` is missing or no profile matches.
    Returns the matched profile dict (a copy, with a `resolved` provenance stub)."""
    if not _norm(mode):
        raise ProfileError(
            "capture_mode is required and is NOT derivable from EXIF — declare it "
            "per scene (e.g. 'fisheye' or 'dewarped'). Refusing to guess.")
    reg = registry if registry is not None else load_registry()
    for prof in reg:
        if (_norm(prof.get("make")) == _norm(make)
                and _norm(prof.get("model")) == _norm(model)
                and _norm(prof.get("mode")) == _norm(mode)):
            out = dict(prof)
            out["resolved"] = {"make": make, "model": model, "mode": mode}
            return out
    raise ProfileError(
        f"no capture profile for make={make!r} model={model!r} mode={mode!r}. "
        f"Add it to {DEFAULT_REGISTRY.name} (seed from the camera's CAPTURE-LESSONS "
        f"+ HUG-SCN-004) — do not default to a guessed camera model.")


def resolve_for_image(image_path: str | os.PathLike, mode: str | None,
                      make: str | None = None, model: str | None = None,
                      registry: list[dict] | None = None) -> dict:
    """Convenience: corroborate make/model from EXIF when not given explicitly,
    then resolve. `mode` is still required (not in EXIF)."""
    if not (make and model):
        exif = read_exif(image_path)
        make = make or exif.get("make")
        model = model or exif.get("model")
    return resolve(make, model, mode, registry=registry)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resolve a capture profile -> camera model.")
    ap.add_argument("--make")
    ap.add_argument("--model")
    ap.add_argument("--mode", help="fisheye | dewarped (required; not in EXIF)")
    ap.add_argument("--image", help="optional frame to read EXIF make/model from")
    ap.add_argument("--registry", help="override registry path")
    a = ap.parse_args(argv)
    reg = load_registry(a.registry) if a.registry else None
    try:
        if a.image:
            prof = resolve_for_image(a.image, a.mode, a.make, a.model, registry=reg)
        else:
            prof = resolve(a.make, a.model, a.mode, registry=reg)
    except ProfileError as e:
        print(f"ProfileError: {e}")
        return 2
    print(json.dumps(prof, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
