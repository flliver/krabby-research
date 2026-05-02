"""Per-variant manifest: capture the inputs/config that produced a MAtCha mesh.

A manifest lives at `data/scenes/<variant>/manifest.json`. The rating app reads
these to display "what settings were used" alongside each render. Future MAtCha
runs should write a manifest as they kick off; the four existing curated
variants are backfilled by hand from journal evidence.

Schema v1 (see `MANIFEST_SCHEMA_V1` for the full skeleton). Unknown fields are
allowed `null`; downstream consumers are expected to render "—" or "(unknown)".

Authoritative source of truth: this file.
"""
from __future__ import annotations

import json
import os
from typing import Any

SCHEMA_VERSION = 1


def empty_manifest() -> dict[str, Any]:
    """Return a fully-populated skeleton with all fields set to safe defaults.

    Backfill / runtime code overrides individual fields. Anything still `null`
    after writing means "we don't know" — the rating UI surfaces this honestly.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        # Identity
        "variant_name": None,            # e.g. "12-strong"
        "scene": None,                   # e.g. "004-sky-house-dining"
        "captured_at": None,             # ISO-8601 timestamp the run started
        # Frame selection
        "frames": {
            "count": None,
            "basenames": [],             # ["frame_0013.jpg", ...]
            "source_dir": None,          # where the candidate pool lives
            "selection_method": None,    # "manual-viewer" | "even-time" | "every-Nth" | ...
            "viewer_filter": None,       # human-readable filter description, if any
            "viewer_slot": None,         # named slot from camera_viewer, if any
        },
        # MAtCha invocation
        "matcha": {
            "git_sha": None,             # Anttwo/MAtCha SHA at runtime
            "image": None,               # docker image:tag
            "alignment_config": None,    # "default" | "strong"
            "dense_regul": None,         # "default" | "strong"
            "dense_pruning": None,       # "default" | "strong"
            "encoder": None,             # vitl/vitb/vits/vitg
            "sfm_config": None,          # "unposed" | ...
            "image_resolution_long_edge": None,
            "chart_resolutions_active": None,  # [0.05, 0.1, 0.2, 0.4] default
            "extra_flags": [],
        },
        # Execution
        "execution": {
            "host": None,                # "tbeeprz" | "bbeeprz" | "JDP-Mac"
            "gpu": None,                 # "RTX 5080 / 16 GB"
            "duration_seconds": None,
            "peak_vram_mib": None,
            "exit_status": None,         # "success" | "failure" | "partial"
        },
        # Output paths (relative to the variant dir)
        "outputs": {
            "tetra_mesh_path": None,
            "cameras_path": None,
            "oriented_mesh_path": None,
        },
        # Post-processing flags (B1-B4)
        "post_processing": {
            "orient": {"applied": False, "gravity_prior": False},
            "decimate": {"applied": False, "target_polys": None},
            "color_projection": {"applied": False},
            "cull": {"applied": False},
        },
        # Free-form prose
        "notes": None,
        # Journal cross-references (relative paths under .../journals/)
        "journal_refs": [],
    }


def variant_dir(scene_root: str, variant_name: str) -> str:
    """Resolve the variant directory under `scene_root` (== data/scenes/)."""
    # Convention: 004-sky-house-dining → curated variants stored as
    #             004-sky-house-curated-<variant>.
    if "-dining" in scene_root.rsplit("/", 1)[-1]:
        # Caller passed the dining root by mistake; redirect.
        raise ValueError(
            f"variant_dir() expects scene_root = data/scenes/, "
            f"not the dining dir itself. Got {scene_root!r}"
        )
    # Find the dir that ends with -<variant>
    for entry in os.listdir(scene_root):
        if entry.endswith(f"-curated-{variant_name}"):
            return os.path.join(scene_root, entry)
    raise FileNotFoundError(f"No variant dir found for {variant_name!r} under {scene_root}")


def manifest_path(scene_root: str, variant_name: str) -> str:
    return os.path.join(variant_dir(scene_root, variant_name), "manifest.json")


def write_manifest(scene_root: str, variant_name: str, manifest: dict[str, Any]) -> str:
    """Atomically write a manifest. Returns the absolute path written to."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"manifest schema_version mismatch: got {manifest.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION}"
        )
    p = manifest_path(scene_root, variant_name)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
    os.replace(tmp, p)
    return p


def read_manifest(scene_root: str, variant_name: str) -> dict[str, Any]:
    p = manifest_path(scene_root, variant_name)
    with open(p) as f:
        return json.load(f)


def list_variants(scene_root: str, scene_prefix: str) -> list[str]:
    """List variant names for a given scene, e.g. '004-sky-house'.

    Returns the suffix part — e.g. ['12', '12-strong', ...] — sorted alphabetically.
    """
    prefix = f"{scene_prefix}-curated-"
    out = []
    for entry in sorted(os.listdir(scene_root)):
        if entry.startswith(prefix):
            out.append(entry[len(prefix):])
    return out


def to_human_summary(m: dict[str, Any]) -> str:
    """Compact one-line/short-block summary for UI display.

    Returns a markdown-friendly string the rating UI drops into a panel.
    """
    matcha = m.get("matcha") or {}
    frames = m.get("frames") or {}
    exec_ = m.get("execution") or {}

    def _fmt(v):
        return v if v not in (None, "") else "—"

    lines = [
        f"**Variant:** `{m.get('variant_name') or '—'}`",
        f"**Frames:** {_fmt(frames.get('count'))} "
        f"(selection: {_fmt(frames.get('selection_method'))})",
        f"**MAtCha config:** alignment=`{_fmt(matcha.get('alignment_config'))}`, "
        f"dense_regul=`{_fmt(matcha.get('dense_regul'))}`, "
        f"dense_pruning=`{_fmt(matcha.get('dense_pruning'))}`",
        f"**Compute:** {_fmt(exec_.get('host'))}, "
        f"{_fmt(exec_.get('duration_seconds'))} sec, "
        f"peak {_fmt(exec_.get('peak_vram_mib'))} MiB",
    ]
    if m.get("notes"):
        lines.append("")
        lines.append(f"_{m['notes']}_")
    return "\n\n".join(lines)
