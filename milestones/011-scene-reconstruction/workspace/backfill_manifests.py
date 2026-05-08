"""Backfill manifest.json for the four existing curated variants.

Source of truth for each field: the journal notes captured 2026-05-01,
plus on-disk evidence (cameras.json mtimes, frame counts, file presence).

Run once. Re-running overwrites — idempotent. After this, future MAtCha
runs should write their own manifest as they kick off (a follow-up TODO
for the runner.sh refactor; out of scope here).
"""
from __future__ import annotations

import json
import os
import sys

# Allow `python backfill_manifests.py` from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from manifest_lib import empty_manifest, write_manifest  # type: ignore[import-not-found]  # noqa: E402

SCENES_ROOT = (
    "/private/var/krabby/workspace/milestones/011-scene-reconstruction/data/scenes"
)
SCENE = "004-sky-house"  # dining = the source-frame pool; variants suffix from here


def _read_basenames(variant_name: str) -> list[str]:
    cam_path = (
        f"{SCENES_ROOT}/{SCENE}-curated-{variant_name}/mast3r_sfm/cameras.json"
    )
    with open(cam_path) as f:
        d = json.load(f)
    return [p.rsplit("/", 1)[-1] for p in d["filepaths"]]


# ---------------------------------------------------------------------------
# Variant: 12 (default-alignment baseline)
# ---------------------------------------------------------------------------
v12 = empty_manifest()
v12.update({
    "variant_name": "12",
    "scene": "004-sky-house-dining",
    "captured_at": "2026-05-01T20:15:44-07:00",
})
v12["frames"] = {
    "count": 12,
    "basenames": _read_basenames("12"),
    "source_dir": "/data/frames/004-sky-house-dining-matcha-24",
    "selection_method": "manual-viewer",
    "viewer_filter": "spatial-cluster + look-at gizmo, picked from n350 viewer",
    "viewer_slot": None,
}
v12["matcha"] = {
    "git_sha": None,
    "image": "krabby-matcha:latest",
    "alignment_config": "default",
    "dense_regul": "default",
    "dense_pruning": "default",
    "encoder": "vitl",
    "sfm_config": "unposed",
    "image_resolution_long_edge": 1024,
    "chart_resolutions_active": [0.05, 0.1, 0.2, 0.4],
    "extra_flags": [],
}
v12["execution"] = {
    "host": "tbeeprz",
    "gpu": "RTX 5080 / 16 GB",
    "duration_seconds": 648,
    "peak_vram_mib": 7884,
    "exit_status": "success",
}
v12["outputs"] = {
    "tetra_mesh_path": "matcha_output/tetra_meshes/tetra_mesh_binary_search_7.ply",
    "cameras_path": "mast3r_sfm/cameras.json",
    "oriented_mesh_path": "oriented/oriented_500k_colored_culled.ply",
}
v12["post_processing"] = {
    "orient": {"applied": True, "gravity_prior": False},
    "decimate": {"applied": True, "target_polys": 500000},
    "color_projection": {"applied": True},
    "cull": {"applied": True},
}
v12["notes"] = (
    "First curated MAtCha run. Default --alignment_config (no regularization). "
    "Visible garbage geometry / hallucinated surfaces in regions with noisy SfM. "
    "Comparison baseline against which strong-alignment was justified."
)
v12["journal_refs"] = [
    "m11-scene-reconstruction/threads/matcha-quality/notes/"
    "2026-05-01T203949-accomplishments-and-next-steps",
]

# ---------------------------------------------------------------------------
# Variant: 12-strong (locked-in default after this run)
# ---------------------------------------------------------------------------
v12_strong = empty_manifest()
v12_strong.update({
    "variant_name": "12-strong",
    "scene": "004-sky-house-dining",
    "captured_at": "2026-05-01T21:01:18-07:00",
})
v12_strong["frames"] = {
    "count": 12,
    "basenames": _read_basenames("12-strong"),
    "source_dir": "/data/frames/004-sky-house-dining-matcha-24",
    "selection_method": "manual-viewer",
    "viewer_filter": "same 12 picks as variant '12' (controlled comparison)",
    "viewer_slot": None,
}
v12_strong["matcha"] = {
    "git_sha": None,
    "image": "krabby-matcha:latest",
    "alignment_config": "strong",
    "dense_regul": "default",
    "dense_pruning": "default",
    "encoder": "vitl",
    "sfm_config": "unposed",
    "image_resolution_long_edge": 1024,
    "chart_resolutions_active": [0.05, 0.1, 0.2, 0.4],
    "extra_flags": [],
}
v12_strong["execution"] = {
    "host": "tbeeprz",
    "gpu": "RTX 5080 / 16 GB",
    "duration_seconds": 648,
    "peak_vram_mib": 7874,
    "exit_status": "success",
}
v12_strong["outputs"] = {
    "tetra_mesh_path": "matcha_output/tetra_meshes/tetra_mesh_binary_search_7.ply",
    "cameras_path": "mast3r_sfm/cameras.json",
    "oriented_mesh_path": "oriented/oriented_500k_colored_culled.ply",
}
v12_strong["post_processing"] = {
    "orient": {"applied": True, "gravity_prior": True},
    "decimate": {"applied": True, "target_polys": 500000},
    "color_projection": {"applied": True},
    "cull": {"applied": True},
}
v12_strong["notes"] = (
    "Same 12 picks as variant '12', only --alignment_config flipped to strong. "
    "Eliminated garbage/hallucinated geometry visible in default. Same compute "
    "profile (within 0.1% wall-clock and VRAM). LOCKED as new default after this run."
)
v12_strong["journal_refs"] = [
    "m11-scene-reconstruction/threads/matcha-quality/notes/"
    "2026-05-01T222604-strong-alignment-config-eliminates-garbage-geometry",
]

# ---------------------------------------------------------------------------
# Variant: 16-strong (frame-count bracket on top of strong)
# ---------------------------------------------------------------------------
v16_strong = empty_manifest()
v16_strong.update({
    "variant_name": "16-strong",
    "scene": "004-sky-house-dining",
    "captured_at": "2026-05-01T22:27:15-07:00",
})
v16_strong["frames"] = {
    "count": 16,
    "basenames": _read_basenames("16-strong"),
    "source_dir": "/data/frames/004-sky-house-dining-matcha-24",
    "selection_method": "manual-viewer",
    "viewer_filter": "12-strong picks + 4 additional picks (Jeremy)",
    "viewer_slot": None,
}
v16_strong["matcha"] = {
    "git_sha": None,
    "image": "krabby-matcha:latest",
    "alignment_config": "strong",
    "dense_regul": "default",
    "dense_pruning": "default",
    "encoder": "vitl",
    "sfm_config": "unposed",
    "image_resolution_long_edge": 1024,
    "chart_resolutions_active": [0.05, 0.1, 0.2, 0.4],
    "extra_flags": [],
}
v16_strong["execution"] = {
    "host": "tbeeprz",
    "gpu": "RTX 5080 / 16 GB",
    "duration_seconds": 804,
    "peak_vram_mib": 10752,  # ~10.5 GB peak
    "exit_status": "success",
}
v16_strong["outputs"] = {
    "tetra_mesh_path": "matcha_output/tetra_meshes/tetra_mesh_binary_search_7.ply",
    "cameras_path": "mast3r_sfm/cameras.json",
    "oriented_mesh_path": "oriented/oriented_500k_colored_culled.ply",
}
v16_strong["post_processing"] = {
    "orient": {"applied": True, "gravity_prior": True},
    "decimate": {"applied": True, "target_polys": 500000},
    "color_projection": {"applied": True},
    "cull": {"applied": True},
}
v16_strong["notes"] = (
    "Frame-count bracket: 12 + 4 additional picks. Visually approx. equal to "
    "12-strong; the +4 picks did not earn their compute cost. First run died "
    "~6 sec in (CUDA disappeared from long-running container — see ops note); "
    "successful re-run after container restart."
)
v16_strong["journal_refs"] = [
    "m11-scene-reconstruction/threads/inbox/notes/"
    "2026-05-01T222605-operational-lesson-cuda-disappears-from-long-running-container",
]

# ---------------------------------------------------------------------------
# Variant: 12-dense-strong (regularization-stack experiment)
# ---------------------------------------------------------------------------
v12_dense = empty_manifest()
v12_dense.update({
    "variant_name": "12-dense-strong",
    "scene": "004-sky-house-dining",
    "captured_at": "2026-05-01T23:31:16-07:00",
})
v12_dense["frames"] = {
    "count": 12,
    "basenames": _read_basenames("12-dense-strong"),
    "source_dir": "/data/frames/004-sky-house-dining-matcha-24",
    "selection_method": "manual-viewer",
    "viewer_filter": "same 12 picks as variant '12' (controlled comparison)",
    "viewer_slot": None,
}
v12_dense["matcha"] = {
    "git_sha": None,
    "image": "krabby-matcha:latest",
    "alignment_config": "strong",
    "dense_regul": "strong",
    "dense_pruning": "default",
    "encoder": "vitl",
    "sfm_config": "unposed",
    "image_resolution_long_edge": 1024,
    "chart_resolutions_active": [0.05, 0.1, 0.2, 0.4],
    "extra_flags": [],
}
v12_dense["execution"] = {
    "host": "tbeeprz",
    "gpu": "RTX 5080 / 16 GB",
    "duration_seconds": None,  # not separately captured in journal
    "peak_vram_mib": None,
    "exit_status": "success",
}
v12_dense["outputs"] = {
    "tetra_mesh_path": "matcha_output/tetra_meshes/tetra_mesh_binary_search_7.ply",
    "cameras_path": "mast3r_sfm/cameras.json",
    "oriented_mesh_path": "oriented/oriented_500k_colored_culled.ply",
}
v12_dense["post_processing"] = {
    "orient": {"applied": True, "gravity_prior": True},
    "decimate": {"applied": True, "target_polys": 500000},
    "color_projection": {"applied": True},
    "cull": {"applied": True},
}
v12_dense["notes"] = (
    "Stacks --dense_regul strong on top of --alignment_config strong. "
    "Visually essentially equal to 12-strong; rolled back as default. "
    "Kept on disk for the rating matrix to confirm dense_regul provides "
    "no measurable benefit on this scene."
)
v12_dense["journal_refs"] = []

# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------
all_manifests = {
    "12": v12,
    "12-strong": v12_strong,
    "12-dense-strong": v12_dense,
    "16-strong": v16_strong,
}

if __name__ == "__main__":
    for name, m in all_manifests.items():
        path = write_manifest(SCENES_ROOT, name, m)
        size_kb = os.path.getsize(path) / 1024
        print(f"  wrote {path}  ({size_kb:.1f} KB)")
    print(f"Backfilled {len(all_manifests)} manifests.")
