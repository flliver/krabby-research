#!/usr/bin/env python3
"""STO-SCN-093 — solver dispatch: capture profile + modality -> solve plan.

The single decision function that ties STO-SCN-091's capture profile together
with everything the 093 validation earned, so the pipeline picks the right
solve path automatically:

  - dewarped / COLMAP-incompatible  -> DA3 feed-forward (no undistort)
  - fisheye                         -> undistort to pinhole (102 calibration) -> FastMap
  - pinhole / rectilinear           -> FastMap directly

plus the pre-cull target (hyperlapse keeps the FULL culled pool — the 539/539
finding; DA3 is view-ceiling bound) and the matcher (exhaustive for
hyperlapse/wide-baseline — sequential fails on hyperlapse, HUG-SCN-004).

Pure dict logic — no store / cv2 / numpy — so it's trivially unit-testable.
`modality` is a capture property ("hyperlapse" | "video" | "photos"), declared
alongside `mode` (it isn't reliably inferable — STO-SCN-096 #3/#4).

CLI:
  solve_plan.py --make DJI --model "DJI Action 3" --mode fisheye \
      --modality hyperlapse [--pool-size N] [--profiles P]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DA3_VIEW_CEILING = 32   # measured feed-forward ceiling (16 GB), STO-SCN-101 corpus


def plan_solve(profile: dict, modality: str, pool_size: int | None = None) -> dict:
    """profile: a resolved capture-profile entry (STO-SCN-091, may carry
    `calibration` from 102). modality: hyperlapse | video | photos."""
    mode = (profile.get("mode") or "").lower()
    modality = (modality or "").lower()
    dead_end = bool(profile.get("dewarp_dead_end"))
    colmap_ok = bool(profile.get("colmap_compatible"))
    has_calib = bool(profile.get("calibration"))
    notes, warnings = [], []

    # --- solver + undistort ---
    if dead_end or not colmap_ok:
        solver, undistort = "da3", False
        notes.append("dewarped / COLMAP-incompatible -> DA3 feed-forward (no undistort, no calibration)")
    elif mode == "fisheye":
        solver, undistort = "fastmap", True
        notes.append("fisheye -> undistort to pinhole -> FastMap (FastMap takes only PINHOLE/SIMPLE_RADIAL)")
        if not has_calib:
            warnings.append("fisheye but NO calibration on profile -> undistort would use "
                            "approximate-FOV intrinsics; run calibrate_camera.py (STO-SCN-102) for accuracy")
    else:
        solver, undistort = "fastmap", False
        notes.append("pinhole / rectilinear -> FastMap directly")

    # --- pre-cull target (0 = keep full blur/dup-culled pool, no thin) ---
    if solver == "da3":
        precull_target = DA3_VIEW_CEILING
        notes.append(f"DA3 ~{DA3_VIEW_CEILING}-view ceiling -> precull target {DA3_VIEW_CEILING} "
                     f"(chunk into a spine beyond that)")
    elif modality == "hyperlapse":
        precull_target = 0
        notes.append("hyperlapse + FastMap -> KEEP the full blur/dup-culled pool, no thin "
                     "(539/539 finding; FastMap is GPU-scalable; 300 was a mast3r artifact)")
    else:
        precull_target = 0
        notes.append("FastMap is GPU-scalable -> keep the full culled pool (no thin)")

    # --- matcher ---
    if modality == "video":
        matcher = "sequential_matcher"
        notes.append("dense ordered video -> sequential matcher")
    else:
        matcher = "exhaustive_matcher"
        if modality == "hyperlapse":
            notes.append("hyperlapse -> exhaustive matcher (sequential fails on hyperlapse, HUG-SCN-004)")

    # --- camera model handed to the solver ---
    if solver == "da3":
        solve_camera_model = None
    elif undistort:
        solve_camera_model = "SIMPLE_PINHOLE"   # post-undistort
    else:
        solve_camera_model = "SIMPLE_RADIAL"

    return {
        "solver": solver,
        "undistort": undistort,
        "undistort_balance": 0.0 if undistort else None,
        "precull_target": precull_target,
        "precull_order": "capture-time (original_name)",
        "matcher": matcher,
        "solve_camera_model": solve_camera_model,
        "modality": modality,
        "mode": mode,
        "notes": notes,
        "warnings": warnings,
    }


def plan_for_scene(make, model, mode, modality, pool_size=None, profiles_path=None) -> dict:
    sys.path.insert(0, str(Path(__file__).parent))
    import capture_profile as cap
    reg = cap.load_registry(profiles_path) if profiles_path else None
    profile = cap.resolve(make, model, mode, registry=reg)
    return plan_solve(profile, modality, pool_size)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Solver dispatch: profile + modality -> solve plan.")
    ap.add_argument("--make", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--modality", required=True, choices=["hyperlapse", "video", "photos"])
    ap.add_argument("--pool-size", type=int, default=None)
    ap.add_argument("--profiles", default=None)
    a = ap.parse_args(argv)
    plan = plan_for_scene(a.make, a.model, a.mode, a.modality, a.pool_size, a.profiles)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
