#!/usr/bin/env python3
"""scout_gauge.py — DA3 `scale_factor` provenance (STO-SCN-105).

⚠️ NOT the splat registration. A real scout disproved the scale_factor theory:
`scale_factor` maps DA3's *colmap* points to the solve, but NOT the displayed
gs_ply. The splat is registered by DIRECT point-cloud alignment in
`scout_register.py`. This module is kept only to read/document the captured
`scale_factor` (correct provenance for the colmap frame).

--- original notes (the disproven theory, retained for context) ---

register the DA3 scout gaussian to the FastMap solve gauge.

DA3 (posed mode) echoes the input cameras unchanged in its npz, but emits the
GAUSSIAN in its own normalized frame: per da3_render_view.py, metric
predictions divide camera translations by `scale_factor` to land in gaussian
space. So

    gaussian_frame = solve_frame / scale_factor   (pure scale about the origin)

and to overlay the splat on the solve-gauge frustums we multiply every
gaussian position by `scale_factor`. There is NO rotation or offset between the
two frames (the input cameras were handed to DA3 verbatim), which is why the
echoed npz aligns to identity and could never recover this — `scale_factor` is
the whole registration.

The verify surface applies the scale LIVE in the viewer (GS dynamicScene) — it
never rewrites the .ply (SH rotation + the header-offset corruption that bit
the earlier cull attempt are both avoided this way).

Pure-stdlib throughout (no numpy) so it runs anywhere the pipeline does.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def read_scale_factor(scout_dir) -> float | None:
    """The scale that maps gaussian-frame positions INTO the solve gauge.
    Returns None when scout_gauge.json is absent (pre-STO-SCN-105 scout) or
    the prediction was non-metric (no scale_factor)."""
    gj = Path(scout_dir) / "scout_gauge.json"
    if not gj.exists():
        return None
    try:
        d = json.loads(gj.read_text())
    except (ValueError, OSError):
        return None
    sf = d.get("scale_factor")
    return float(sf) if sf else None


def estimate_scale_from_points(gaussian_std, solve_std) -> float:
    """Independent cross-check: the gaussian↔solve scale equals the ratio of
    the two point clouds' core spreads (per-axis std, robust to the DA3 far
    halo if `gaussian_std` is computed on the core). On 001-patio the scout
    gaussian std ~[1.54,0.93,1.48] vs SfM points ~[0.75,0.53,0.57] gave ~2.1,
    matching the measured scale_factor. Use to sanity-check read_scale_factor,
    not as the primary source (the persisted value is exact)."""
    g = math.sqrt(sum(float(x) ** 2 for x in gaussian_std))
    s = math.sqrt(sum(float(x) ** 2 for x in solve_std))
    if s == 0:
        raise ValueError("solve_std is zero — no spread to scale against")
    return g / s


def splat_transform(scout_dir) -> dict:
    """The transform the verify viewer applies to the scout splat so it lands
    in the solve gauge. Uniform scale about the origin; identity rotation +
    zero translation (the frames share an origin and orientation — only scale
    differs). `scale=1.0` with `registered=False` when no scale_factor is
    available, so the caller can warn instead of silently mis-overlaying."""
    sf = read_scale_factor(scout_dir)
    return {
        "scale": sf if sf else 1.0,
        "registered": sf is not None,
        "source": "scale_factor" if sf else "unregistered",
    }


if __name__ == "__main__":
    import sys
    print(json.dumps(splat_transform(sys.argv[1]), indent=2))
