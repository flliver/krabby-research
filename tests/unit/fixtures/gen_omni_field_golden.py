# SPDX-License-Identifier: BSD-3-Clause
"""Golden fixture for the omni FIELD terrain (site teacher DESIGN §3.10: "golden fixture first").

Captured from the UNMODIFIED field code (HEAD 2bcc01ae, ``git archive`` of ``parkour_isaaclab/terrains``
into a scratch tree, before any site-terrain edit to ``terrains/utils.py`` / the generator), so the site
work can prove the legacy field every F19 stage trained on is byte-identical afterwards: mesh vertices +
faces, the origin, the sector-anchor goals and their heights, the dilated edge mask, the heights and the
exact block mask (the 7th decorator value) of the DECORATED ``omni_field_terrain``, roughness ON at the
shipped ``noise_range = (0.02, 0.02)``:

* sparse 14, ramp edges, d 0.5;
* dense 28, wall edges, d 0.9;
* holes-like (``KRABBY_FIELD_KINDS=0:0:1``, ``KRABBY_FIELD_SIZE_SCALE=0.6``, wall), dense 28, d 0.8;
* the same holes cfg at d 5.0 (the holes rows' ``KRABBY_FLAT_TERRAIN_DIFF=4.0:6.0`` band; beyond §3.10's three).

No Isaac Sim: the loader of ``gen_omni_ring_golden.py`` (inert package stubs, real dotted names);
``numpy.random`` is the module-global RNG the production decorator uses, seeded per case.

Importable surface for the byte-identity test: ``CASES``, ``FIELDS``, ``OUT``, ``make_cfg(case)``,
``run_case(omni, case)``.

Run directly to (re)write ``omni_field_golden.npz`` -- ONLY on a tree whose field code is the reference
(re-capturing after an edit would make the test vacuous):
    ``python tests/unit/fixtures/gen_omni_field_golden.py``
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gen_omni_ring_golden as _ring  # noqa: E402

OUT = HERE / "omni_field_golden.npz"

#: (name, n_blocks, difficulty, seed, overrides)
_HOLES = {"bar_fraction": 0.0, "pit_fraction": 1.0, "vertical_edges": True, "size_scale": 0.6}
CASES = (
    ("sparse14_ramp_d050", 14, 0.50, 611, {}),
    ("dense28_wall_d090", 28, 0.90, 622, {"vertical_edges": True}),
    ("holes28_wall_d080", 28, 0.80, 633, _HOLES),
    ("holes28_wall_d500", 28, 5.00, 644, _HOLES),
)
FIELDS = ("vertices", "faces", "origin", "goals", "goal_heights", "edge_mask", "heights", "block_mask")


def make_cfg(case) -> types.SimpleNamespace:
    """``OmniFieldTerrainCfg`` defaults as ``TerrainGenerator.__init__`` hands them to the function
    (size / scales / slope threshold propagated from ``OMNI_FIELD_TERRAINS_CFG``; border_width 0),
    with the case's knob overrides applied as ``apply_field_geometry`` writes them."""
    _name, n_blocks, _d, _seed, over = case
    cfg = types.SimpleNamespace(
        border_width=0.0, horizontal_scale=0.08, vertical_scale=0.005, platform_len=2.5, platform_height=0.0,
        slope_threshold=1.5, edge_width_thresh=0.05, use_simplified=False, size=(10.0, 10.0),
        apply_roughness=True, apply_flat=False, downsampled_scale=0.075, noise_range=(0.02, 0.02),
        noise_step=0.005, pad_width=0.1, pad_height=0.0, clear_radius=1.7, ring_outer_radius=4.80,
        field_margin=0.16, n_blocks=n_blocks, bar_fraction=0.4, pit_fraction=0.2, min_gap=0.0, max_tries=64,
        bar_length_range=(1.0, 2.0), bar_thickness_range=(0.16, 0.32), slab_side_range=(0.6, 1.2),
        hurdle_height_range="0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty",
        step_height="0.02 + 0.06 * difficulty", pit_depth="0.02 + 0.06 * difficulty", vertical_edges=False,
    )
    over = dict(over)
    scale = over.pop("size_scale", None)
    cfg.__dict__.update(over)
    if scale is not None:
        for attr in ("bar_length_range", "bar_thickness_range", "slab_side_range"):
            lo, hi = getattr(cfg, attr)
            setattr(cfg, attr, (float(lo) * scale, float(hi) * scale))
    return cfg


def run_case(omni, case, function: str = "omni_field_terrain") -> dict[str, np.ndarray]:
    """The decorated field function's first seven values for ``case`` (``function`` lets the site
    test run ``site_terrains.omni_site_field_terrain`` through the same capture)."""
    name, _n, difficulty, seed, _over = case
    np.random.seed(seed)
    out = getattr(omni, function)(difficulty, make_cfg(case), 8)
    meshes, origin, goals, goal_heights, edge_mask, heights, block_mask = out[:7]
    return {
        "vertices": np.asarray(meshes[0].vertices),
        "faces": np.asarray(meshes[0].faces),
        "origin": np.asarray(origin),
        "goals": np.asarray(goals),
        "goal_heights": np.asarray(goal_heights),
        "edge_mask": np.asarray(edge_mask),
        "heights": np.asarray(heights),
        "block_mask": np.asarray(block_mask),
    }


def main() -> None:
    _utils, _fwd, omni = _ring.load_terrain_modules()
    arrays = {}
    for case in CASES:
        res = run_case(omni, case)
        for key in FIELDS:
            arrays[f"{case[0]}__{key}"] = res[key]
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
