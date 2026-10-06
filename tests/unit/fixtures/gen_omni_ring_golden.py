# SPDX-License-Identifier: BSD-3-Clause
"""Golden fixture for the omni RING terrain (omni phase-2 plan, step 2: "golden fixture first").

Captured from the UNMODIFIED ring code (commit ec979a98, before any field-geometry edit to
``omni_terrains.py`` / ``terrains/utils.py`` / the generator), so the field-geometry work can
prove the ring course every phase-1 head trained on is byte-identical afterwards: heights,
mesh vertices + faces, the dilated edge mask, the origin, the sector-anchor goals and their
heights, for the DECORATED ``omni_ring_terrain`` (sparse 8 and dense 16 blocks) and
``omni_flat_terrain``, with roughness ON at the shipped ``noise_range = (0.02, 0.02)`` and at the
class default ``(0.02, 0.06)``.

No Isaac Sim: the terrain functions are path-loaded under their real dotted names with the
package ``__init__``s replaced by inert stubs (the technique ``test_crab_hex_omni_ring_terrain.py``
documents). ``numpy.random`` is the module-global RNG the production decorators use, seeded per
case.

Importable surface for the byte-identity test: ``CASES``, ``make_cfg(case)``, ``run_case(omni, case)``
and ``load_terrain_modules()``.

Run directly to (re)write ``omni_ring_golden.npz`` -- ONLY on a tree whose ring code is the
reference (re-capturing after an edit would make the test vacuous):
    ``python tests/unit/fixtures/gen_omni_ring_golden.py``
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
LIB = REPO / "parkour" / "parkour_isaaclab"
OUT = Path(__file__).resolve().parent / "omni_ring_golden.npz"

#: (name, function, n_blocks or None for flat, difficulty, seed, noise_range)
CASES = (
    ("ring8_d042_shipped", "omni_ring_terrain", 8, 0.42, 101, (0.02, 0.02)),
    ("ring16_d070_shipped", "omni_ring_terrain", 16, 0.70, 202, (0.02, 0.02)),
    ("ring8_d005_classdefault", "omni_ring_terrain", 8, 0.05, 303, (0.02, 0.06)),
    ("ring16_d095_classdefault", "omni_ring_terrain", 16, 0.95, 505, (0.02, 0.06)),
    ("flat_d050_shipped", "omni_flat_terrain", None, 0.50, 404, (0.02, 0.02)),
)
FIELDS = ("vertices", "faces", "origin", "goals", "goal_heights", "edge_mask", "heights")


def _synthetic_package(name: str, path: Path) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(path)]
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
    return mod


def _load_real_module(name: str, file_path: Path) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def load_terrain_modules():
    """``(utils, fwd, omni)`` -- get-or-create, order-independent with the other terrain tests."""
    pkg = _synthetic_package("parkour_isaaclab", LIB)
    terrains = _synthetic_package("parkour_isaaclab.terrains", LIB / "terrains")
    pkg.terrains = terrains
    utils = _load_real_module("parkour_isaaclab.terrains.utils", LIB / "terrains" / "utils.py")
    terrains.utils = utils
    extreme = _synthetic_package("parkour_isaaclab.terrains.extreme_parkour", LIB / "terrains" / "extreme_parkour")
    terrains.extreme_parkour = extreme
    fwd = _load_real_module(
        "parkour_isaaclab.terrains.extreme_parkour.extreme_parkour_terrians",
        LIB / "terrains" / "extreme_parkour" / "extreme_parkour_terrians.py",
    )
    extreme.extreme_parkour_terrians = fwd
    omni_pkg = _synthetic_package("parkour_isaaclab.terrains.omni", LIB / "terrains" / "omni")
    terrains.omni = omni_pkg
    omni = _load_real_module(
        "parkour_isaaclab.terrains.omni.omni_terrains", LIB / "terrains" / "omni" / "omni_terrains.py"
    )
    omni_pkg.omni_terrains = omni
    return utils, fwd, omni


def make_cfg(case) -> types.SimpleNamespace:
    """The ring/flat sub-terrain cfg as ``TerrainGenerator.__init__`` hands it to the function
    (size/scales/slope_threshold propagated from ``OMNI_RING_TERRAINS_CFG``; border_width 0)."""
    _name, _func, n_blocks, _d, _seed, noise_range = case
    cfg = types.SimpleNamespace(
        border_width=0.0, horizontal_scale=0.08, vertical_scale=0.005, platform_len=2.5,
        platform_height=0.0, slope_threshold=1.5, edge_width_thresh=0.05, use_simplified=False,
        size=(10.0, 10.0), apply_roughness=True, apply_flat=False, downsampled_scale=0.075,
        noise_range=noise_range, noise_step=0.005, pad_width=0.1, pad_height=0.0,
        clear_radius=2.0, ring_outer_radius=4.5,
    )
    if n_blocks is not None:
        cfg.n_blocks = n_blocks
        cfg.bar_fraction = 0.5
        cfg.hurdle_height_range = "0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty"
        cfg.step_height = "0.02 + 0.06 * difficulty"
    return cfg


def run_case(omni, case) -> dict[str, np.ndarray]:
    name, func_name, _n, difficulty, seed, _noise = case
    np.random.seed(seed)
    out = getattr(omni, func_name)(difficulty, make_cfg(case), 8)
    meshes, origin, goals, goal_heights, edge_mask, heights = out[:6]
    return {
        "vertices": np.asarray(meshes[0].vertices),
        "faces": np.asarray(meshes[0].faces),
        "origin": np.asarray(origin),
        "goals": np.asarray(goals),
        "goal_heights": np.asarray(goal_heights),
        "edge_mask": np.asarray(edge_mask),
        "heights": np.asarray(heights),
    }


def main() -> None:
    _utils, _fwd, omni = load_terrain_modules()
    arrays = {}
    for case in CASES:
        res = run_case(omni, case)
        for key in FIELDS:
            arrays[f"{case[0]}__{key}"] = res[key]
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT} ({len(arrays)} arrays)")


if __name__ == "__main__":
    main()
