# SPDX-License-Identifier: BSD-3-Clause
"""``crab_hex_omni_task/mdp/site_task_math.py`` -- the TRIMMED pure math of the site per-kind telemetry (rough-terrain
port, ``experiments/2026-11-10_rough_terrain/PLAN.md`` step 1, 2026-11-10; the subset of the closed site teacher's
``tests/unit/test_site_task_math.py`` at ``worktree-sen0581-cameras`` 2a3e3c51 that covers what was ported). CPU only:
the module is path-loaded standalone and ``site_raster`` is loaded under its real dotted name with stub parent
packages (``test_site_raster.py``'s loader).

Pinned here:

* the module is pure, standalone and carries ONLY the ported subset (no promotion table, rule ``loc``, slack,
  edge / void, blocker or impact math);
* the kind / policy literals equal ``site_raster``'s (this file loads standalone, so it restates them);
* :func:`crossed` (NaN = never crossed, inclusive at the radius);
* the hazard inner radius per kind (trench / straddle from the tile's ``cross_r``; the fixed table otherwise; NaN for
  flat, legacy field and blocker kinds) and its per-tile table;
* the radii literals equal ``site_terrains``' (``TRENCH_R_M``, ``STRADDLE_R_M``, ``BAND_R_IN_M``, ``PAD_R_M``, the
  first terrace riser and the inner snag ring).

Dropped from the site teacher's file (features not ported): the §7.2 / ``loc`` truth tables, the replay-script
equalities (their ``experiments/2026-10-28_site_teacher`` scripts are not on this branch), the slack / corridor /
latch / distance-field / sole / edge / chassis / blocker / impact tests and the ``site_reach`` envelope literals.
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
OMNI_MDP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp"
PARKOURS_DIR = REPO / "parkour/parkour_isaaclab/envs/mdp/parkours"
SITE_TERRAINS_PY = REPO / "parkour/parkour_isaaclab/terrains/omni/site_terrains.py"


def _load_real_module(dotted: str, path: Path) -> types.ModuleType:
    """Get-or-create loader of ``test_omni_p2b_knobs.py`` (real source dirs as stub packages' ``__path__``)."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    parts = dotted.split(".")
    directory = path.parent
    for i in range(len(parts) - 1, 0, -1):
        name = ".".join(parts[:i])
        pkg = sys.modules.get(name)
        if pkg is None:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(directory)]
            sys.modules[name] = pkg
        elif str(directory) not in getattr(pkg, "__path__", []):
            pkg.__path__ = list(getattr(pkg, "__path__", [])) + [str(directory)]
        directory = directory.parent
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
    return mod


def _load_standalone(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PKG = "parkour_tasks.crab_hex_omni_task.mdp"
_load_real_module("parkour_isaaclab.envs.mdp.parkours.exposure_stats", PARKOURS_DIR / "exposure_stats.py")
ox = _load_real_module(f"{_PKG}.omni_exposure", OMNI_MDP / "omni_exposure.py")
sr = _load_real_module(f"{_PKG}.site_raster", OMNI_MDP / "site_raster.py")
stm = _load_standalone("_t_site_task_math_standalone", OMNI_MDP / "site_task_math.py")


# ============================================================================ constants
def test_module_is_pure_and_standalone():
    src = (OMNI_MDP / "site_task_math.py").read_text()
    for bad in ("import isaaclab", "from isaaclab", "import omni", "from omni", "from .", "import numpy"):
        assert bad not in src, bad
    assert set(stm.__all__) <= set(dir(stm))


def test_only_the_ported_subset_is_present():
    """The trimmed module carries the telemetry math only: none of the site teacher's promotion / slack / reward /
    termination functions (those features are not ported, so nothing may call them)."""
    for gone in ("promotion_site", "promotion_site_loc", "promotion_site_rule", "blocked_probe_points",
                 "corridor_points", "corridor_flags", "latch_update", "effective_command_corridor",
                 "blocker_distance_field", "sole_points", "void_fraction", "near_edge", "blocker_contact", "impact"):
        assert not hasattr(stm, gone), gone
    for gone in ("promotion_mask_site", "promotion_mask_site_rule", "site_hazard_inner_r_table", "SiteBlockerSteps",
                 "SITE_BLOCKER_STEMS", "SITE_ONSET_WINDOW_S"):
        assert not hasattr(ox, gone), gone


def test_kind_literals_match_site_raster():
    assert stm.KIND_GROUND == sr.KIND_GROUND
    assert stm.VOID_KINDS == sr.VOID_KINDS
    assert stm.BLOCKER_KINDS == sr.BLOCKER_KINDS
    assert (stm.POLICY_LEGACY, stm.POLICY_CROSS, stm.POLICY_VOID, stm.POLICY_BLOCKER) == (
        sr.POLICY_LEGACY, sr.POLICY_CROSS, sr.POLICY_VOID, sr.POLICY_BLOCKER)


# ============================================================================ crossed
def test_crossed_is_false_on_nan_and_inclusive():
    r = torch.tensor([1.0, 4.1, 5.0, 9.0, 3.0])
    c = torch.tensor([4.1, 4.1, 4.1, float("nan"), float("inf")])
    assert stm.crossed(r, c).tolist() == [False, True, True, False, False]
    assert torch.equal(ox.site_crossed(r, c), stm.crossed(r, c))


# ============================================================================ hazard inner radius
def test_hazard_inner_radius_per_kind():
    """Trench / straddle: ``c - w / 2`` from ``cross_r = c + w / 2 + 0.5``; the fixed table otherwise (pits / rubble /
    ramps 2.0, ledges / terraces 1.7, snags 2.12); NaN for kinds without a hazard."""
    for w in (0.15, 0.3, 0.45):
        assert stm.hazard_inner_radius("trench", 2.4 + w / 2 + 0.5) == pytest.approx(2.4 - w / 2, abs=1e-12)
    for w in (0.25, 0.45):
        assert stm.hazard_inner_radius("straddle", 2.6 + w / 2 + 0.5) == pytest.approx(2.6 - w / 2, abs=1e-12)
    for kind, want in (("pits", 2.0), ("rubble", 2.0), ("ramps", 2.0), ("ledge_down", 1.7), ("ledge_up", 1.7),
                       ("terraces", 1.7), ("snags", 2.12)):
        assert stm.hazard_inner_radius(kind, 3.2) == pytest.approx(want, abs=1e-12), kind
    for kind in ("flat", "field_sparse", "field_dense", "chests", "walls"):
        assert math.isnan(stm.hazard_inner_radius(kind, 3.0))
    t = torch.tensor([2.975, float("nan")], dtype=torch.float64)
    assert torch.equal(stm.hazard_inner_radius("pits", t), torch.full_like(t, 2.0))
    assert stm.hazard_inner_radius("trench", t)[0].item() == stm.hazard_inner_radius("trench", 2.975)


def test_hazard_inner_radius_table():
    kinds = tuple(sr.SITE_KINDS)
    k = {name: i for i, name in enumerate(kinds)}
    tile_kind = torch.tensor([[k["trench"], k["pits"], k["walls"], -1]], dtype=torch.int8)
    cross = torch.tensor([[3.0, 4.1, float("nan"), float("nan")]])
    tab = stm.hazard_inner_radius_table(tile_kind, cross, kinds)
    assert tab.dtype == torch.float32 and tab.shape == (1, 4)
    assert tab[0, 0].item() == pytest.approx(2.3) and tab[0, 1].item() == 2.0
    assert math.isnan(tab[0, 2].item()) and math.isnan(tab[0, 3].item())


def test_radius_literals_are_site_terrains():
    """The restated radii equal the geometry module's (``site_terrains`` is not imported by this file: it needs the
    terrain library's package; its literals are read from source)."""
    import ast

    tree = ast.parse(SITE_TERRAINS_PY.read_text())
    lit = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    try:
                        lit[tgt.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
                elif isinstance(tgt, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    for t, v in zip(tgt.elts, node.value.elts):
                        if isinstance(t, ast.Name):
                            lit[t.id] = ast.literal_eval(v)
    assert stm.TRENCH_CENTRE_R_M == lit["TRENCH_R_M"] and stm.STRADDLE_CENTRE_R_M == lit["STRADDLE_R_M"]
    assert stm.HAZARD_INNER_R_M["pits"] == lit["BAND_R_IN_M"] == stm.HAZARD_INNER_R_M["rubble"]
    assert stm.HAZARD_INNER_R_M["ledge_down"] == lit["PAD_R_M"] == lit["TERRACE_RISER_R_M"][0]
    assert stm.HAZARD_INNER_R_M["snags"] == pytest.approx(lit["SNAG_R_M"][0] - lit["SNAG_WIDTH_M"] / 2, abs=1e-12)
