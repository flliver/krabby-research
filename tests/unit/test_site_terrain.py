# SPDX-License-Identifier: BSD-3-Clause
"""Site terrain (site teacher DESIGN §3, §10.2 "W1"): ``terrains/omni/site_terrains.py``, the 5-value body path of
``terrains/utils.py`` and the generator's site storage, the site knobs / dispatch / applier of
``mdp/omni_terrain_knobs.py`` and the pad-level spawn of ``mdp/omni_events.py``.

No Isaac Sim: the terrain modules are path-loaded with the golden loader's inert package stubs
(``fixtures/gen_omni_ring_golden.py::load_terrain_modules``), the knob module standalone, ``omni_events`` under a
private stub package (the ``test_crab_hex_omni_rsi.py`` technique). Cfgs are ``SimpleNamespace`` stand-ins mirroring
the real classes after ``TerrainGenerator.__init__`` propagates size / scales / slope threshold.

What is pinned:

* golden neutrality (DESIGN §3.10): ``fixtures/omni_field_golden.npz`` was captured from HEAD 2bcc01ae's field code
  (``git archive``) BEFORE any terrain edit; ``omni_field_terrain`` stays byte-identical, and the site field body
  reproduces it value for value at the same RNG state; 3- / 4-value bodies keep returning 6 / 7 values;
* per kind: ladder values at d in {0, 0.5, 1}, plant caps, kind raster vs heights (void below the level, blockers
  above), the seam rule, the +-4.80 m square, forced crossing (720 rays), blockers never closed, straddle / blocker
  disjointness, determinism, vertical risers and riser-free ramps;
* the 8-value decorator in the bordered frame, generator storage incl. tiles generated before allocation;
* the column map (``KRABBY_SITE_MIX`` -> exactly ``count`` consecutive columns per kind), the site difficulty map,
  the geometry dispatch (site accepted, RING_* refused, FIELD_* applied to the legacy columns, SITE_* ignored with a
  print under ring / field) and the applier's caps / refusals / spawn arming;
* ``spawn_level_z`` off = today's spawn bitwise, on = + the tile's pad level.

Rough-terrain port (``experiments/2026-11-10_rough_terrain/PLAN.md`` step 1, 2026-11-10): ported from the closed site
teacher's ``worktree-sen0581-cameras`` 2a3e3c51. Site physics DR is not ported, so the applier's ramp cap is 20 deg
whatever the env holds, and a set ``KRABBY_SITE_*`` knob that the port does not carry is REFUSED (in every geometry);
the site campaign's lower-band manifest test (its ``experiments/2026-10-28_site_teacher`` scripts are not on this
branch) is dropped.
"""
from __future__ import annotations

import ast
import copy
import importlib
import importlib.machinery
import importlib.util
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("trimesh", reason="the terrain mesh utilities (terrains/utils.py) import trimesh")

REPO = Path(__file__).resolve().parents[2]
OMNI_MDP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/mdp"
TERRAIN_KNOBS_PY = OMNI_MDP / "omni_terrain_knobs.py"
SITE_RASTER_PY = OMNI_MDP / "site_raster.py"
GEN_SRC = REPO / "parkour/parkour_isaaclab/terrains/parkour_terrain_generator.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

import gen_omni_field_golden as field_golden  # noqa: E402
import gen_omni_ring_golden as golden  # noqa: E402


def _load(path: Path, name: str) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


tk = _load(TERRAIN_KNOBS_PY, "crab_hex_omni_terrain_knobs")
sr = _load(SITE_RASTER_PY, "site_raster__site_terrain_test")
utils, fwd, omni = golden.load_terrain_modules()
st = golden._load_real_module("parkour_isaaclab.terrains.omni.site_terrains",
                              golden.LIB / "terrains" / "omni" / "site_terrains.py")
K = st._site_raster()       # the kind constants as site_terrains itself sees them

HS, VS = 0.08, 0.005
REDUCED = (9.92, 9.92)
HAZARDS = tuple(st.LADDERS)
RING_KINDS = ("trench", "ledge_down", "ledge_up", "terraces", "snags", "straddle", "rubble", "ramps")
DS = (0.0, 0.5, 1.0)


def site_cfg(kind: str, *, size=REDUCED, apply_roughness: bool = False, **kw) -> types.SimpleNamespace:
    """``OmniSiteTerrainCfg`` defaults as the generator hands them to the function."""
    cfg = types.SimpleNamespace(
        border_width=0.0, horizontal_scale=HS, vertical_scale=VS, platform_len=2.5, platform_height=0.0,
        slope_threshold=1.5, edge_width_thresh=0.05, use_simplified=False, size=size,
        apply_roughness=apply_roughness, apply_flat=False, downsampled_scale=0.075, noise_range=(0.02, 0.02),
        noise_step=0.005, pad_width=0.1, pad_height=0.0, kind=kind, clear_radius=1.7, ring_outer_radius=4.80,
        field_margin=0.16, vertical_edges=True, site_d_range=(0.0, 1.0), gen_d_range=(0.0, 1.0),
        ramp_max_deg=20.0, max_tries=64,
    )
    cfg.__dict__.update(kw)
    return cfg


def layout(kind, d, seed=0, **kw):
    return st.site_layout(np.random.RandomState(seed), d, site_cfg(kind, **kw))


def grid(shape=(124, 124)):
    xs = (np.arange(shape[0]) + 0.5) * HS - 0.5 * REDUCED[0]
    dx, dy = np.meshgrid(xs, xs, indexing="ij")
    return dx, dy


DX, DY = grid()
R = np.hypot(DX, DY)
CHEB = np.maximum(np.abs(DX), np.abs(DY))


def rays(lay, r_to=None, n_rays=720, step=0.02):
    """``(kind, level)`` along 720 rays from the tile centre (0.5 deg apart, 0.02 m steps, nearest pixel)."""
    r_to = lay.cross_r if r_to is None else r_to
    ang = np.arange(n_rays) * (2.0 * np.pi / n_rays)
    rr = np.arange(0.0, r_to + 1e-9, step)
    x = rr[None, :] * np.cos(ang)[:, None]
    y = rr[None, :] * np.sin(ang)[:, None]
    w = lay.kind.shape[0]
    i = np.clip(np.round((x + 0.5 * REDUCED[0]) / HS - 0.5).astype(int), 0, w - 1)
    j = np.clip(np.round((y + 0.5 * REDUCED[1]) / HS - 0.5).astype(int), 0, w - 1)
    return lay.kind[i, j], lay.level[i, j]


def px(m):
    return int(round(m / VS))


# ============================================================================== golden neutrality (§3.10)
class TestFieldGolden:
    @pytest.mark.parametrize("case", field_golden.CASES, ids=[c[0] for c in field_golden.CASES])
    def test_omni_field_terrain_is_byte_identical(self, case):
        gold = np.load(field_golden.OUT)
        res = field_golden.run_case(omni, case)
        for key in field_golden.FIELDS:
            want = gold[f"{case[0]}__{key}"]
            assert res[key].dtype == want.dtype and np.array_equal(res[key], want), key

    @pytest.mark.parametrize("case", field_golden.CASES, ids=[c[0] for c in field_golden.CASES])
    def test_site_field_body_reproduces_the_field_and_adds_layers(self, case):
        """``omni_site_field_terrain`` = ``omni_field_terrain`` on the first seven values at the same RNG state."""
        gold = np.load(field_golden.OUT)
        res = field_golden.run_case(st, case, function="omni_site_field_terrain")
        for key in field_golden.FIELDS:
            assert np.array_equal(res[key], gold[f"{case[0]}__{key}"]), key
        np.random.seed(case[3])
        out = st.omni_site_field_terrain(case[2], field_golden.make_cfg(case), 8)
        assert len(out) == 8
        layers, heights, block = out[7], out[5], out[6]
        sr.check_tile_layers(layers, heights.shape)
        assert np.array_equal(layers["kind"] != sr.KIND_GROUND, block)
        assert ((layers["kind"] == sr.KIND_FIELD_BLOCK) == (layers["surface0"] > 0)).all()
        assert ((layers["kind"] == sr.KIND_FIELD_PIT) == (layers["surface0"] < 0)).all()
        assert not np.isin(layers["kind"], sr.VOID_KINDS).any()            # legacy pits are never void
        assert (layers["level"] == 0).all() and (layers["material"] == 0).all()
        assert math.isnan(layers["tile"]["cross_r"]) and math.isnan(layers["tile"]["d"])
        assert layers["tile"]["centre_level_m"] == 0.0

    def test_the_golden_was_captured_on_a_non_trivial_field(self):
        gold = np.load(field_golden.OUT)
        for case in field_golden.CASES:
            assert gold[f"{case[0]}__block_mask"].any(), case[0]
        holes = gold["holes28_wall_d500__heights"]
        assert holes.min() < -px(0.25)                                        # the d 5.0 pits are deep

    def test_three_and_four_value_bodies_keep_six_and_seven_values(self):
        np.random.seed(1)
        assert len(omni.omni_ring_terrain(0.5, golden.make_cfg(golden.CASES[0]), 8)) == 6
        assert len(omni.omni_flat_terrain(0.5, golden.make_cfg(golden.CASES[-1]), 8)) == 6
        np.random.seed(2)
        assert len(omni.omni_field_terrain(0.5, field_golden.make_cfg(field_golden.CASES[0]), 8)) == 7


# ============================================================================== difficulty and ladders
def test_site_difficulty_map():
    cfg = site_cfg("trench", gen_d_range=(0.2, 1.0), site_d_range=(0.1, 0.5))
    for row in range(10):
        d = 0.2 + 0.8 * row / 9
        assert st.site_difficulty(d, cfg) == pytest.approx(0.1 + 0.4 * row / 9, abs=1e-12)
    assert st.site_difficulty(0.2, cfg) == pytest.approx(0.1) and st.site_difficulty(1.0, cfg) == pytest.approx(0.5)
    assert st.site_difficulty(7.0, cfg) == pytest.approx(0.5)                         # clipped into the band
    flat_band = site_cfg("trench", gen_d_range=(4.0, 4.0), site_d_range=(0.3, 0.9))
    assert st.site_difficulty(4.0, flat_band) == pytest.approx(0.3)                  # degenerate band -> u = 0
    # the legacy band (holes rows 4.0:6.0) does not move the site ladder: u spans [0, 1] over it
    holes = site_cfg("pits", gen_d_range=(4.0, 6.0))
    assert st.site_difficulty(4.0, holes) == 0.0 and st.site_difficulty(6.0, holes) == 1.0


@pytest.mark.parametrize("kind,param", [(k, p) for k in st.LADDERS for p in st.LADDERS[k]])
def test_ladders_are_linear_and_capped(kind, param):
    lo, hi = st.LADDERS[kind][param]
    cap = st.CAPS.get((kind, param), math.inf)
    for d in (0.0, 0.25, 0.5, 1.0):
        assert st.ladder_value(kind, param, d) == pytest.approx(min(cap, lo + (hi - lo) * d))
        assert st.ladder_value(kind, param, d) <= cap


def test_caps_clamp_the_ladder_when_lowered(monkeypatch):
    monkeypatch.setitem(st.CAPS, ("trench", "width_m"), 0.30)
    assert st.ladder_value("trench", "width_m", 1.0) == 0.30
    lay = layout("trench", 1.0)
    width = (np.abs(R - st.TRENCH_R_M) <= 0.15 + 1e-9) == (lay.kind == K.KIND_TRENCH)
    assert width.all()


def test_ramp_cap_by_friction():
    assert st.ramp_cap_deg(None) == 20.0
    assert st.ramp_cap_deg(0.25) == pytest.approx(math.degrees(math.atan(0.25)) - 3.0)
    assert st.ramp_cap_deg(0.25) == pytest.approx(11.04, abs=0.01)
    assert st.ramp_cap_deg(2.0) == 20.0
    for mu in (None, 0.1, 0.25, 0.5, 0.8, 1.5):
        assert tk.site_ramp_cap_deg(mu) == st.ramp_cap_deg(mu)


# ============================================================================== per kind
class TestLadderValues:
    @pytest.mark.parametrize("d", DS)
    def test_trench(self, d):
        lay = layout("trench", d)
        w = st.ladder_value("trench", "width_m", d)
        assert ((lay.kind == K.KIND_TRENCH) == (np.abs(R - 2.4) <= 0.5 * w)).all()
        assert (lay.surface0[lay.kind == K.KIND_TRENCH] == -px(st.ladder_value("trench", "depth_m", d))).all()
        assert lay.cross_r == pytest.approx(2.4 + 0.5 * w + 0.5)

    @pytest.mark.parametrize("d", DS)
    @pytest.mark.parametrize("kind,sign", [("ledge_down", 1), ("ledge_up", -1)])
    def test_ledges(self, kind, sign, d):
        lay = layout(kind, d)
        h = st.ladder_value(kind, "height_m", d)
        pad = R <= 1.7
        assert (lay.level[pad] == sign * px(h)).all() and (lay.level[~pad] == 0).all()
        assert ((lay.kind == K.KIND_LEDGE) == pad).all()
        assert np.array_equal(lay.surface0, lay.level) and lay.cross_r == pytest.approx(1.7 + 1.5)

    @pytest.mark.parametrize("d", DS)
    def test_terraces(self, d):
        a = px(st.ladder_value("terraces", "riser_m", d))
        signs = set()
        for seed in range(12):
            lay = layout("terraces", d, seed)
            band = [(R >= r0) & (R < r1) for r0, r1 in zip((1.7, 2.3, 2.9), (2.3, 2.9, 3.5))]
            sign = int(np.sign(lay.level[band[0]][0]))
            signs.add(sign)
            for b, steps in zip(band, (1, 2, 1)):
                assert (lay.level[b] == sign * steps * a).all()
            assert (lay.level[(R < 1.7) | (R >= 3.5)] == 0).all()
            assert ((lay.kind == K.KIND_TERRACE) == (lay.level != 0)).all()
        assert signs == {-1, 1}                                     # both variants occur
        assert lay.cross_r == 4.0

    @pytest.mark.parametrize("d", DS)
    def test_snags_and_straddle(self, d):
        lay = layout("snags", d)
        rings = (np.abs(R - 2.2) <= 0.08) | (np.abs(R - 3.0) <= 0.08)
        assert ((lay.kind == K.KIND_SNAG) == rings).all()
        assert (lay.surface0[rings] == px(st.ladder_value("snags", "height_m", d))).all()
        for seed in range(6):
            lay = layout("straddle", d, seed)
            m = lay.kind == K.KIND_STRADDLE
            w = 2.0 * (lay.cross_r - 0.5 - 2.6)
            assert 0.25 - 1e-9 <= w <= 0.45 + 1e-9
            assert (m == (np.abs(R - 2.6) <= 0.5 * w)).all()
            assert (lay.surface0[m] == px(st.ladder_value("straddle", "height_m", d))).all()

    @pytest.mark.parametrize("d", DS)
    def test_pits(self, d):
        side = st.ladder_value("pits", "side_m", d)
        n_target = min(120, round(0.40 * math.pi * (3.6 ** 2 - 2.0 ** 2) / side ** 2))
        for seed in range(4):
            lay = layout("pits", d, seed)
            m = lay.kind == K.KIND_PIT
            assert lay.n_placed + lay.n_skipped == n_target and lay.n_placed > 0
            assert (lay.surface0[m] == -px(st.ladder_value("pits", "depth_m", d))).all()
            assert R[m].min() >= 2.0 and R[m].max() <= 3.6                   # inside the annulus
            # >= 2 clear pixels between pits: every component grown by 2 px touches no other
            from scipy.ndimage import label                                  # noqa: PLC0415
            lab, n_comp = label(m, structure=np.ones((3, 3)))
            assert n_comp == lay.n_placed
            for k in range(1, n_comp + 1):
                grown = omni._dilate_square(lab == k, 2)
                assert not (grown & m & (lab != k)).any()
            # each pit is a square of side s * U(0.9, 1.1) (area check, rasterised)
            areas = np.bincount(lab.ravel())[1:] * HS * HS
            assert areas.min() >= (0.9 * side - 2 * HS) ** 2 * 0.8 and areas.max() <= (1.1 * side + 2 * HS) ** 2

    @pytest.mark.parametrize("d", DS)
    def test_chests(self, d):
        h_lad = st.ladder_value("chests", "height_m", d)
        for seed in range(6):
            lay = layout("chests", d, seed)
            m = lay.kind == K.KIND_BLOCKER_CHEST
            assert lay.n_placed + lay.n_skipped == 4 + round(3 * d) and lay.n_placed >= 3
            h = lay.surface0[m] * VS
            assert h.min() >= max(0.60, h_lad - 0.05) - VS and h.max() <= min(1.20, h_lad + 0.05) + VS
            assert math.isnan(lay.cross_r)

    @pytest.mark.parametrize("d", DS)
    def test_walls(self, d):
        for seed in range(6):
            lay = layout("walls", d, seed)
            m = lay.kind == K.KIND_BLOCKER_WALL
            assert lay.n_placed + lay.n_skipped == 2 + round(2 * d) and lay.n_placed >= 2
            assert (lay.surface0[m] == px(st.ladder_value("walls", "height_m", d))).all()
            assert R[m].min() >= 2.6 - 0.12 and math.isnan(lay.cross_r)

    @pytest.mark.parametrize("d", DS)
    def test_rubble(self, d):
        h_max = st.ladder_value("rubble", "height_max_m", d)
        lay = layout("rubble", d)
        band = (R >= 2.0) & (R <= 3.6)
        assert ((lay.kind == K.KIND_RUBBLE) == band).all()
        z = lay.surface0[band]
        assert z.min() >= 0 and z.max() <= px(h_max) and z.max() >= px(h_max) - 2
        # 4-px axis-aligned cells: constant inside each cell
        cells = lay.surface0.reshape(31, 4, 31, 4)
        mask = band.reshape(31, 4, 31, 4)
        for ci in range(31):
            for cj in range(31):
                v = cells[ci, :, cj, :][mask[ci, :, cj, :]]
                assert v.size == 0 or (v == v[0]).all()

    @pytest.mark.parametrize("d", DS)
    def test_ramps(self, d):
        alpha = st.ladder_value("ramps", "slope_deg", d)
        lay = layout("ramps", d)
        berm = (R >= 2.0) & (R < 3.6)
        assert ((lay.kind == K.KIND_RAMP) == berm).all()
        top = (R >= 2.6) & (R < 3.0)
        assert (lay.surface0[top] == px(0.6 * math.tan(math.radians(alpha)))).all()
        assert lay.surface0.max() * VS <= 0.22 + VS
        lay_capped = layout("ramps", d, ramp_max_deg=11.0)
        assert lay_capped.surface0.max() == px(0.6 * math.tan(math.radians(min(alpha, 11.0))))


# ============================================================================== bottom rung (PLAN_rev1 B4)
#: PLAN_rev1's B4 table at d_site = 0 (probe C §2), plus the chests floor; pits side / walls / ramps keep today's.
B4_AT_D0 = {
    ("pits", "depth_m"): 0.03, ("trench", "width_m"): 0.05, ("trench", "depth_m"): 0.03,
    ("straddle", "height_m"): 0.04, ("terraces", "riser_m"): 0.02, ("ledge_down", "height_m"): 0.02,
    ("ledge_up", "height_m"): 0.02, ("snags", "height_m"): 0.02, ("rubble", "height_max_m"): 0.02,
    ("chests", "height_m"): 0.85,
}
#: ... and at d_site = 0.5: ``bottom + (today(0.6) - bottom) * 0.5 / 0.6``, written out.
B4_AT_D05 = {
    ("pits", "depth_m"): 0.03 + (0.28 - 0.03) * 5 / 6, ("trench", "width_m"): 0.05 + (0.33 - 0.05) * 5 / 6,
    ("trench", "depth_m"): 0.03 + (0.31 - 0.03) * 5 / 6, ("straddle", "height_m"): 0.04 + (0.48 - 0.04) * 5 / 6,
    ("terraces", "riser_m"): 0.02 + (0.14 - 0.02) * 5 / 6, ("ledge_down", "height_m"): 0.02 + (0.23 - 0.02) * 5 / 6,
    ("ledge_up", "height_m"): 0.02 + (0.20 - 0.02) * 5 / 6, ("snags", "height_m"): 0.02 + (0.132 - 0.02) * 5 / 6,
    ("rubble", "height_max_m"): 0.02 + (0.17 - 0.02) * 5 / 6, ("chests", "height_m"): 0.85 + (0.96 - 0.85) * 5 / 6,
}
KNEE_DS = (0.6, 0.6 + 1e-9, 0.65, 6 / 9, 0.75, 7 / 9, 8 / 9, 0.93, 1.0)


class TestBottomRung:
    def test_the_b4_table_is_the_plan_table(self):
        assert st.LADDER_KNEE_D == 0.6 and set(st.SITE_LADDER_BOTTOMS) == {"b4"}
        assert st.SITE_LADDER_BOTTOMS["b4"] == B4_AT_D0 and st.SITE_LADDER_PIT_COVER0 == {"b4": 0.05}

    @pytest.mark.parametrize("kind,param", [(k, p) for k in st.LADDERS for p in st.LADDERS[k]])
    def test_values_at_d0_and_d05(self, kind, param):
        today = st.ladder_value(kind, param, 0.0), st.ladder_value(kind, param, 0.5)
        assert (st.ladder_value(kind, param, 0.0, None), st.ladder_value(kind, param, 0.5, None)) == today
        b4 = st.ladder_value(kind, param, 0.0, "b4"), st.ladder_value(kind, param, 0.5, "b4")
        if (kind, param) in B4_AT_D0:
            assert b4[0] == pytest.approx(B4_AT_D0[(kind, param)], abs=1e-12)
            assert b4[1] == pytest.approx(B4_AT_D05[(kind, param)], abs=1e-12)
        else:                                                    # pits side, walls, ramps: today's ladder
            assert b4 == today

    def test_pit_cover(self):
        assert st.pit_cover(0.0) == st.pit_cover(0.3, None) == st.PIT_COVER == 0.40
        assert st.pit_cover(0.0, "b4") == pytest.approx(0.05, abs=1e-12)
        assert st.pit_cover(0.5, "b4") == pytest.approx(0.05 + 0.35 * 5 / 6, abs=1e-12)
        assert all(st.pit_cover(d, "b4") == 0.40 for d in KNEE_DS)

    @pytest.mark.parametrize("kind,param", [(k, p) for k in st.LADDERS for p in st.LADDERS[k]])
    def test_linear_below_the_knee_continuous_at_it_and_todays_above(self, kind, param):
        knee = st.ladder_value(kind, param, 0.6)
        bottom = B4_AT_D0.get((kind, param))
        for d in np.linspace(0.0, 0.6, 13)[:-1]:
            want = st.ladder_value(kind, param, d) if bottom is None else bottom + (knee - bottom) * d / 0.6
            assert st.ladder_value(kind, param, d, "b4") == pytest.approx(want, abs=1e-12)
        assert st.ladder_value(kind, param, 0.6 - 1e-12, "b4") == pytest.approx(knee, abs=1e-9)
        for d in KNEE_DS:                                        # today's expression, bit for bit
            assert st.ladder_value(kind, param, d, "b4") == st.ladder_value(kind, param, d)

    def test_stamped_levels_match_the_probe_table(self):
        """probe C §2's "B4 L0 / L1 / L2 / L5 (stamped)" column (training level L -> d_site = L / 9)."""
        table = {("pits", K.KIND_PIT): (0.03, 0.075, 0.125, 0.26), ("trench", K.KIND_TRENCH): (0.03, 0.08, 0.135, 0.29),
                 ("straddle", K.KIND_STRADDLE): (0.04, 0.12, 0.205, 0.445)}
        for (kind, kid), metres in table.items():
            for level, want in zip((0, 1, 2, 5), metres):
                lay = st.site_layout(np.random.RandomState(0), level / 9, site_cfg(kind), "b4")
                assert np.abs(lay.surface0[lay.kind == kid]).max() == px(want), (kind, level)
        for kind, want in (("terraces", (0.02, 0.04, 0.065, 0.13)), ("ledge_down", (0.02, 0.06, 0.10, 0.215)),
                           ("snags", (0.02, 0.04, 0.06, 0.125)), ("rubble", (0.02, 0.05, 0.075, 0.16))):
            for level, m in zip((0, 1, 2, 5), want):
                assert px(st.primary_metres(kind, level / 9, "b4")) == px(m), (kind, level)

    @pytest.mark.parametrize("kind", HAZARDS)
    def test_layouts_byte_identical_at_and_above_the_knee(self, kind):
        """Every d >= 0.6 tile (the eval band, training levels 6-9): heights, goals, riser, layers, tile record."""
        for d in KNEE_DS:
            for seed in (0, 1):
                outs = []
                for ladder in (None, "b4"):
                    np.random.seed(1000 + seed)                  # the roughness draws from the global stream
                    outs.append(st.site_height_field(np.random.RandomState(seed), 0.2 + 0.8 * d,
                                                     site_cfg(kind, apply_roughness=True, gen_d_range=(0.2, 1.0)),
                                                     8, ladder))
                a, b = outs
                for i in range(4):
                    assert np.array_equal(a[i], b[i]), (kind, d, seed, i)
                for key in ("kind", "level", "surface0", "material"):
                    assert np.array_equal(a[4][key], b[4][key]), (kind, d, seed, key)
                ta, tb = a[4]["tile"], b[4]["tile"]
                assert all(ta[k] == tb[k] or (math.isnan(ta[k]) and math.isnan(tb[k])) for k in ta)

    @pytest.mark.parametrize("kind", HAZARDS)
    def test_decorated_b4_body_equals_todays_at_and_above_the_knee(self, kind):
        for d in (0.6, 7 / 9, 1.0):
            np.random.seed(5)
            a = st.omni_site_terrain(d, site_cfg(kind, size=(10.0, 10.0), apply_roughness=True), 8)
            np.random.seed(5)
            b = st.omni_site_terrain_b4(d, site_cfg(kind, size=(10.0, 10.0), apply_roughness=True), 8)
            assert len(a) == len(b) == 8
            assert np.array_equal(a[0][0].vertices, b[0][0].vertices) and np.array_equal(a[0][0].faces, b[0][0].faces)
            for i in range(1, 7):
                assert np.array_equal(a[i], b[i]), (kind, d, i)
            for key in sr.LAYER_DTYPES:
                assert np.array_equal(a[7][key], b[7][key]), (kind, d, key)

    @pytest.mark.parametrize("kind", ("trench", "ledge_down", "straddle", "chests", "pits"))
    def test_below_the_knee_b4_is_shallower(self, kind):
        """d 0 / 0.3: the primary feature (chests: the box heights) is lower on b4 except the chests, which rise to
        the 0.85 m floor; ledge / straddle / chest draws are unchanged (same footprint)."""
        for d in (0.0, 0.3):
            a = st.site_layout(np.random.RandomState(4), d, site_cfg(kind))
            b = st.site_layout(np.random.RandomState(4), d, site_cfg(kind), "b4")
            ha, hb = np.abs(a.surface0).max(), np.abs(b.surface0).max()
            if kind == "chests":
                assert np.array_equal(a.kind, b.kind) and hb > ha
                h_b4 = st.ladder_value("chests", "height_m", d, "b4")
                chest = b.surface0[b.kind == K.KIND_BLOCKER_CHEST]
                assert chest.min() >= px(h_b4 - 0.05) - 1 and chest.max() <= px(h_b4 + 0.05) + 1
                assert chest.min() * VS > 0.70                    # every training chest above the straddle cap
            else:
                assert hb < ha
            if kind in ("ledge_down", "straddle"):
                assert np.array_equal(a.kind, b.kind) and a.cross_r == b.cross_r

    def test_pits_count_follows_the_cover_ramp(self):
        area = math.pi * (3.6 ** 2 - 2.0 ** 2)
        for d, cover in ((0.0, 0.05), (0.3, 0.05 + 0.35 * 0.5)):
            side = st.ladder_value("pits", "side_m", d)
            lay = st.site_layout(np.random.RandomState(0), d, site_cfg("pits"), "b4")
            assert lay.n_placed + lay.n_skipped == min(120, round(cover * area / side ** 2))

    @pytest.mark.parametrize("ladder", (None, "b4"))
    @pytest.mark.parametrize("d", DS)
    def test_primary_metres_is_what_the_layout_stamps(self, ladder, d):
        kids = {"pits": K.KIND_PIT, "trench": K.KIND_TRENCH, "snags": K.KIND_SNAG, "straddle": K.KIND_STRADDLE}
        assert set(st.PRIMARY_PARAM) == {"pits", "trench", "ledge_down", "ledge_up", "terraces", "snags", "straddle",
                                         "rubble"}
        for kind, kid in kids.items():
            lay = st.site_layout(np.random.RandomState(2), d, site_cfg(kind), ladder)
            assert np.abs(lay.surface0[lay.kind == kid]).max() == px(st.primary_metres(kind, d, ladder)), kind
        for kind in ("ledge_down", "ledge_up"):
            lay = st.site_layout(np.random.RandomState(2), d, site_cfg(kind), ladder)
            assert abs(int(lay.level[62, 62])) == px(st.primary_metres(kind, d, ladder)), kind
        lay = st.site_layout(np.random.RandomState(2), d, site_cfg("terraces"), ladder)
        assert np.abs(lay.level).max() == 2 * px(st.primary_metres("terraces", d, ladder))
        lay = st.site_layout(np.random.RandomState(2), d, site_cfg("rubble"), ladder)
        assert lay.surface0.max() <= px(st.primary_metres("rubble", d, ladder))
        for kind in ("chests", "walls", "ramps"):
            with pytest.raises(KeyError):
                st.primary_metres(kind, d, ladder)

    def test_refusals_and_ladder_of(self):
        for bad in ("b5", "B4", ""):
            with pytest.raises(ValueError, match="unknown site ladder"):
                st.ladder_value("trench", "depth_m", 0.1, bad)
            with pytest.raises(ValueError, match="unknown site ladder"):
                st.site_layout(np.random.RandomState(0), 0.1, site_cfg("trench"), bad)
        assert st.ladder_of(st.omni_site_terrain) is None and st.ladder_of(st.omni_site_terrain_b4) == "b4"
        assert st.SITE_LADDER_BODIES == {"b4": st.omni_site_terrain_b4}
        assert st.omni_site_terrain_b4.__name__ == "omni_site_terrain_b4"            # the dumped function leaf
        assert tuple(st.SITE_LADDER_BODIES) == tuple(st.SITE_LADDER_BOTTOMS) == tk.SITE_LADDERS
        with pytest.raises(ValueError, match="not a site hazard body"):
            st.ladder_of(st.omni_site_field_terrain)


# ============================================================================== raster, seams, crossing
@pytest.mark.parametrize("kind", HAZARDS)
@pytest.mark.parametrize("d", DS)
def test_raster_consistent_with_heights_and_seam_rule(kind, d):
    for seed in range(3):
        lay = layout(kind, d, seed)
        assert lay.kind.dtype == np.int8 and lay.level.dtype == np.int16 and lay.surface0.dtype == np.int16
        assert lay.riser.dtype == np.int16 and np.array_equal(lay.raw, lay.surface0.astype(np.float64))
        void = np.isin(lay.kind, sr.VOID_KINDS)
        assert (lay.surface0[void] < lay.level[void]).all()               # void cells below the level
        blk = np.isin(lay.kind, sr.BLOCKER_KINDS)
        assert (lay.surface0[blk] > lay.level[blk]).all()                 # blockers above it
        ground = lay.kind == K.KIND_GROUND
        assert (lay.surface0[ground] == 0).all() and (lay.level[ground] == 0).all()
        assert (lay.level[CHEB >= st.SEAM_R_M] == 0).all()                # seam rule
        assert not (lay.kind[CHEB > 4.80 + 1e-9] != 0).any()              # inside the +-4.80 m square
        assert not ((lay.kind != 0) & (R <= 1.7) & (lay.kind != K.KIND_LEDGE)).any()   # the pad is clear
        assert not np.isin(lay.kind, (K.KIND_RESERVED_SOFT, K.KIND_RESERVED_LOWMU, K.KIND_RESERVED_15)).any()
        if kind not in ("ledge_down", "ledge_up", "terraces"):
            assert (lay.level == 0).all()


@pytest.mark.parametrize("kind", RING_KINDS)
@pytest.mark.parametrize("d", DS)
def test_ring_kinds_force_a_crossing_on_every_ray(kind, d):
    for seed in range(2):
        lay = layout(kind, d, seed)
        k, lv = rays(lay)
        hazard = k != K.KIND_GROUND
        if kind in ("ledge_down", "ledge_up", "terraces"):
            riser = px(st.ladder_value(kind, "riser_m" if kind == "terraces" else "height_m", d))
            step = np.abs(np.diff(lv.astype(int), axis=1)).max(axis=1)
            hazard_ray = step >= riser - px(0.01)
        else:
            hazard_ray = hazard.any(axis=1)
        assert hazard_ray.all(), (kind, d, seed, 1.0 - hazard_ray.mean())


@pytest.mark.parametrize("d", DS)
def test_pits_block_most_rays(d):
    """Pooled over 20 seeded tiles >= 0.80 of the 720 rays hit a pit pixel; no tile below 0.70 (DESIGN §3.5 --
    the per-tile minimum at d = 1 is ~0.76, see the module's deviation note)."""
    fracs = []
    for seed in range(20):
        lay = layout("pits", d, seed)
        k, _ = rays(lay)
        fracs.append((k == K.KIND_PIT).any(axis=1).mean())
    assert np.mean(fracs) >= 0.80, fracs
    assert min(fracs) >= 0.70, fracs


def _gaps_have_a_wide_chord(lay, r_eval, chord=2.5):
    """True when some run of rays NOT hitting a blocker spans a chord >= ``chord`` at radius ``r_eval``."""
    k, _ = rays(lay, r_to=4.4)
    free = ~np.isin(k, sr.BLOCKER_KINDS).any(axis=1)
    n = free.size
    if free.all():
        return True
    start = int(np.argmin(free))
    run = best = 0
    for i in range(1, n + 1):
        if free[(start + i) % n]:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return 2.0 * r_eval * math.sin(0.5 * best * 2.0 * math.pi / n) >= chord


@pytest.mark.parametrize("kind,r_eval", [("chests", 3.3), ("walls", 3.0)])
@pytest.mark.parametrize("d", DS)
def test_blockers_leave_passage(kind, r_eval, d):
    for seed in range(10):
        lay = layout(kind, d, seed)
        k, _ = rays(lay, r_to=4.4)
        blocked = np.isin(k, sr.BLOCKER_KINDS).any(axis=1)
        assert blocked.mean() <= 0.60, (kind, d, seed, blocked.mean())
        assert not blocked.all()                                           # never a closed ring
        assert _gaps_have_a_wide_chord(lay, r_eval), (kind, d, seed)


def test_walls_angular_cover_and_clearance():
    for d in DS:
        for seed in range(10):
            lay = layout("walls", d, seed)
            m = lay.kind == K.KIND_BLOCKER_WALL
            from scipy.ndimage import label                                  # noqa: PLC0415
            lab, n = label(m, structure=np.ones((3, 3)))
            assert n == lay.n_placed
            pts = [np.stack([DX[lab == i], DY[lab == i]], -1) for i in range(1, n + 1)]
            for a in range(n):
                for b in range(a + 1, n):
                    gap = np.sqrt(((pts[a][:, None] - pts[b][None]) ** 2).sum(-1)).min() - HS
                    assert gap >= st.BLOCKER_CLEARANCE_M - 1e-9


def _blocker_rule(min_side, h):
    return (min_side >= 0.6 and h >= 0.6) or h >= 0.8


def test_straddle_and_blockers_are_disjoint():
    """PLAN item 12 (DESIGN §3.3): no straddle object satisfies the blocker rule; every blocker does."""
    from scipy.ndimage import label  # noqa: PLC0415

    for d in DS:
        for seed in range(6):
            lay = layout("straddle", d, seed)
            width = 2.0 * (lay.cross_r - 0.5 - st.STRADDLE_R_M)             # the crossing (radial) width
            h = lay.surface0[lay.kind == K.KIND_STRADDLE].max() * VS
            assert width <= 0.45 + 1e-9 and h <= st.STRADDLE_CAP_M + 1e-9
            assert not _blocker_rule(width, h)
            for kind, kid, sides in (("chests", K.KIND_BLOCKER_CHEST, None), ("walls", K.KIND_BLOCKER_WALL, None)):
                blay = layout(kind, d, seed)
                lab, n = label(blay.kind == kid, structure=np.ones((3, 3)))
                for i in range(1, n + 1):
                    comp = lab == i
                    hb = blay.surface0[comp].max() * VS
                    # the footprint's short side, from the rasterised extent (>= the drawn side minus a pixel)
                    min_side = 0.6 if kind == "chests" else 0.2
                    assert _blocker_rule(min_side, hb), (kind, d, seed, hb)
                    if kind == "chests":
                        assert hb >= 0.6 - VS
                    else:
                        assert hb >= 0.8 - VS
    assert len({K.KIND_STRADDLE, K.KIND_BLOCKER_CHEST, K.KIND_BLOCKER_WALL}) == 3
    assert st.CHEST_SIDES_M[0][0] >= 0.6 and st.CHEST_SIDES_M[1][0] >= 0.6 and st.TABLE_SIDES_M[1][0] >= 0.6
    assert st.LADDERS["chests"]["height_m"][0] >= 0.6 and st.LADDERS["walls"]["height_m"][0] >= 0.8
    assert st.STRADDLE_WIDTH_M[1] <= 0.45 and st.STRADDLE_CAP_M <= 0.70


@pytest.mark.parametrize("kind", HAZARDS)
def test_determinism_under_a_seeded_randomstate(kind):
    a, b = layout(kind, 0.7, 11), layout(kind, 0.7, 11)
    for x, y in zip(a, b):
        assert (np.array_equal(x, y) if isinstance(x, np.ndarray) else (x == y or (x != x and y != y)))
    if kind in ("pits", "chests", "walls", "rubble", "terraces", "straddle"):
        diff = [not np.array_equal(layout(kind, 0.7, s).surface0, a.surface0) for s in range(12, 18)]
        assert any(diff)


def test_layout_refusals():
    with pytest.raises(ValueError, match="unknown site kind"):
        layout("lava", 0.5)
    with pytest.raises(ValueError, match="difficulty"):
        layout("trench", 1.5)
    with pytest.raises(ValueError, match="clear_radius"):
        layout("trench", 0.5, clear_radius=2.0)
    with pytest.raises(ValueError, match="half-width"):
        layout("trench", 0.5, ring_outer_radius=4.90)
    assert layout("ledge_down", 0.5, clear_radius=1.6).cross_r == pytest.approx(3.1)


# ============================================================================== decorated bodies (8-tuple)
def _decorated(kind, d, seed=3, **kw):
    np.random.seed(seed)
    return st.omni_site_terrain(d, site_cfg(kind, size=(10.0, 10.0), apply_roughness=True, **kw), 8)


@pytest.mark.parametrize("kind", HAZARDS)
def test_decorator_returns_eight_values_in_the_bordered_frame(kind):
    out = _decorated(kind, 0.6)
    assert len(out) == 8
    meshes, origin, goals, goal_heights, edge_mask, heights, block_mask, layers = out
    assert heights.shape == (126, 126) and block_mask.shape == heights.shape
    sr.check_tile_layers(layers, heights.shape)
    for key in sr.LAYER_DTYPES:
        border = np.ones(heights.shape, dtype=bool)
        border[1:-1, 1:-1] = False
        assert (layers[key][border] == sr.LAYER_INIT[key]).all(), key
    np.random.seed(3)
    raw = st.site_height_field(np.random, 0.6, site_cfg(kind, apply_roughness=True), 8)
    for key in sr.LAYER_DTYPES:
        assert np.array_equal(layers[key][1:-1, 1:-1], raw[4][key]), key
    sr.check_tile_layers(raw[4], (124, 124))
    assert layers["tile"]["d"] == pytest.approx(0.6)
    # the heights are surface0 + roughness (+-2 cm, plus the cubic spline's overshoot) inside the 1-px pad strip
    inner = heights[2:-2, 2:-2].astype(int) - layers["surface0"][2:-2, 2:-2].astype(int)
    assert inner.min() >= -px(0.035) and inner.max() <= px(0.035)
    assert np.array_equal(block_mask, np.pad(raw[3], 1) != 0)
    if kind in ("ledge_down", "ledge_up"):
        assert layers["tile"]["centre_level_m"] == pytest.approx(layers["level"][63, 63] * VS)
        assert layers["tile"]["centre_level_m"] != 0.0
    else:
        assert layers["tile"]["centre_level_m"] == 0.0


@pytest.mark.parametrize("kind", ("trench", "ledge_down", "snags", "straddle", "chests", "walls", "rubble"))
def test_vertical_risers_on_feature_edges(kind):
    """Every riser step is a vertical wall: the riser-mode edge mask covers the low side of every step, and the
    mesh's vertex / face counts are those of the plain mesh (vertices move, none are added)."""
    np.random.seed(5)
    cfg = site_cfg(kind, apply_roughness=False)
    raw = st.site_height_field(np.random, 0.8, cfg, 8)
    riser = np.pad(raw[3], 1)
    hf = np.pad(raw[0], 1)
    v, f, mx, my = utils.convert_height_field_to_mesh(hf, HS, VS, 1.5, return_y_edges=True, riser_field=riser)
    v0, f0, _mx0, _my0 = utils.convert_height_field_to_mesh(hf, HS, VS, 1.5, return_y_edges=True)
    assert v.shape == v0.shape and f.shape == f0.shape
    step_x = np.zeros_like(mx)
    step_x[:-1] |= riser[1:] > riser[:-1]
    step_x[1:] |= riser[:-1] > riser[1:]
    assert (mx[step_x] != 0).all()
    assert riser.any()


def test_ramps_get_no_risers():
    for d in DS:
        out = _decorated("ramps", d)
        np.random.seed(3)
        raw = st.site_height_field(np.random, d, site_cfg("ramps", apply_roughness=True), 8)
        assert not raw[3].any()                                            # riser field: ramp cells contribute 0
        assert not out[6].any()                                            # no block mask on a ramp tile
        assert not out[4][2:-2, 2:-2].any()                                # and no wall in the edge mask


# ============================================================================== generator storage
def test_store_site_layers_allocates_lazily_with_the_contract_init():
    holder = types.SimpleNamespace(**{attr: None for attr in utils.SITE_GENERATOR_ATTRS.values()})
    utils.store_site_layers(holder, 0, 0, None, (3, 4), (126, 126))           # a flat tile first: nothing
    assert all(getattr(holder, a) is None for a in utils.SITE_GENERATOR_ATTRS.values())
    out = _decorated("ledge_up", 0.5)
    utils.store_site_layers(holder, 1, 2, out[7], (3, 4), (126, 126))
    for key, dtype in sr.LAYER_DTYPES.items():
        arr = getattr(holder, sr.GENERATOR_ATTRS[key])
        assert arr.dtype == dtype and arr.shape == (3, 4, 126, 126)
        assert np.array_equal(arr[1, 2], out[7][key])
        assert (arr[0, 0] == sr.LAYER_INIT[key]).all()                       # the tile stored before allocation
    for key in sr.TILE_INIT:
        arr = getattr(holder, sr.GENERATOR_ATTRS[key])
        assert arr.dtype == np.float32 and arr.shape == (3, 4)
        assert np.isnan(arr[0, 0]) if math.isnan(sr.TILE_INIT[key]) else arr[0, 0] == sr.TILE_INIT[key]
    assert holder.site_centre_level_m[1, 2] == pytest.approx(out[7]["tile"]["centre_level_m"])
    assert holder.site_d[1, 2] == pytest.approx(0.5)


def test_library_restatements_equal_the_raster_contract():
    assert utils.SITE_GENERATOR_ATTRS == sr.GENERATOR_ATTRS
    assert utils.SITE_LAYER_DTYPES == sr.LAYER_DTYPES
    assert set(utils.SITE_TILE_INIT) == set(sr.TILE_INIT)
    for key, v in sr.TILE_INIT.items():
        assert (math.isnan(v) and math.isnan(utils.SITE_TILE_INIT[key])) or v == utils.SITE_TILE_INIT[key]
    assert all(v == 0 for v in sr.LAYER_INIT.values())                       # the zero-filled border is LAYER_INIT
    assert tk._SITE_SUBTERRAIN_KEYS == sr.SITE_SUBTERRAIN_KEYS and tk._SITE_KINDS == sr.SITE_KINDS


def test_generator_source_initialises_site_attrs_before_the_base_constructor():
    src = GEN_SRC.read_text()
    tree = ast.parse(src)
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ParkourTerrainGenerator"]
    (init,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    body = ast.get_source_segment(src, init)
    assert body.index("SITE_GENERATOR_ATTRS.values()") < body.index("super().__init__(")
    for name in ("_generate_random_terrains", "_generate_curriculum_terrains"):
        (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name]
        seg = ast.get_source_segment(src, fn)
        assert "block_mask, layers = self._get_terrain_mesh(" in seg
        assert "self._store_site_layers(sub_row, sub_col, layers)" in seg
    assert "layers = out[7] if len(out) > 7 else None" in src


# ============================================================================== column map (§3.4)
def _curriculum_columns(props, num_cols):
    """The generator's curriculum column rule (``parkour_terrain_generator._generate_curriculum_terrains``)."""
    p = np.array(props, dtype=float)
    p /= p.sum()
    return [int(np.min(np.where(i / num_cols + 0.001 < np.cumsum(p))[0])) for i in range(num_cols)]


def _mix_tg():
    subs = {key: types.SimpleNamespace(proportion=0.123) for key in sr.SITE_SUBTERRAIN_KEYS}
    return types.SimpleNamespace(num_cols=20, sub_terrains=subs)


def _assert_columns(mix):
    tg = _mix_tg()
    tk.apply_site_mix(tg, mix)
    total = sum(mix.values())
    assert tg.num_cols == total
    props = [tg.sub_terrains[k].proportion for k in sr.SITE_SUBTERRAIN_KEYS]
    assert props == [mix[n] / total for n in sr.SITE_KINDS]
    cols = _curriculum_columns(props, total)
    want = [i for i, n in enumerate(sr.SITE_KINDS) for _ in range(mix[n])]
    assert cols == want, (mix, cols)


def test_column_map_default_mix():
    mix = tk.parse_site_mix(tk.DEFAULT_SITE_MIX)
    _assert_columns(mix)
    assert "if index / self.cfg.num_cols + 0.001 < np.cumsum(proportions)" not in GEN_SRC.read_text()
    assert "np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))" in GEN_SRC.read_text()


def test_column_map_random_mixes():
    rng = random.Random(1)
    for _ in range(200):
        counts = {k: 0 for k in sr.SITE_KINDS}
        for _ in range(rng.randint(1, 40)):
            counts[rng.choice(sr.SITE_KINDS)] += 1
        _assert_columns(counts)


# ============================================================================== knob names and parsers (§1.2)
def test_site_knob_names_are_the_contract():
    assert tk.SITE_MIX_VAR == "KRABBY_SITE_MIX" and tk.SITE_DIFF_VAR == "KRABBY_SITE_DIFF"
    assert tk.SITE_LADDER_VAR == "KRABBY_SITE_LADDER"                     # PLAN_rev1 B4
    assert tk.SITE_GEOM == "site" and tk.SITE_VARS == (tk.SITE_MIX_VAR, tk.SITE_DIFF_VAR, tk.SITE_LADDER_VAR)
    assert tk.SITE_KNOB_PARSERS == {tk.SITE_MIX_VAR: tk.parse_site_mix, tk.SITE_DIFF_VAR: tk.parse_site_diff,
                                    tk.SITE_LADDER_VAR: tk.parse_site_ladder}
    for name in ("SITE_GEOM", "SITE_MIX_VAR", "SITE_DIFF_VAR", "SITE_KNOB_PARSERS", "parse_site_mix",
                 "parse_site_diff", "DEFAULT_SITE_MIX", "apply_omni_site_knobs", "SITE_LADDER_VAR", "SITE_LADDERS",
                 "parse_site_ladder"):
        assert name in tk.__all__
    assert tk.OMNI_GEOMS == ("ring", "field", "site")


def test_parse_site_ladder():
    assert tk.parse_site_ladder(None) is None and tk.parse_site_ladder("") is None and tk.parse_site_ladder(" ") is None
    assert tk.parse_site_ladder("b4") == "b4" and tk.parse_site_ladder(" B4 ") == "b4"
    for bad in ("b5", "b4,b4", "0", "off", "today", "b 4"):
        with pytest.raises(ValueError, match=tk.SITE_LADDER_VAR):
            tk.parse_site_ladder(bad)


def test_column_kinds_restated_equal_to_site_raster():
    assert tk._SITE_KINDS == sr.SITE_KINDS


def test_default_mix_parses_to_twenty_columns():
    mix = tk.parse_site_mix(tk.DEFAULT_SITE_MIX)
    assert list(mix) == list(sr.SITE_KINDS) and sum(mix.values()) == 20
    assert mix["flat"] == 4 and mix["field_sparse"] == mix["field_dense"] == mix["pits"] == 2
    assert all(mix[k] == 1 for k in sr.SITE_KINDS[4:])


def test_parse_site_mix_grammar():
    assert tk.parse_site_mix(None) is None and tk.parse_site_mix("  ") is None
    mix = tk.parse_site_mix(" walls = 3 , flat=1,pits=0 ")
    assert list(mix) == list(sr.SITE_KINDS) and mix["walls"] == 3 and mix["flat"] == 1 and sum(mix.values()) == 4
    for bad in ("flat=2.0", "flat=-1", "flat=+1", "flat", "=3", "flat=1,,pits=1", "flat=1,", "lava=1",
                "flat=1,flat=2", "flat=0", "pits=0,trench=0", "flat=41", "flat=20,pits=21", "flat=1e1", "flat=x"):
        with pytest.raises(ValueError):
            tk.parse_site_mix(bad)
    assert sum(tk.parse_site_mix("flat=40").values()) == 40


def test_parse_site_mix_round_trips_random_mixes():
    rng = random.Random(0)
    for _ in range(200):
        counts = {k: 0 for k in sr.SITE_KINDS}
        for _ in range(rng.randint(1, 40)):
            counts[rng.choice(sr.SITE_KINDS)] += 1
        text = ",".join(f"{k}={v}" for k, v in counts.items())
        assert tk.parse_site_mix(text) == counts


def test_parse_site_diff():
    assert tk.parse_site_diff(None) is None and tk.parse_site_diff("") is None
    assert tk.parse_site_diff("0:1") == (0.0, 1.0)
    assert tk.parse_site_diff(" 0.25 : 0.75 ") == (0.25, 0.75)
    assert tk.parse_site_diff("0.5:0.5") == (0.5, 0.5)
    for bad in ("0.6:0.4", "-0.1:0.5", "0:1.1", "0.5", "0:0.5:1", "a:b", "nan:1", "0:inf"):
        with pytest.raises(ValueError):
            tk.parse_site_diff(bad)


# ============================================================================== geometry dispatch + applier (§3.1)
class _Flat:
    def __init__(self, **kw):
        self.proportion = 0.2
        self.clear_radius = 1.7
        self.ring_outer_radius = 4.80
        self.noise_range = (0.02, 0.02)
        self.__dict__.update(kw)


class _SiteField(_Flat):
    def __init__(self, n_blocks, **kw):
        super().__init__(n_blocks=n_blocks, bar_fraction=0.4, pit_fraction=0.2, vertical_edges=False, min_gap=0.0,
                         bar_length_range=(1.0, 2.0), bar_thickness_range=(0.16, 0.32), slab_side_range=(0.6, 1.2),
                         **kw)


class _Site(_Flat):
    def __init__(self, kind, **kw):
        super().__init__(kind=kind, site_d_range=(0.0, 1.0), gen_d_range=(0.0, 1.0), ramp_max_deg=20.0, **kw)


CLASSES = (_Flat, _SiteField, _Site)
SITE_ENV = {tk.GEOM_VAR: "site"}


def _site_generator():
    subs = {"parkour_flat": _Flat(), "site_field_sparse": _SiteField(14), "site_field_dense": _SiteField(28)}
    for name, key in zip(sr.SITE_KINDS[3:], sr.SITE_SUBTERRAIN_KEYS[3:]):
        subs[key] = _Site(name)
    return types.SimpleNamespace(num_cols=20, difficulty_range=(0.2, 1.0), sub_terrains=subs)


def _env_cfg(tg=None):
    return types.SimpleNamespace(
        scene=types.SimpleNamespace(terrain=types.SimpleNamespace(terrain_generator=tg or _site_generator())),
        parkours=types.SimpleNamespace(base_parkour=types.SimpleNamespace(clear_radius_m=2.0, ring_outer_radius_m=4.5)),
        events=types.SimpleNamespace(reset_root_state=types.SimpleNamespace(params={"offset": 0.0})),
    )


def _snapshot(cfg):
    return copy.deepcopy(cfg)


def _same(a, b):
    """Deep compare two stand-in cfg trees (``SimpleNamespace`` / stand-in classes by their ``__dict__``)."""
    if hasattr(a, "__dict__") and not isinstance(a, type):
        return type(a) is type(b) and _same(vars(a), vars(b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


class TestDispatch:
    def test_require_omni_geom_accepts_site_and_its_knob_families(self):
        assert tk.require_omni_geom(SITE_ENV) == "site"
        assert tk.require_omni_geom({tk.GEOM_VAR: " Site "}) == "site"
        assert tk.require_omni_geom(dict(SITE_ENV, **{v: "1" for v in tk.FIELD_VARS})) == "site"   # FIELD_* allowed
        for var in tk.RING_VARS:
            with pytest.raises(RuntimeError, match=var):
                tk.require_omni_geom(dict(SITE_ENV, **{var: "2.0"}))

    @pytest.mark.parametrize("geom", ["ring", "field"])
    def test_site_knobs_are_ignored_with_a_print_under_ring_and_field(self, geom, capsys, monkeypatch):
        called = []
        monkeypatch.setattr(tk, "apply_omni_field_knobs", lambda cfg, env, classes=None: called.append("field"))
        monkeypatch.setattr(tk, "apply_ring_geometry", lambda *a, **k: called.append("ring"))
        monkeypatch.setattr(tk, "assert_ring_generator", lambda *a, **k: None)
        cfg = types.SimpleNamespace(scene=types.SimpleNamespace(terrain=types.SimpleNamespace(
            terrain_generator=types.SimpleNamespace(sub_terrains={}))))
        env = {tk.GEOM_VAR: geom, tk.SITE_MIX_VAR: "flat=1", tk.SITE_DIFF_VAR: "0:0.5"}
        tk.apply_omni_ring_knobs(cfg, env, classes=(object, object))
        out = capsys.readouterr().out
        assert f"[KNOB] {tk.SITE_MIX_VAR}=flat=1 ignored (geometry is {geom})" in out
        assert f"[KNOB] {tk.SITE_DIFF_VAR}=0:0.5 ignored (geometry is {geom})" in out
        assert called == [geom]

    def test_site_dispatches_to_the_site_applier(self, monkeypatch):
        seen = []
        monkeypatch.setattr(tk, "apply_omni_site_knobs", lambda cfg, env, classes=None: seen.append(classes))
        tk.apply_omni_ring_knobs(types.SimpleNamespace(), SITE_ENV, site_classes=CLASSES)
        assert seen == [CLASSES]

    def test_install_deep_copies_the_site_generator(self):
        cfg = types.SimpleNamespace(scene=types.SimpleNamespace(terrain=types.SimpleNamespace(terrain_generator="ring")))
        gen = _site_generator()
        assert tk.install_omni_terrain_generator(cfg, SITE_ENV, site_generator=gen) == "site"
        installed = cfg.scene.terrain.terrain_generator
        assert installed is not gen and _same(installed, gen)
        cfg2 = types.SimpleNamespace(scene=types.SimpleNamespace(terrain=types.SimpleNamespace(terrain_generator="r")))
        assert tk.install_omni_terrain_generator(cfg2, {tk.GEOM_VAR: "ring"}, site_generator=gen) == "ring"
        assert cfg2.scene.terrain.terrain_generator == "r"


class TestSiteApplier:
    def test_defaults(self, capsys):
        cfg = _env_cfg()
        tk.apply_omni_site_knobs(cfg, SITE_ENV, classes=CLASSES)
        tg = cfg.scene.terrain.terrain_generator
        assert tg.num_cols == 20
        mix = tk.parse_site_mix(tk.DEFAULT_SITE_MIX)
        for name, key in zip(sr.SITE_KINDS, sr.SITE_SUBTERRAIN_KEYS):
            sub = tg.sub_terrains[key]
            assert sub.proportion == mix[name] / 20 and sub.clear_radius == 1.7
        for key in sr.SITE_SUBTERRAIN_KEYS[3:]:
            sub = tg.sub_terrains[key]
            assert sub.site_d_range == (0.0, 1.0) and sub.gen_d_range == (0.2, 1.0) and sub.ramp_max_deg == 20.0
        assert (cfg.parkours.base_parkour.clear_radius_m, cfg.parkours.base_parkour.ring_outer_radius_m) == (1.7, 4.8)
        assert cfg.events.reset_root_state.params == {"offset": 0.0, "spawn_level_z": True}
        assert tg.sub_terrains["site_field_sparse"].n_blocks == 14 and tg.sub_terrains["site_field_dense"].n_blocks == 28
        assert "[KNOB]" not in capsys.readouterr().out

    def test_armed_knobs(self, capsys):
        cfg = _env_cfg()
        env = dict(SITE_ENV, **{
            tk.SITE_MIX_VAR: "flat=1,pits=3,walls=2", tk.SITE_DIFF_VAR: "0.25:0.75",
            tk.FIELD_BLOCKS_VAR: "10:20", tk.FIELD_KINDS_VAR: "0:0:1", tk.FIELD_EDGES_VAR: "wall",
            tk.FIELD_SIZE_SCALE_VAR: "0.6", tk.FIELD_PAD_R_VAR: "1.65", tk.FLAT_FRAC_VAR: "0.8",
        })
        tk.apply_omni_site_knobs(cfg, env, classes=CLASSES)
        tg = cfg.scene.terrain.terrain_generator
        assert tg.num_cols == 6
        assert tg.sub_terrains["parkour_flat"].proportion == 1 / 6 and tg.sub_terrains["site_pits"].proportion == 0.5
        assert tg.sub_terrains["site_trench"].proportion == 0.0
        for key in ("site_field_sparse", "site_field_dense"):
            sub = tg.sub_terrains[key]
            assert sub.bar_fraction == 0.0 and sub.pit_fraction == 1.0 and sub.vertical_edges is True
            assert sub.slab_side_range == pytest.approx((0.36, 0.72))
        assert (tg.sub_terrains["site_field_sparse"].n_blocks, tg.sub_terrains["site_field_dense"].n_blocks) == (10, 20)
        assert all(tg.sub_terrains[k].clear_radius == 1.65 for k in sr.SITE_SUBTERRAIN_KEYS)
        for key in sr.SITE_SUBTERRAIN_KEYS[3:]:
            sub = tg.sub_terrains[key]
            # rough-terrain port: no site foot friction, so the ramp cap stays at 20 deg
            assert sub.site_d_range == (0.25, 0.75) and sub.ramp_max_deg == 20.0
            assert not hasattr(sub, "n_blocks")                               # FIELD_* never reach the hazards
        out = capsys.readouterr().out
        assert f"[KNOB] {tk.SITE_MIX_VAR}=flat=1,pits=3,walls=2 consumed" in out
        assert f"[KNOB] {tk.SITE_DIFF_VAR}=0.25:0.75 consumed" in out
        assert f"[KNOB] {tk.FLAT_FRAC_VAR}=0.8 ignored (geometry is site" in out
        assert cfg.parkours.base_parkour.clear_radius_m == 1.65

    @pytest.mark.parametrize("env,match", [
        ({"KRABBY_RSI_FRAC": "0.3"}, "KRABBY_RSI_FRAC"),
        ({tk.FIELD_PAD_R_VAR: "2.0"}, "above the site pad"),
        ({tk.FIELD_PAD_R_VAR: "1.5"}, "below"),
        ({tk.SITE_MIX_VAR: "flat=0"}, tk.SITE_MIX_VAR),
        ({tk.SITE_DIFF_VAR: "0.9:0.1"}, tk.SITE_DIFF_VAR),
    ])
    def test_refusals(self, env, match):
        with pytest.raises((RuntimeError, ValueError), match=match):
            tk.apply_omni_site_knobs(_env_cfg(), dict(SITE_ENV, **env), classes=CLASSES)

    def test_rsi_zero_is_accepted(self):
        tk.apply_omni_site_knobs(_env_cfg(), dict(SITE_ENV, KRABBY_RSI_FRAC="0"), classes=CLASSES)

    def test_roughness_cap(self):
        tg = _site_generator()
        tg.sub_terrains["site_rubble"].noise_range = (0.02, 0.06)
        with pytest.raises(RuntimeError, match="site_rubble"):
            tk.apply_omni_site_knobs(_env_cfg(tg), SITE_ENV, classes=CLASSES)

    def test_class_assertion(self):
        tg = _site_generator()
        tg.sub_terrains["site_walls"] = _Site("chests")
        with pytest.raises(RuntimeError, match="site_walls"):
            tk.apply_omni_site_knobs(_env_cfg(tg), SITE_ENV, classes=CLASSES)
        tg = _site_generator()
        tg.sub_terrains["site_field_dense"] = _Flat()
        with pytest.raises(RuntimeError, match="site_field_dense"):
            tk.apply_omni_site_knobs(_env_cfg(tg), SITE_ENV, classes=CLASSES)
        ring = types.SimpleNamespace(num_cols=20, difficulty_range=(0, 1),
                                     sub_terrains={"parkour_flat": _Flat(), "omni_ring_sparse": _Flat()})
        with pytest.raises(RuntimeError, match="install_omni_terrain_generator"):
            tk.apply_omni_site_knobs(_env_cfg(ring), SITE_ENV, classes=CLASSES)
        cfg = _env_cfg()
        cfg.scene.terrain.terrain_generator = None
        with pytest.raises(RuntimeError, match="OMNI_SITE_TERRAINS_CFG"):
            tk.apply_omni_site_knobs(cfg, SITE_ENV, classes=CLASSES)

    def test_ring_and_field_cfgs_untouched_by_site_knobs(self, monkeypatch):
        """Neutrality (§10.1.6): under ring / field the site knobs write nothing (the cfg after the dispatcher
        with SITE_* set equals the cfg after it without them)."""
        monkeypatch.setattr(tk, "apply_ring_geometry", lambda *a, **k: None)
        monkeypatch.setattr(tk, "assert_ring_generator", lambda *a, **k: None)
        monkeypatch.setattr(tk, "apply_omni_field_knobs", lambda *a, **k: None)
        for geom in ("ring", "field"):
            a, b = _env_cfg(), _env_cfg()
            tk.apply_omni_ring_knobs(a, {tk.GEOM_VAR: geom}, classes=(object, object))
            tk.apply_omni_ring_knobs(b, {tk.GEOM_VAR: geom, tk.SITE_MIX_VAR: "pits=4", tk.SITE_DIFF_VAR: "0:0.2"},
                                     classes=(object, object))
            assert _same(a, b)


class TestSiteLadderKnob:
    """``KRABBY_SITE_LADDER`` (PLAN_rev1 B4): blank writes nothing; ``b4`` swaps every hazard column's body."""

    def test_blank_writes_nothing(self, capsys):
        for env in (SITE_ENV, dict(SITE_ENV, **{tk.SITE_LADDER_VAR: ""}), dict(SITE_ENV, **{tk.SITE_LADDER_VAR: " "})):
            a, b = _env_cfg(), _env_cfg()
            tk.apply_omni_site_knobs(a, SITE_ENV, classes=CLASSES)
            tk.apply_omni_site_knobs(b, env, classes=CLASSES, ladder_bodies={"b4": "never used"})
            assert _same(a, b)
            assert not any(hasattr(s, "function") for s in b.scene.terrain.terrain_generator.sub_terrains.values())
        assert tk.SITE_LADDER_VAR not in capsys.readouterr().out

    def test_b4_swaps_every_hazard_body_and_nothing_else(self, capsys):
        a, b = _env_cfg(), _env_cfg()
        tk.apply_omni_site_knobs(a, SITE_ENV, classes=CLASSES)
        tk.apply_omni_site_knobs(b, dict(SITE_ENV, **{tk.SITE_LADDER_VAR: "B4"}), classes=CLASSES)   # real bodies
        subs = b.scene.terrain.terrain_generator.sub_terrains
        for key in sr.SITE_SUBTERRAIN_KEYS[3:]:
            assert subs[key].function is st.omni_site_terrain_b4 and st.ladder_of(subs[key].function) == "b4"
            del subs[key].function
        assert _same(a, b)                                                # the body is the only write
        assert f"[KNOB] {tk.SITE_LADDER_VAR}=B4 consumed" in capsys.readouterr().out

    def test_injected_bodies_and_refusal(self):
        cfg = _env_cfg()
        tk.apply_omni_site_knobs(cfg, dict(SITE_ENV, **{tk.SITE_LADDER_VAR: "b4"}), classes=CLASSES,
                                 ladder_bodies={"b4": "body-b4"})
        assert {cfg.scene.terrain.terrain_generator.sub_terrains[k].function
                for k in sr.SITE_SUBTERRAIN_KEYS[3:]} == {"body-b4"}
        with pytest.raises(ValueError, match=tk.SITE_LADDER_VAR):
            tk.apply_omni_site_knobs(_env_cfg(), dict(SITE_ENV, **{tk.SITE_LADDER_VAR: "b5"}), classes=CLASSES)

    @pytest.mark.parametrize("geom", ["ring", "field"])
    def test_ignored_with_a_print_under_ring_and_field(self, geom, capsys, monkeypatch):
        monkeypatch.setattr(tk, "apply_ring_geometry", lambda *a, **k: None)
        monkeypatch.setattr(tk, "assert_ring_generator", lambda *a, **k: None)
        monkeypatch.setattr(tk, "apply_omni_field_knobs", lambda *a, **k: None)
        a, b = _env_cfg(), _env_cfg()
        tk.apply_omni_ring_knobs(a, {tk.GEOM_VAR: geom}, classes=(object, object))
        tk.apply_omni_ring_knobs(b, {tk.GEOM_VAR: geom, tk.SITE_LADDER_VAR: "b4"}, classes=(object, object))
        assert _same(a, b)
        assert f"[KNOB] {tk.SITE_LADDER_VAR}=b4 ignored (geometry is {geom})" in capsys.readouterr().out


def test_real_site_cfg_keys_and_classes():
    """Under Kit: OMNI_SITE_TERRAINS_CFG carries SITE_SUBTERRAIN_KEYS in order, with the right classes and the
    default-mix proportions, and the site applier accepts it with its real classes (skipped without pxr)."""
    try:
        from parkour_isaaclab.terrains.omni.config.omni_site import OMNI_SITE_TERRAINS_CFG
        from parkour_isaaclab.terrains.omni.omni_terrains_cfg import (
            OmniFieldTerrainCfg,
            OmniFlatTerrainCfg,
            OmniSiteFieldTerrainCfg,
            OmniSiteTerrainCfg,
        )
    except (ModuleNotFoundError, ImportError) as exc:  # no pxr / no Kit
        pytest.skip(f"isaaclab terrain cfgs need a Kit process: {exc}")
    subs = OMNI_SITE_TERRAINS_CFG.sub_terrains
    assert tuple(subs) == sr.SITE_SUBTERRAIN_KEYS
    assert type(subs["parkour_flat"]) is OmniFlatTerrainCfg
    for key in ("site_field_sparse", "site_field_dense"):
        assert type(subs[key]) is OmniSiteFieldTerrainCfg and isinstance(subs[key], OmniFieldTerrainCfg)
    for name, key in zip(sr.SITE_KINDS[3:], sr.SITE_SUBTERRAIN_KEYS[3:]):
        assert type(subs[key]) is OmniSiteTerrainCfg and subs[key].kind == name
    mix = tk.parse_site_mix(tk.DEFAULT_SITE_MIX)
    assert OMNI_SITE_TERRAINS_CFG.num_cols == sum(mix.values())
    for name, key in zip(sr.SITE_KINDS, sr.SITE_SUBTERRAIN_KEYS):
        assert subs[key].proportion == pytest.approx(mix[name] / 20)
    tk.assert_site_sub_terrains(subs, OmniFlatTerrainCfg, OmniSiteFieldTerrainCfg, OmniSiteTerrainCfg)


# ============================================================================== spawn on raised / sunken pads (§3.8)
_EV_PKG = "_omni_mdp_site_terrain_test"


def _omni_events():
    """``mdp/omni_events.py`` under a private stub package (``test_crab_hex_omni_rsi.py``'s technique)."""
    parkours = REPO / "parkour/parkour_isaaclab/envs/mdp/parkours"
    name = "parkour_isaaclab.envs.mdp.parkours.exposure_stats"
    if name not in sys.modules:
        parts = name.split(".")
        directory = parkours
        for i in range(len(parts) - 1, 0, -1):
            pkg_name = ".".join(parts[:i])
            pkg = sys.modules.get(pkg_name)
            if pkg is None:
                pkg = types.ModuleType(pkg_name)
                pkg.__path__ = [str(directory)]
                sys.modules[pkg_name] = pkg
            elif str(directory) not in list(getattr(pkg, "__path__", [])):
                pkg.__path__ = list(getattr(pkg, "__path__", [])) + [str(directory)]
            directory = directory.parent
        mod = _load(parkours / "exposure_stats.py", name)
        setattr(sys.modules[".".join(parts[:-1])], parts[-1], mod)
    isaaclab = sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    managers = sys.modules.get("isaaclab.managers")
    if managers is None:
        managers = types.ModuleType("isaaclab.managers")
        sys.modules["isaaclab.managers"] = managers
    if not hasattr(managers, "SceneEntityCfg"):
        class SceneEntityCfg:  # noqa: D401 - stand-in
            def __init__(self, name="robot", **kw):
                self.name = name
        managers.SceneEntityCfg = SceneEntityCfg
    isaaclab.managers = managers
    if _EV_PKG not in sys.modules:
        spec = importlib.machinery.ModuleSpec(_EV_PKG, loader=None, is_package=True)
        spec.submodule_search_locations = [str(OMNI_MDP)]
        sys.modules[_EV_PKG] = importlib.util.module_from_spec(spec)
    return importlib.import_module(f"{_EV_PKG}.omni_events"), importlib.import_module(f"{_EV_PKG}.site_raster")


class _Robot:
    def __init__(self, n):
        default = torch.zeros(n, 13)
        default[:, 2] = 1.05
        default[:, 3] = 1.0
        self.data = types.SimpleNamespace(default_root_state=default)
        self.writes = {}

    def write_root_pose_to_sim(self, pose, env_ids):
        self.writes["pose"] = pose.clone()

    def write_root_velocity_to_sim(self, vel, env_ids):
        self.writes["vel"] = vel.clone()


class _Scene:
    def __init__(self, robot, origins, terrain):
        self._items = {"robot": robot}
        self.env_origins = origins
        self.terrain = terrain

    def __getitem__(self, key):
        return self._items[key]


def _spawn_env(n=6, with_layers=True):
    ev, raster = _omni_events()
    robot = _Robot(n)
    origins = torch.zeros(n, 3)
    origins[:, 0] = torch.arange(n).float() * 10.0 - 15.0
    origins[:, 2] = 0.37                                     # the importer's origin z (max height near the centre)
    levels = torch.tensor([0, 1, 2, 0, 1, 2])[:n]
    types_ = torch.tensor([0, 1, 2, 3, 0, 1])[:n]
    terrain = types.SimpleNamespace(terrain_levels=levels, terrain_types=types_)
    parkour = types.SimpleNamespace(note_spawn_xy=lambda *a, **k: None)
    env = types.SimpleNamespace(
        device="cpu", scene=_Scene(robot, origins, terrain),
        parkour_manager=types.SimpleNamespace(active_terms=["base_parkour"], get_term=lambda name: parkour),
    )
    centre = torch.zeros(3, 4)
    centre[1, 1], centre[2, 2], centre[0, 3] = 0.25, -0.20, 0.10
    if with_layers:
        frame = raster.TileFrame(size_xy=(10.0, 10.0), hs=0.08, vs=0.005, tile_px=(2, 2), grid_rc=(3, 4))
        z = np.zeros((3, 4, 2, 2))
        setattr(env, raster.SITE_LAYERS_ATTR, raster.make_test_layers(z.astype(np.int8), z, z, frame,
                                                                     tile_centre_level_m=centre))
    else:
        setattr(env, raster.SITE_LAYERS_ATTR, False)
    want = centre[levels, types_]
    return ev, env, robot, want


def _spawn(ev, env, robot, **kw):
    torch.manual_seed(7)
    ev.reset_root_state_ring(env, torch.arange(robot.data.default_root_state.shape[0]), **kw)
    return robot.writes["pose"]


def test_spawn_level_z_off_is_todays_spawn_and_on_adds_the_pad_level():
    ev, env, robot, want = _spawn_env()
    base = _spawn(ev, env, robot)
    off = _spawn(ev, env, robot, spawn_level_z=False)
    assert torch.equal(base, off)
    assert torch.equal(base[:, 2], torch.full((6,), 1.05))                       # origin z dropped, as today
    on = _spawn(ev, env, robot, spawn_level_z=True)
    assert torch.equal(on[:, [0, 1, 3, 4, 5, 6]], base[:, [0, 1, 3, 4, 5, 6]])
    assert torch.allclose(on[:, 2], base[:, 2] + want)
    assert want.abs().sum() > 0


def test_spawn_level_z_without_site_layers_is_zero():
    ev, env, robot, _ = _spawn_env(with_layers=False)
    assert torch.equal(_spawn(ev, env, robot, spawn_level_z=True), _spawn(ev, env, robot))


# ============================================================================== rough-terrain port: refusals
@pytest.mark.parametrize("var", tk.SITE_UNPORTED_VARS)
@pytest.mark.parametrize("geom", ("site", "field", "ring"))
def test_unported_site_knobs_are_refused_in_every_geometry(var, geom):
    """Every site-teacher knob this branch does not carry is refused when set (never silently inert), in every
    geometry, through the dispatcher; blank counts as unset."""
    env = {tk.GEOM_VAR: geom, var: "1"}
    with pytest.raises(RuntimeError, match=var):
        tk.apply_omni_ring_knobs(_env_cfg(), env, site_classes=CLASSES)
    tk.refuse_unported_site_knobs({tk.GEOM_VAR: geom, var: "  "})


def test_unported_list_is_the_site_branch_strip_set_plus_the_critic_group():
    """The refused set is exactly the site branch's ``site_keys.EVAL_STRIP_KEYS`` (19) + ``EVAL_CARRY_KEYS`` (the
    critic group), restated; disjoint from the ported terrain knobs."""
    assert len(tk.SITE_UNPORTED_VARS) == 20 and len(set(tk.SITE_UNPORTED_VARS)) == 20
    assert not set(tk.SITE_UNPORTED_VARS) & set(tk.SITE_VARS)
    assert tk.site_friction_floor({"KRABBY_SITE_FOOT_FRICTION": "0.25:1.25"}) is None


#: The rough-terrain plan's wall-free site kinds (``experiments/2026-11-10_rough_terrain/PLAN.md`` Design): no chests /
#: walls (blocker layouts stay in ``site_terrains`` but get 0 columns).
ROUGH_MIX = "flat=3,field_sparse=2,field_dense=1,pits=2,ledge_down=2,ledge_up=2,rubble=2"
ROUGH_KINDS = {"pits": K.KIND_PIT, "ledge_down": K.KIND_LEDGE, "ledge_up": K.KIND_LEDGE, "rubble": K.KIND_RUBBLE}


@pytest.mark.parametrize("ladder", ("", "b4"))
def test_rough_terrain_mix_builds_the_needed_site_kinds(ladder):
    """``KRABBY_SITE_MIX`` with chests / walls at 0 + ``KRABBY_SITE_DIFF`` + ``KRABBY_SITE_LADDER``: exactly the mix's
    columns, no blocker column, and every needed kind (ledge_up, ledge_down, pits, rubble) builds through the
    decorated body at the generator's difficulty endpoints, on the applied d mapping and ladder, with its own kind in
    the raster, no blocker pixel, and the pad level the spawn reads (raised for ledge_down, sunk for ledge_up)."""
    cfg = _env_cfg()
    env = dict(SITE_ENV, **{tk.SITE_MIX_VAR: ROUGH_MIX, tk.SITE_DIFF_VAR: "0.0:1.0", tk.SITE_LADDER_VAR: ladder})
    tk.apply_omni_site_knobs(cfg, env, classes=CLASSES)
    tg = cfg.scene.terrain.terrain_generator
    assert tg.num_cols == 14
    for blocker in ("site_chests", "site_walls"):
        assert tg.sub_terrains[blocker].proportion == 0.0
    cols = _curriculum_columns([tg.sub_terrains[k].proportion for k in sr.SITE_SUBTERRAIN_KEYS], tg.num_cols)
    assert sorted({sr.SITE_KINDS[i] for i in cols}) == sorted(
        ("flat", "field_sparse", "field_dense", "pits", "ledge_down", "ledge_up", "rubble"))
    for kind, pixel_kind in ROUGH_KINDS.items():
        sub = tg.sub_terrains[f"site_{kind}"]
        body = st.SITE_LADDER_BODIES[ladder] if ladder else st.omni_site_terrain
        assert (sub.function is body) if ladder else not hasattr(sub, "function")
        for d in tg.difficulty_range:
            np.random.seed(5)
            out = body(d, site_cfg(kind, size=(10.0, 10.0), apply_roughness=True, site_d_range=sub.site_d_range,
                                   gen_d_range=sub.gen_d_range, clear_radius=sub.clear_radius), 8)
            layers = out[7]
            sr.check_tile_layers(layers, out[5].shape)
            assert (layers["kind"] == pixel_kind).any(), (kind, d)
            assert not np.isin(layers["kind"], K.BLOCKER_KINDS).any(), (kind, d)
            centre = layers["tile"]["centre_level_m"]
            if kind == "ledge_down":
                assert centre > 0.0
            elif kind == "ledge_up":
                assert centre < 0.0
            else:
                assert centre == 0.0
    assert cfg.events.reset_root_state.params["spawn_level_z"] is True
