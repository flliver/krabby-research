"""Omni phase 2 -- the pre-launch audit fixes (2026-10-06), CPU only: nothing here starts Isaac/Kit or a GPU.

* B1  ``scripts/p2_smoke.py`` (the CHARTER 5 smokes S1-S4: the plan composed with run_candidates' own functions, the
      pure judges, the parent's refusals), ``p2_read.py --smoke-s5`` / ``--write-smokes-ok`` / ``--smokes-ok-check``,
      and ``p2_queue.sh`` refusing to start without a valid ``$OUT/SMOKES_OK``;
* M1  the code-state guard (``p2_live.py --code-guard``; ``run_arm.sh`` run_ref and every --row-eval / --lr-eval);
* M2  the CAL2 gate before ``P2__A1_s18`` (``p2_read.py --cal2-gate``; ``p2_queue.sh``);
* M3  ``p2_live.dump_facts``: a DR event dumped as ``null`` is absent;
* M4  REPLAY_GATE.md / p2_replay.py no longer say preflight ignores ``verdict``; the queue header has no E1 step;
* M5  after a driver infra retry the watcher reads ``<tag>_r<k>_train.log``.

The shell scripts are exercised through stubs: ``P2_PY`` (a stub python answering --code-guard / --preflight),
``P2_CAP=""`` (no systemd scope), ``P2_OUT`` / ``CAMPAIGN_DIR`` in tmp, ``RUN_ARM`` / ``READ`` stubs for the queue, and
``P2_RUN_ARM_LIB=1`` to source run_arm.sh's functions without running its command line.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = OMNI / "experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
RUN_CANDIDATES = OMNI / "experiments/tools/run_candidates.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def smoke():
    return _load(SCRIPTS / "p2_smoke.py", "_t_p2_smoke")


@pytest.fixture(scope="module")
def live():
    return _load(SCRIPTS / "p2_live.py", "_t_p2_live_prelaunch")


@pytest.fixture(scope="module")
def read():
    return _load(SCRIPTS / "p2_read.py", "_t_p2_read_prelaunch")


@pytest.fixture(scope="module")
def rc():
    return _load(RUN_CANDIDATES, "_t_run_candidates_prelaunch")


@pytest.fixture(scope="module")
def cands(rc):
    return rc.load_candidates(CAMP / "candidates.yaml")


@pytest.fixture(scope="module")
def anchors():
    return yaml.safe_load(open(CAMP / "anchors.yaml"))


# ===================================================================================== B1: p2_smoke.py, the plan
def test_the_smoke_env_is_the_driver_p2a_env_of_the_a1_chain(smoke, rc, cands, anchors):
    plan = smoke.compose_plan(rc, cands, "P2__A1_s18", anchors=anchors)
    recipe = rc.reference_recipe(cands, "P2__A1_s18")
    want = rc.stage_env(recipe, "p2a")
    assert plan["task"] == rc.ph.PHASES["2a"].task == "Isaac-Crab-Hex-Omni-Teacher-v0"
    assert plan["base"] == "2a" and plan["stage"] == "p2a" and plan["teacher_mode"] == "2a"
    env = plan["child_env_vars"]
    assert env["KRABBY_OMNI_PHASE"] == "2a" and env["KRABBY_PLANT"] == cands["plant"] == "A15+B"
    for k, v in want.items():
        if smoke.is_placeholder(v):
            continue
        assert env[k] == v, k
    # A1's PROMOTE: while it is a placeholder (candidates_v5) the smoke launches at the wall-mode CAL1's value and
    # records the substitution; once calibrated (v6, CHARTER 9) the smoke runs the chain's own value, unsubstituted
    assert smoke.DEFAULT_FILL == "P2__CALW1_s18"
    if smoke.is_placeholder(want["KRABBY_TERRAIN_PROMOTE"]):
        assert env["KRABBY_TERRAIN_PROMOTE"] == "0.45:0.25"
        assert plan["substituted"]["KRABBY_TERRAIN_PROMOTE"]["from"] == "P2__CALW1_s18"
    else:
        assert env["KRABBY_TERRAIN_PROMOTE"] == want["KRABBY_TERRAIN_PROMOTE"]
        assert "KRABBY_TERRAIN_PROMOTE" not in (plan["substituted"] or {})
    assert env["KRABBY_FIELD_EDGES"] == "wall" and env["KRABBY_FLAT_TERRAIN_DIFF"] == "0.20:1.0"
    assert smoke.terrain_expect(env) == {"difficulty": (0.20, 1.0), "wall": True}
    assert not any(smoke.is_placeholder(v) for v in env.values())
    assert plan["effective_env"]["KRABBY_HEX_TEACHER_MODE"] == "2a"        # activate_phase's 2a preset
    assert plan["effective_env"]["KRABBY_FLAT_TERRAIN_GEOM"] == "field"
    assert plan["seed"] == 18
    assert plan["head"]["sha256"] == anchors["anchors"][18]["sha256"]
    assert plan["head"]["checkpoint"].endswith(anchors["anchors"][18]["checkpoint"])


def test_the_child_environ_strips_parent_krabby_vars(smoke, rc, cands):
    plan = smoke.compose_plan(rc, cands, "P2__A1_s18")
    env = smoke.child_environ(rc, plan, parent={"PATH": "/usr/bin", "KRABBY_LEAK": "1", "KRABBY_OMNI_PHASE": "1c"})
    assert "KRABBY_LEAK" not in env and env["KRABBY_OMNI_PHASE"] == "2a"
    assert env["OMNI_KIT_ACCEPT_EULA"] == "yes" and env["PATH"] == "/usr/bin"
    assert "KRABBY_LEAK" not in os.environ                  # the caller's environment is restored


def test_a_placeholder_without_a_fill_is_dropped_not_launched(smoke, rc, cands):
    import copy
    cands = copy.deepcopy(cands)                    # since v6 A1's PROMOTE is filled: plant a placeholder in a copy
    ov = cands["chains"]["P2__A1_s18"]["stages"][0]["overlay"]
    ov["KRABBY_TERRAIN_PROMOTE"] = "TBD_PENDING_CALIBRATION_WALLS"
    plan = smoke.compose_plan(rc, cands, "P2__A1_s18", fill_from=None)
    assert "KRABBY_TERRAIN_PROMOTE" not in plan["child_env_vars"]
    assert plan["dropped"] == ["KRABBY_TERRAIN_PROMOTE"]


def test_placeholder_rule_matches_run_arm(smoke):
    assert smoke.is_placeholder("TBD_PENDING_CALIBRATION_S5") and smoke.is_placeholder(" tbd") and \
        smoke.is_placeholder("x_PENDING")
    assert not smoke.is_placeholder("0.45:0.25") and not smoke.is_placeholder("")


def test_the_isaac_match_is_anchored(smoke):
    m = smoke.ISAAC_RE.match
    assert m("/home/u/isaac_venv/bin/python /r/parkour/scripts/rsl_rl/train.py --task X")
    assert m("/home/u/isaac_venv/bin/python /r/eval_crab_hex_gait.py --headless")
    assert not m("bash run_arm.sh --row-eval x eval_crab_hex_gait.py")
    assert not m("grep -E isaac_venv/bin/python train.py")
    assert not m("python3 p2_queue.sh")


# ===================================================================================== B1: the judges
def _s1_summary(smoke, **over):
    def sub(cls, p, **kw):
        d = {"class": cls, "proportion": p, "noise_range": [0.02, 0.02], "clear_radius": 1.7,
             "ring_outer_radius": 4.8, "n_blocks": None, "bar_fraction": None, "pit_fraction": None,
             "vertical_edges": None}
        d.update(kw)
        return d
    s = {"curriculum": True, "difficulty_range": [0.2, 0.7],
         "sub_terrains": {"parkour_flat": sub("OmniFlatTerrainCfg", 0.5),
                          "omni_field_sparse": sub("OmniFieldTerrainCfg", 0.25, n_blocks=14, bar_fraction=0.4,
                                                   pit_fraction=0.2, vertical_edges=False),
                          "omni_field_dense": sub("OmniFieldTerrainCfg", 0.25, n_blocks=28, bar_fraction=0.4,
                                                  pit_fraction=0.2, vertical_edges=False)},
         "parkour": {"clear_radius_m": 1.7, "ring_outer_radius_m": 4.8}}
    for path, v in over.items():
        node = s
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = v
    return s


def test_s1_passes_the_charter_cfg(smoke):
    r = smoke.judge_s1(_s1_summary(smoke))
    assert r["result"] == "PASS", [c for c in r["checks"] if not c[1]]


def test_s1_proportions_are_normalised(smoke):
    s = _s1_summary(smoke, **{"sub_terrains.parkour_flat.proportion": 1.0,
                              "sub_terrains.omni_field_sparse.proportion": 0.5,
                              "sub_terrains.omni_field_dense.proportion": 0.5})
    assert smoke.judge_s1(s)["result"] == "PASS"


@pytest.mark.parametrize("path,value", [
    ("curriculum", False), ("difficulty_range", [0.05, 0.2]),
    ("sub_terrains.parkour_flat.noise_range", [0.02, 0.06]),
    ("sub_terrains.parkour_flat.clear_radius", 2.0),
    ("sub_terrains.omni_field_dense.ring_outer_radius", 4.5),
    ("sub_terrains.omni_field_sparse.vertical_edges", True),
    ("sub_terrains.omni_field_sparse.pit_fraction", 0.0),
    ("sub_terrains.omni_field_dense.n_blocks", 14),
    ("sub_terrains.omni_field_dense.class", "OmniRingTerrainCfg"),
    ("sub_terrains.omni_field_dense.proportion", 0.5),
    ("parkour.clear_radius_m", 2.0), ("parkour.ring_outer_radius_m", 4.5),
])
def test_s1_fails_each_charter_value(smoke, path, value):
    assert smoke.judge_s1(_s1_summary(smoke, **{path: value}))["result"] == "FAIL"


def test_s1_follows_the_composed_terrain(smoke):
    """CHARTER 9 walls re-tune: S1 expects the composed env's edges and difficulty range (unset = ramp, 0.20:0.70)."""
    want = smoke.terrain_expect({"KRABBY_FIELD_EDGES": "wall", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:1.0"})
    assert want == {"difficulty": (0.2, 1.0), "wall": True}
    wall = _s1_summary(smoke, **{"difficulty_range": [0.2, 1.0], "sub_terrains.omni_field_sparse.vertical_edges": True,
                                 "sub_terrains.omni_field_dense.vertical_edges": True})
    assert smoke.judge_s1(wall, want)["result"] == "PASS"
    assert smoke.judge_s1(_s1_summary(smoke), want)["result"] == "FAIL"           # a ramp build under a wall arm
    assert smoke.judge_s1(wall)["result"] == "FAIL"                               # a wall build under the default
    assert smoke.terrain_expect(None) == {"difficulty": smoke.S1_DIFFICULTY, "wall": False}
    with pytest.raises(ValueError):
        smoke.terrain_expect({"KRABBY_FIELD_EDGES": "cliff"})


def test_s2_wall_mode_edge_checks(smoke):
    """Wall mode inverts the edge checks: every field tile walls, and no wall pixel sits off a block."""
    import numpy as np
    occ = np.zeros((20, 20), dtype=bool)
    occ[8:12, 8:12] = True
    edge = np.zeros_like(occ)
    edge[7, 8:12] = True                                                              # a riser beside the block
    h = np.where(occ, 0.05, 0.0)
    cfg = {"hurdle_height_range": "0.02, 0.06", "step_height": "0.05", "pit_depth": "0.05", "clear_radius": 0.1,
           "ring_outer_radius": 0.8, "n_blocks": 14, "n_pit": 0}
    st = smoke.tile_stats(h, occ, edge, 0.08, 0.5, cfg)
    assert st["edge_px"] == 4 and st["edge_px_off_block"] == 0
    names = {c[0] for c in smoke.judge_s2([{"key": "omni_field_sparse", "row": 0, "col": 0, "d": 0.5, "stats": st}],
                                          {"omni_field_sparse": cfg}, wall=True)["checks"]}
    assert any("walls present" in n for n in names) and not any("ramp mode" in n for n in names)
    far = edge.copy()
    far[0, 0] = True                                                                  # a wall pixel far from any block
    st2 = smoke.tile_stats(h, occ, far, 0.08, 0.5, cfg)
    assert st2["edge_px_off_block"] == 1
    r = smoke.judge_s2([{"key": "omni_field_sparse", "row": 0, "col": 0, "d": 0.5, "stats": st2}],
                       {"omni_field_sparse": cfg}, wall=True)
    assert not dict((c[0], c[1]) for c in r["checks"])[
        "every wall pixel within 2 px of a block (wall mode; roughness never walls)"]


def test_s1_fails_a_ring_generator(smoke):
    s = _s1_summary(smoke)
    s["sub_terrains"] = {"parkour_flat": s["sub_terrains"]["parkour_flat"],
                         "omni_ring_sparse": dict(s["sub_terrains"]["omni_field_sparse"], **{"class": "OmniRingTerrainCfg"})}
    assert smoke.judge_s1(s)["result"] == "FAIL"


def test_row_difficulty(smoke):
    assert smoke.row_difficulty(0, 10, 0.2, 0.7, True, False) == pytest.approx(0.2)
    assert smoke.row_difficulty(9, 10, 0.2, 0.7, True, False) == pytest.approx(0.7)
    assert smoke.row_difficulty(0, 10, 0.2, 0.7, True, True) == pytest.approx(0.25)     # an upper bound
    assert smoke.row_difficulty(3, 10, 0.2, 0.7, False, False) == pytest.approx(0.7)


@pytest.fixture(scope="module")
def field_tiles():
    """Real decorated omni_field_terrain tiles (the fixture loader the field-terrain tests use), d 0.2 and 0.7."""
    if str(FIXTURES) not in sys.path:
        sys.path.insert(0, str(FIXTURES))
    import gen_omni_ring_golden as golden  # noqa: PLC0415

    _u, _f, omni = golden.load_terrain_modules()
    out = []
    for i, (n, d) in enumerate([(14, 0.2), (28, 0.2), (14, 0.2), (28, 0.2)]):
        cfg = types.SimpleNamespace(
            border_width=0.0, horizontal_scale=0.08, vertical_scale=0.005, platform_len=2.5, platform_height=0.0,
            slope_threshold=1.5, edge_width_thresh=0.05, use_simplified=False, size=(10.0, 10.0),
            apply_roughness=True, apply_flat=False, downsampled_scale=0.075, noise_range=(0.02, 0.02),
            noise_step=0.005, pad_width=0.1, pad_height=0.0, clear_radius=1.7, ring_outer_radius=4.8,
            field_margin=0.16, n_blocks=n, bar_fraction=0.4, pit_fraction=0.2, min_gap=0.0, max_tries=64,
            bar_length_range=(1.0, 2.0), bar_thickness_range=(0.16, 0.32), slab_side_range=(0.6, 1.2),
            hurdle_height_range="0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty",
            step_height="0.02 + 0.06 * difficulty", pit_depth="0.02 + 0.06 * difficulty", vertical_edges=False)
        np.random.seed(100 + i)
        res = omni.omni_field_terrain(d, cfg, 8)
        out.append({"key": "omni_field_sparse" if n == 14 else "omni_field_dense", "d": d, "h": res[5] * 0.005,
                    "occ": res[6], "edge": res[4], "cfg": vars(cfg)})
    return out


def _cfgs():
    base = {"hurdle_height_range": "0.02 + 0.05 * difficulty, 0.03 + 0.06 * difficulty",
            "step_height": "0.02 + 0.06 * difficulty", "pit_depth": "0.02 + 0.06 * difficulty",
            "clear_radius": 1.7, "ring_outer_radius": 4.8, "n_pit": 3}
    return {"omni_field_sparse": dict(base, n_blocks=14, n_pit=3), "omni_field_dense": dict(base, n_blocks=28, n_pit=6)}


def _tiles(smoke, raw):
    cf = _cfgs()
    return [{"key": t["key"], "row": 0, "col": i, "d": t["d"],
             "stats": smoke.tile_stats(t["h"], t["occ"], t["edge"], 0.08, t["d"], cf[t["key"]])}
            for i, t in enumerate(raw)]


def test_s2_tile_stats_on_the_real_field_function(smoke, field_tiles):
    for t in field_tiles:
        st = smoke.tile_stats(t["h"], t["occ"], t["edge"], 0.08, t["d"], _cfgs()[t["key"]])
        assert st["pad_abs_max"] <= smoke.PAD_ABS_MAX + 1e-9      # the pad is roughness only
        assert st["pad_occ_px"] == 0                               # and holds no block pixel
        want = 0.085 * (14 if t["key"].endswith("sparse") else 28) / 14
        assert 0.5 * want <= st["coverage"] <= 1.5 * want
        assert st["sunken_px"] > 0                                 # pits are sunken
        assert st["formula_top"] == pytest.approx(max(0.03 + 0.06 * t["d"], 0.02 + 0.06 * t["d"]))
        assert st["excess"] <= smoke.HEIGHT_TOL and st["pit_over"] <= smoke.HEIGHT_TOL
        assert st["edge_px"] == 0                                  # ramp mode is ramp-only (pit/raised gap)


def test_s2_thresholds_follow_their_derivation(smoke):
    """pad = realised roughness + 1 step + margin; heights / pits = realised roughness + 2 steps + margin; and each
    sits above the CPU-predicted value it was calibrated on."""
    assert smoke.PAD_ABS_MAX == pytest.approx(0.035 + 0.005 + 0.005)
    assert smoke.HEIGHT_TOL == pytest.approx(0.035 + 0.010 + 0.005)
    cpu = smoke.S2_CPU
    assert cpu["n_field_tiles"] >= 400 and cpu["pad_occ_px"] == 0 and cpu["edge_px"] == 0
    assert cpu["pad_abs_max"] < smoke.PAD_ABS_MAX
    assert cpu["max_excess"] < smoke.HEIGHT_TOL and cpu["max_pit_over"] < smoke.HEIGHT_TOL
    for k, lo in smoke.CORNER_RATE_MIN.items():
        assert cpu["corner_rate"][k] > lo
    assert smoke.CORNER_RATE_MIN == {"omni_field_sparse": 0.70, "omni_field_dense": 0.90}
    table = smoke.s2_threshold_table()
    assert all(r["cpu"] is not None for r in table) and len(table) == 10
    assert "CPU 0.0350" in smoke.format_s2_table()


def test_s2_cpu_replica_passes_and_matches_the_record(smoke):
    """A small CPU replica of the generator (2 seeded grids = 200 field tiles) passes S2 with the real terrain code."""
    res = smoke.cpu_s2_calibration(2)
    assert res["result"] == "PASS", [c for c in res["checks"] if not c[1]]
    assert res["n_field_tiles"] == 200 and res["pad_occ_px"] == 0 and res["edge_px"] == 0
    assert smoke.gen_column_keys() == ["parkour_flat"] * 10 + ["omni_field_sparse"] * 5 + ["omni_field_dense"] * 5


def test_s2_judge_fails_each_defect(smoke, field_tiles):
    base = [dict(t) for t in field_tiles]
    r0 = smoke.judge_s2(_tiles(smoke, base), _cfgs())
    names = [c[0] for c in r0["checks"]]
    assert len(names) == 8 and r0["n_field_tiles"] == 4 and r0["result"] == "PASS"
    pad_occ, pad_h, cover, _corners, heights, pits, edges_bp, edges_share = names

    def fails(bad, name):
        return not dict((c[0], c[1]) for c in smoke.judge_s2(_tiles(smoke, bad), _cfgs())["checks"])[name]

    # a raised pixel in the pad: the height check
    bad = [dict(t) for t in base]
    h = bad[0]["h"].copy()
    h[63, 63] = 0.05
    bad[0]["h"] = h
    assert fails(bad, pad_h) and not fails(bad, pad_occ)
    # a block pixel in the pad, however flat: the exact occupancy check
    bad = [dict(t) for t in base]
    o = bad[0]["occ"].copy()
    o[63, 63] = True
    bad[0] = dict(bad[0], occ=o)
    assert fails(bad, pad_occ) and not fails(bad, pad_h)
    # an empty tile: coverage out of band, no pits
    bad = [dict(t) for t in base]
    bad[1] = dict(bad[1], occ=np.zeros_like(bad[1]["occ"]), h=np.zeros_like(bad[1]["h"]))
    assert fails(bad, cover) and fails(bad, pits)
    # an edge pixel away from any pit (a single-block riser): not a raised-to-pit wall, but 1 of 4 tiles > the 2 % cap
    bad = [dict(t) for t in base]
    e = np.zeros_like(bad[2]["edge"], dtype=bool)
    e[10, 10] = True
    bad[2] = dict(bad[2], edge=e)
    assert fails(bad, edges_share) and not fails(bad, edges_bp)
    # a block taller than the formula + HEIGHT_TOL (0.05 m): 0.06 m over fails, 0.045 m over passes
    top = max(0.03 + 0.06 * 0.2, 0.02 + 0.06 * 0.2)
    for over, want_fail in ((0.06, True), (0.045, False)):
        bad = [dict(t) for t in base]
        h = bad[3]["h"].copy()
        h[5, 5] = top + over
        bad[3] = dict(bad[3], h=h)
        assert fails(bad, heights) is want_fail
    # a pit deeper than pit_depth + HEIGHT_TOL
    bad = [dict(t) for t in base]
    h = bad[3]["h"].copy()
    h[5, 5] = -(0.02 + 0.06 * 0.2 + 0.06)
    bad[3] = dict(bad[3], h=h)
    assert fails(bad, pits)
    # no field tile at all
    assert smoke.judge_s2([], _cfgs())["result"] == "FAIL"


def test_s2_corner_rule_is_the_pooled_per_corner_rate_per_key(smoke):
    def tile(key, corners):
        return {"key": key, "row": 0, "col": 0, "d": 0.2,
                "stats": {"pad_abs_max": 0.0, "pad_occ_px": 0, "coverage": 0.085 if key.endswith("sparse") else 0.17,
                          "corners": corners, "excess": 0.0, "max_h": 0.0, "formula_top": 0.04, "min_h": -0.03,
                          "pit_depth": 0.032, "pit_over": -0.002, "sunken_px": 5, "rough_amp": 0.02, "edge_px": 0,
                          "edge_px_block_pit": 0}}
    two, three, four = [True, True, False, False], [True, True, True, False], [True] * 4
    sp, de = "omni_field_sparse", "omni_field_dense"
    # sparse 0.75 (no tile has all 4 -- the old rule failed this), dense 15/16: PASS
    ok = smoke.judge_s2([tile(sp, three)] * 4 + [tile(de, four)] * 3 + [tile(de, three)], _cfgs())
    assert ok["result"] == "PASS", ok["checks"]
    assert ok["corner_rate"][sp] == pytest.approx(0.75) and ok["corner_rate"][de] == pytest.approx(0.9375)
    assert ok["all4_share"][sp] == 0.0
    # sparse 0.625 < 0.70: FAIL
    no = smoke.judge_s2([tile(sp, three), tile(sp, two)] * 2 + [tile(de, four)] * 2, _cfgs())
    assert no["result"] == "FAIL" and no["corner_rate"][sp] == pytest.approx(0.625)
    # dense 0.875 < 0.90 while sparse is fine: FAIL
    no = smoke.judge_s2([tile(sp, four)] * 2 + [tile(de, three)] * 2, _cfgs())
    assert no["result"] == "FAIL" and no["corner_rate"][de] == pytest.approx(0.75)


def _s3(**over):
    s = {"resets": 256, "levels_unchanged": True, "feet_over_block_t0": 0, "feet_below_terrain_t0": 0,
         "min_foot_clearance_t0": 0.02, "crab_failure_1s": 0, "toe_radius_max_m": 1.45}
    s.update(over)
    return s


def test_s3_passes_and_records_the_toe_radius(smoke):
    r = smoke.judge_s3(_s3())
    assert r["result"] == "PASS" and r["toe_radius_max_m"] == pytest.approx(1.45)


@pytest.mark.parametrize("over", [{"resets": 199}, {"levels_unchanged": False}, {"feet_over_block_t0": 1},
                                  {"feet_below_terrain_t0": 2}, {"crab_failure_1s": 1}, {"crab_failure_1s": None},
                                  {"toe_radius_max_m": 1.61}, {"toe_radius_max_m": None}])
def test_s3_fails(smoke, over):
    assert smoke.judge_s3(_s3(**over))["result"] == "FAIL"


def _means(**over):
    m = {"reach_edge_frac": 0.95, "field_frac_mean": 0.6, "block_steps_mean": 300.0, "off_tile_block_steps_mean": 4.0}
    m.update(over)
    return m


def test_s4_passes(smoke):
    r = smoke.judge_s4(_means(), {"obst_episodes": 250}, {})
    assert r["result"] == "PASS" and r["block_steps_mean"] == 300.0


@pytest.mark.parametrize("over", [{"reach_edge_frac": 0.89}, {"field_frac_mean": 0.50}, {"block_steps_mean": 229.0},
                                  {"off_tile_block_steps_mean": 0.0}, {"block_steps_mean": float("nan")},
                                  {"reach_edge_frac": None}])
def test_s4_fails_each_threshold(smoke, over):
    assert smoke.judge_s4(_means(**over), {"obst_episodes": 250}, {})["result"] == "FAIL"


def test_s4_fails_a_carry_difference_or_no_obstacle_episode(smoke):
    assert smoke.judge_s4(_means(), {"obst_episodes": 250}, {"KRABBY_STD_PARAM": ["soft", "0"]})["result"] == "FAIL"
    assert smoke.judge_s4(_means(), {"obst_episodes": 0}, {})["result"] == "FAIL"


def test_carry_diff_reads_defaults_and_numbers(smoke):
    d = {"KRABBY_STD_PARAM": "0"}
    assert smoke.carry_diff({"KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}, {"KRABBY_OMNI_FACING_OBS_CLIP": "0.60"},
                            ["KRABBY_OMNI_FACING_OBS_CLIP"], d) == {}
    assert smoke.carry_diff({}, {"KRABBY_STD_PARAM": "0"}, ["KRABBY_STD_PARAM"], d) == {}
    assert smoke.carry_diff({"KRABBY_STD_PARAM": "soft"}, {}, ["KRABBY_STD_PARAM"], d) == {"KRABBY_STD_PARAM": ["soft", "0"]}


def test_the_b_head_carry_equals_the_a1_env(smoke, rc, cands, anchors):
    """The S4 precondition as it will read at launch: the s18 anchor stamp vs the composed A1 env."""
    plan = smoke.compose_plan(rc, cands, "P2__A1_s18", anchors=anchors)
    meta_p = Path(plan["head"]["env_stamp"]) / "run_meta.json"
    if not meta_p.is_file():
        pytest.skip("the s18 anchor stamp is not on this machine")
    anchor_env = json.loads(meta_p.read_text())["env_vars"]
    eff = {k: v for k, v in plan["effective_env"].items() if k.startswith("KRABBY_")}
    assert smoke.carry_diff(anchor_env, eff, smoke.S4_CARRY_STRICT, rc.rp.EVAL_CARRY_DEFAULTS) == {}


def test_summary_and_all_pass(smoke):
    doc = {"plan": {"checks": list(smoke.CHECKS)}, "finished": True, "errors": [],
           "results": {"S1": {"result": "PASS", "checks": []}, "S2": {"result": "PASS", "checks": []},
                       "S3": smoke.judge_s3(_s3()), "S4": smoke.judge_s4(_means(), {"obst_episodes": 9}, {})}}
    assert smoke.all_pass(doc)
    lines = smoke.summary_lines(doc)
    assert lines[2].startswith("S3 PASS toe_radius_max 1.45 m") and "reach_edge_frac 0.95" in lines[3]
    doc["errors"] = ["boom"]
    assert not smoke.all_pass(doc)
    doc2 = {"plan": {"checks": ["S1"]}, "results": {}, "finished": True, "errors": []}
    assert smoke.summary_lines(doc2)[0].startswith("S1 FAIL (not run")


def _fake_plan(smoke, dirty=False, sha="x" * 64, ckpt=None):
    return {"checks": list(smoke.CHECKS), "code_state": {"dirty": dirty, "dirty_paths": ["a.py"] if dirty else []},
            "head": {"checkpoint": str(ckpt) if ckpt else "/nonexistent.pt", "sha256": sha}, "task": "T",
            "num_envs": 1, "reference": "P2__A1_s18", "chain": "P2__A1_s18", "stage": "p2a", "substituted": {}}


def test_the_parent_refuses_dirty_code_a_bad_head_and_a_busy_gpu(smoke, monkeypatch, tmp_path, capsys):
    ck = tmp_path / "model.pt"
    ck.write_bytes(b"head")
    good = smoke.sha256_file(ck)
    monkeypatch.setattr(smoke, "_load", lambda *a, **k: None)
    launched = []
    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *a, **k: launched.append(a) or (_ for _ in ()).throw(
        AssertionError("must not launch")))
    args = types.SimpleNamespace(print_plan=False, allow_dirty=False, no_idle_check=False, out_root=tmp_path / "s",
                                 python="py", device="cpu")
    monkeypatch.setattr(smoke, "build_plan", lambda a, rc=None: _fake_plan(smoke, dirty=True, sha=good, ckpt=ck))
    assert smoke.parent_main(args) == 7
    monkeypatch.setattr(smoke, "build_plan", lambda a, rc=None: _fake_plan(smoke, sha="0" * 64, ckpt=ck))
    assert smoke.parent_main(args) == 7
    monkeypatch.setattr(smoke, "build_plan", lambda a, rc=None: _fake_plan(smoke, sha=good, ckpt=ck))
    monkeypatch.setattr(smoke, "isaac_busy", lambda *a: ["123: isaac_venv/bin/python train.py"])
    assert smoke.parent_main(args) == 6
    assert not launched and not (tmp_path / "s").exists()


def test_the_parent_exit_code_comes_from_smokes_json(smoke, monkeypatch, tmp_path):
    ck = tmp_path / "model.pt"
    ck.write_bytes(b"head")
    plan = _fake_plan(smoke, sha=smoke.sha256_file(ck), ckpt=ck)
    monkeypatch.setattr(smoke, "_load", lambda *a, **k: None)
    monkeypatch.setattr(smoke, "build_plan", lambda a, rc=None: plan)
    monkeypatch.setattr(smoke, "isaac_busy", lambda *a: [])
    monkeypatch.setattr(smoke, "child_environ", lambda rc, p: {})

    def fake_popen(cmd, cwd, env, stdout, stderr, results=None):
        stamp = Path(cmd[cmd.index("--_child") + 1])
        res = {"S1": {"result": "PASS", "checks": []}, "S2": {"result": "PASS", "checks": []},
               "S3": smoke.judge_s3(_s3()), "S4": smoke.judge_s4(_means(), {"obst_episodes": 9}, {})}
        res.update(fake_popen.over)
        (stamp / "smokes.json").write_text(json.dumps({"plan": plan, "results": res, "finished": True, "errors": []}))
        return types.SimpleNamespace(wait=lambda: 0, send_signal=lambda s: None)
    monkeypatch.setattr(smoke.subprocess, "Popen", fake_popen)
    args = types.SimpleNamespace(print_plan=False, allow_dirty=False, no_idle_check=False, out_root=tmp_path / "s",
                                 python="py", device="cpu")
    fake_popen.over = {}
    assert smoke.parent_main(args) == 0
    fake_popen.over = {"S2": {"result": "FAIL", "checks": [["x", False, "y"]]}}
    monkeypatch.setattr(smoke, "utc_stamp", lambda: "second")
    assert smoke.parent_main(args) == 1
    txt = (tmp_path / "s" / "second" / "smokes.txt").read_text()
    assert "S2 FAIL :: x = y" in txt and txt.strip().endswith("smokes.json") and "SMOKES FAIL" in txt


# ===================================================================================== M3
def test_dump_facts_reads_a_null_dr_event_as_absent(live):
    env = {"rewards": {}, "events": {"randomize_rigid_body_mass": {"params": {}}, "randomize_rigid_body_com": None,
                                     "push_by_setting_velocity": {"params": {"velocity_range": {"x": [-0.5, 0.5]}}}}}
    f = live.dump_facts(env, None, None)
    assert f["dr_events"] == {"randomize_rigid_body_mass": True, "randomize_rigid_body_com": False,
                              "push_by_setting_velocity": True}


# ===================================================================================== M1: the code guard
LIVE = {"code_sha": "c" * 64, "eval_sha": "e" * 64, "dirty": False, "dirty_paths": []}


@pytest.mark.parametrize("live_over,state,ref,reanchor,want", [
    ({}, "none", None, False, 0),
    ({"dirty": True, "dirty_paths": ["x.py"]}, "none", None, False, 7),
    ({"dirty": None}, "none", None, False, 7),
    ({}, "ok", {"code_sha": "c" * 64, "eval_sha": "e" * 64, "dirty": False}, False, 0),
    ({}, "ok", {"code_sha": "d" * 64, "eval_sha": "e" * 64, "dirty": False}, False, 7),
    ({}, "ok", {"code_sha": "c" * 64, "eval_sha": "f" * 64, "dirty": False}, False, 7),
    ({}, "ok", {"code_sha": "c" * 64, "eval_sha": "e" * 64, "dirty": True}, False, 7),
    ({}, "missing", None, False, 7),
    ({}, "ok", {"code_sha": "d" * 64, "eval_sha": "e" * 64, "dirty": False}, True, 0),
    ({"dirty": True}, "ok", {"code_sha": "d" * 64}, True, 7),
])
def test_judge_code_guard(live, live_over, state, ref, reanchor, want):
    assert live.judge_code_guard(dict(LIVE, **live_over), state, ref, reanchor)["rc"] == want


def _stamp(root: Path, name: str, row: str, ts: str, sidecar: dict | None, seed: int = 1) -> Path:
    d = root / name / row / f"seed{seed:03d}" / ts
    d.mkdir(parents=True)
    (d / "complete.json").write_text("{}")
    if sidecar is not None:
        (d / "code_state.json").write_text(json.dumps(sidecar))
    return d


def test_anchor_reference_sidecar(live, tmp_path):
    assert live.anchor_reference_sidecar(tmp_path, "ANC_s18")[0] == "none"
    _stamp(tmp_path, "ANC_s18", "suite__omni", "2026-10-07_01-00-00", {"code_sha": "a"})
    _stamp(tmp_path, "ANC_s18", "field_hard__omni", "2026-10-07_09-00-00", {"code_sha": "z"})   # not the reference row
    st, sc, path = live.anchor_reference_sidecar(tmp_path, "ANC_s18")
    assert st == "ok" and sc == {"code_sha": "a"} and path.endswith("2026-10-07_01-00-00")
    _stamp(tmp_path, "ANC_s18", "suite__omni", "2026-10-07_02-00-00", None)
    assert live.anchor_reference_sidecar(tmp_path, "ANC_s18")[0] == "missing"       # the newest has no sidecar


def test_code_guard_end_to_end(live, tmp_path, monkeypatch):
    monkeypatch.setattr(live, "live_code_state", lambda root: dict(LIVE))
    a = types.SimpleNamespace(lrout=tmp_path, anchor_name="ANC_s18", code_root=tmp_path, reanchor=False)
    assert live.code_guard(a)[0] == 0
    _stamp(tmp_path, "ANC_s18", "suite__omni", "2026-10-07_01-00-00",
           {"code_sha": "c" * 64, "eval_sha": "e" * 64, "dirty": False})
    assert live.code_guard(a)[0] == 0
    monkeypatch.setattr(live, "live_code_state", lambda root: dict(LIVE, code_sha="9" * 64))
    rc_, res = live.code_guard(a)
    assert rc_ == 7 and res["fails"][0].startswith("code_sha")

    def boom(root):
        raise RuntimeError("git gone")
    monkeypatch.setattr(live, "live_code_state", boom)
    assert live.code_guard(a)[0] == 7                                                    # unreadable refuses


# ===================================================================================== M2: the CAL2 gate
#: the wall-mode A1 terrain (CHARTER 9 walls re-tune, candidates_v5)
WALL_OV = {"KRABBY_FIELD_EDGES": "wall", "KRABBY_FLAT_TERRAIN_DIFF": "0.20:1.0"}


def _cands(a1="0.40:0.20", c2="0.40:0.20", **a1_over):
    return {"chains": {"P2__A1_s18": {"stages": [{"overlay": dict(WALL_OV, KRABBY_TERRAIN_PROMOTE=a1, **a1_over)}]},
                       "P2__CALW2_s18": {"stages": [{"overlay": {"KRABBY_TERRAIN_PROMOTE": c2}}]}}}


def _cal_rec(chain="P2__CALW2_s18", result="PASS", promote="0.40:0.20", **over):
    r = {"ref": chain, "chain": chain, "smoke": True,
         "smoke_audit": json.dumps({"rc": 0, "result": result, "fails": [], "promote": promote, "levels_moved": True})}
    r.update(over)
    return r


def test_cal2_gate_passes_a_matched_passed_smoke(read):
    r = read.cal2_gate(_cal_rec(), _cands())
    assert r["ok"], r["fails"]


@pytest.mark.parametrize("rec,cands", [
    (None, _cands()),
    (_cal_rec(result="FAIL"), _cands()),
    (_cal_rec(killed="collapse"), _cands()),
    (_cal_rec(smoke=False), _cands()),
    (_cal_rec(chain="P2__CAL1_s18"), _cands()),
    (_cal_rec(chain="P2__CAL2_s18"), _cands()),                    # the ramp-era CAL2 no longer counts
    (_cal_rec(promote="0.45:0.25"), _cands()),
    (_cal_rec(), _cands(a1="TBD_PENDING_CALIBRATION_S5")),
    (_cal_rec(), _cands(a1="0.45:0.25", c2="0.40:0.20")),
    (_cal_rec(), _cands(c2="0.45:0.25")),
    (dict(_cal_rec(), smoke_audit="not json"), _cands()),
])
def test_cal2_gate_refuses(read, rec, cands):
    assert not read.cal2_gate(rec, cands)["ok"]


def test_cal2_gate_on_the_campaign_file_refuses_today(read):
    """candidates.yaml still carries the PROMOTE placeholder: the gate must refuse whatever the record says."""
    assert not read.cal2_gate(_cal_rec(), yaml.safe_load(open(CAMP / "candidates.yaml")))["ok"]


# ===================================================================================== B1: S5 and SMOKES_OK
def _s5_stamp(root: Path, falls: dict[int, int], n: int = 7, sha: str = "s" * 64, raw: bool = True,
              scenario: str = "field_light__omni", name: str = "SMOKE_S5_s18", edges: str = "wall") -> Path:
    """A field_light stamp: ``falls`` = {env_index: last segment_id} for the fall episodes."""
    d = _stamp(root, name, "field_light__omni", "2026-10-07_03-00-00",
               {"code_sha": "c" * 64, "eval_sha": "e" * 64, "dirty": False})
    labels = ["stand", "fwd", "lat_l", "diag", "back", "turn_l90", "turn_r90", "move_turn"]
    (d / "run_meta.json").write_text(json.dumps({
        "scenario_id": scenario, "checkpoint_sha256": sha, "hold_labels": labels,
        "env_vars_manifest": {"KRABBY_FIELD_EDGES": edges, "KRABBY_FLAT_TERRAIN_GEOM": "field"},
        "env_vars": {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_STD_PARAM": "soft",
                     "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25", "KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}}))
    eps = [{"env_index": i, "termination_reason": "fall" if i in falls else "schedule_complete"} for i in range(n)]
    (d / "scenario_metrics.json").write_text(json.dumps({"aggregate": {"n_episodes": n}, "episodes": eps}))
    (d / "raw").mkdir()
    if raw:
        for i in range(n):
            seg = np.array([-1, 0, 1, falls.get(i, 7)], dtype=np.int16)
            np.savez(d / "raw" / f"episode_{i:02d}.npz", segment_id=seg)
    return d


def test_s5_passes_with_falls_only_outside_the_stand_hold(read, tmp_path):
    d = _s5_stamp(tmp_path, {2: 3, 5: 6})
    r = read.smoke_s5_read(str(d), "s" * 64)
    assert r["ok"], r["fails"]
    assert r["stand_falls"] == 0 and r["other_falls"] == 2 and r["n_episodes"] == 7


@pytest.mark.parametrize("kw,want_in", [
    ({"falls": {1: 0}}, "stand-hold falls"),
    ({"falls": {1: -1}}, "stand-hold falls"),
    ({"falls": {}, "n": 8}, "8 episodes"),
    ({"falls": {}, "sha": "t" * 64}, "sha256"),
    ({"falls": {3: 2}, "raw": False}, "fail closed"),
    ({"falls": {}, "scenario": "field_hard__omni"}, "scenario"),
])
def test_s5_fails(read, tmp_path, kw, want_in):
    d = _s5_stamp(tmp_path, **kw)
    r = read.smoke_s5_read(str(d), "s" * 64)
    assert not r["ok"] and any(want_in in f for f in r["fails"]), r["fails"]


def test_resolve_stamp_by_name_or_path(read, tmp_path):
    d = _s5_stamp(tmp_path, {})
    assert read.resolve_stamp(str(d), "field_light__omni", 1, tmp_path) == str(d)
    assert read.resolve_stamp("SMOKE_S5_s18", "field_light__omni", 1, tmp_path).rstrip("/") == str(d)
    assert read.resolve_stamp("NOPE", "field_light__omni", 1, tmp_path) is None


def _smokes_dir(root: Path, smoke, **res_over) -> Path:
    d = root / "smokes" / "2026-10-07T00-00-00Z"
    d.mkdir(parents=True)
    res = {"S1": {"result": "PASS"}, "S2": {"result": "PASS"}, "S3": smoke.judge_s3(_s3()),
           "S4": dict(smoke.judge_s4(_means(), {"obst_episodes": 9}, {}), toe_radius_max_walking_m=1.5)}
    res.update(res_over)
    doc = {"plan": {"reference": "P2__A1_s18", "code_state": dict(LIVE), "substituted": {}},
           "results": res, "finished": True, "errors": [], "s1_summary": _s1_terrain_summary()}
    (d / "smokes.json").write_text(json.dumps(doc))
    return d


def _s1_terrain_summary(wall: bool = True, diff=(0.2, 1.0)) -> dict:
    """The S1 summary fields the SMOKES_OK terrain guard reads (CHARTER 9 walls re-tune)."""
    return {"difficulty_range": list(diff),
            "sub_terrains": {"parkour_flat": {"vertical_edges": None},
                             "omni_field_sparse": {"vertical_edges": wall}, "omni_field_dense": {"vertical_edges": wall}}}


def _cal1():
    return _cal_rec(chain="P2__CALW1_s18", promote="0.45:0.25", run_dir="/runs/cal1")


def test_smokes_ok_build_and_check(read, smoke, tmp_path):
    sd = _smokes_dir(tmp_path, smoke)
    s5 = read.smoke_s5_read(str(_s5_stamp(tmp_path, {})), "s" * 64)
    fails, payload = read.build_smokes_ok(sd, s5, _cal1(), dict(LIVE), _cands())
    assert not fails, fails
    assert payload["s3_toe_radius_max_m"] == pytest.approx(1.45) and payload["s4"]["block_steps_mean"] == 300.0
    assert payload["terrain"] == {"difficulty_range": [0.2, 1.0], "wall": True}
    assert read.smokes_ok_fails(payload, dict(LIVE), _cands()) == []
    assert read.smokes_ok_fails(payload, dict(LIVE, code_sha="9" * 64), _cands())  # the code moved since
    assert read.smokes_ok_fails(payload, dict(LIVE, dirty=True), _cands())
    assert read.smokes_ok_fails(None, dict(LIVE), _cands()) == ["no SMOKES_OK"]
    # CHARTER 9 walls re-tune: a SMOKES_OK counts only for the terrain A1 trains on
    assert read.smokes_ok_fails(payload, dict(LIVE), _cands(KRABBY_FIELD_EDGES="ramp"))
    assert read.smokes_ok_fails(payload, dict(LIVE), _cands(KRABBY_FLAT_TERRAIN_DIFF="0.20:0.70"))
    assert read.smokes_ok_fails(payload, dict(LIVE), None)                             # no candidates: fail closed
    assert read.smokes_ok_fails({k: v for k, v in payload.items() if k != "terrain"}, dict(LIVE), _cands())
    line = read.charter_line(payload)
    assert line.startswith("- ") and "max toe radius 1.450 m" in line and "block steps per obstacle-tile episode 300.0" \
        in line and line.endswith("\n")


@pytest.mark.parametrize("case", ["s2_fail", "toe", "s4_missing", "s5", "cal1_missing", "cal1_fail", "dirty_live",
                                  "code_moved", "dirty_smoke", "errors", "ramp_smoke", "diff_smoke", "s5_ramp",
                                  "ramp_cal1"])
def test_smokes_ok_refuses(read, smoke, tmp_path, case):
    over = {"s2_fail": {"S2": {"result": "FAIL"}},
            "toe": {"S3": smoke.judge_s3(_s3(toe_radius_max_m=1.7))},
            "s4_missing": {"S4": {"result": "PASS"}}}.get(case, {})
    sd = _smokes_dir(tmp_path, smoke, **over)
    if case == "dirty_smoke":
        doc = json.loads((sd / "smokes.json").read_text())
        doc["plan"]["code_state"]["dirty"] = True
        (sd / "smokes.json").write_text(json.dumps(doc))
    if case == "errors":
        doc = json.loads((sd / "smokes.json").read_text())
        doc["errors"] = ["Kit died"]
        (sd / "smokes.json").write_text(json.dumps(doc))
    if case in ("ramp_smoke", "diff_smoke"):
        doc = json.loads((sd / "smokes.json").read_text())
        doc["s1_summary"] = (_s1_terrain_summary(wall=False) if case == "ramp_smoke"
                             else _s1_terrain_summary(diff=(0.2, 0.7)))
        (sd / "smokes.json").write_text(json.dumps(doc))
    s5 = read.smoke_s5_read(str(_s5_stamp(tmp_path, {1: 0} if case == "s5" else {},
                                          edges="ramp" if case == "s5_ramp" else "wall")), "s" * 64)
    cal1 = {"cal1_missing": None, "cal1_fail": _cal_rec(chain="P2__CALW1_s18", result="FAIL"),
            "ramp_cal1": _cal_rec(chain="P2__CAL1b_s18", promote="0.45:0.25")}.get(case, _cal1())
    lv = {"dirty_live": dict(LIVE, dirty=True), "code_moved": dict(LIVE, eval_sha="0" * 64)}.get(case, dict(LIVE))
    fails, payload = read.build_smokes_ok(sd, s5, cal1, lv, _cands())
    assert fails and payload["result"] == "FAIL"


def test_write_smokes_ok_cli(read, smoke, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    (out / "arms").mkdir(parents=True)
    (out / "arms" / "P2__CALW1_s18.record.json").write_text(json.dumps(_cal1()))
    sd = _smokes_dir(tmp_path, smoke)
    s5 = _s5_stamp(out / "lr", {}, sha=yaml.safe_load(open(CAMP / "anchors.yaml"))["anchors"][18]["sha256"],
                   name="SMOKE_S5W_s18")
    monkeypatch.setattr(read, "_live_code_state", lambda root=None: dict(LIVE))
    charter = (CAMP / "CHARTER.md").read_text()
    argv = ["p2_read.py", "--write-smokes-ok", "--smokes", str(sd), "--s5", "SMOKE_S5W_s18", "--out", str(out),
            "--no-charter-log"]
    monkeypatch.setattr(sys, "argv", argv)
    assert read.main() == 0
    assert json.loads((out / "SMOKES_OK").read_text())["result"] == "PASS"
    assert (CAMP / "CHARTER.md").read_text() == charter                               # --no-charter-log
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--smokes-ok-check", "--out", str(out)])
    assert read.main() == 0
    monkeypatch.setattr(read, "_live_code_state", lambda root=None: dict(LIVE, code_sha="1" * 64))
    assert read.main() == 4
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--smokes-ok-check", "--out", str(tmp_path / "empty")])
    assert read.main() == 4
    _ = capsys.readouterr()


# ===================================================================================== shell: run_arm.sh
def _stub_py(tmp: Path, guard_rc: int = 7) -> Path:
    p = tmp / "stub_py"
    p.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        echo "$@" >> {tmp}/stub_py.calls
        case " $* " in
          *" --code-guard "*) echo '{{"rc": {guard_rc}, "result": "STUB"}}'; exit {guard_rc} ;;
          *" --preflight "*) echo '{{"result": "PASS"}}'; exit 0 ;;
        esac
        exit 0
        """))
    p.chmod(0o755)
    return p


def _env(tmp: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2_"))}
    camp = tmp / "camp"
    (camp / "logs").mkdir(parents=True, exist_ok=True)
    (tmp / "code").mkdir(exist_ok=True)
    env.update(P2_OUT=str(tmp / "out"), CAMPAIGN_DIR=str(camp), P2_CAP="", CODE_ROOT=str(tmp / "code"))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _bash(script: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120)


RUN_ARM = SCRIPTS / "run_arm.sh"
QUEUE = SCRIPTS / "p2_queue.sh"


def test_scripts_parse():
    for f in (RUN_ARM, QUEUE):
        assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0, f


def test_row_eval_cli_refuses_on_the_code_guard_before_any_eval(tmp_path):
    env = _env(tmp_path, P2_PY=_stub_py(tmp_path, 7))
    ck, st, man = tmp_path / "m.pt", tmp_path / "stamp", tmp_path / "man.yaml"
    ck.write_text("x")
    st.mkdir()
    (st / "run_meta.json").write_text(json.dumps({"env_vars": {}}))
    man.write_text("x")
    base = f"bash {RUN_ARM} --row-eval NAME {ck} {st} {man} field_light__omni 1"
    r = _bash(base + " --episodes 7", env)
    assert r.returncode == 7, r.stderr
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "STOP row-eval NAME field_light__omni env_seed 1 refused: code-state guard" in log and "NOTIFY" in log
    calls = (tmp_path / "stub_py.calls").read_text()
    assert "--code-guard" in calls and "eval_crab_hex_gait" not in calls
    assert _bash(base + " --episodes x", env).returncode == 2
    assert _bash(base + " --episodes", env).returncode == 2
    assert _bash(base + " --episodes 0", env).returncode == 2
    r = _bash(f"bash {RUN_ARM} --lr-eval NAME {ck} {st}", env)
    assert r.returncode == 7


def test_row_eval_passes_episodes_through_to_the_harness(tmp_path):
    """Lib mode: the guard and the sidecar stubbed, the harness a stub that records its argv and writes a stamp."""
    code = tmp_path / "code"
    ev = code / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
    ev.parent.mkdir(parents=True)
    ev.write_text(textwrap.dedent("""\
        import json, sys, pathlib
        a = sys.argv
        out = pathlib.Path(a[a.index('--output-root') + 1]) / a[a.index('--scenario') + 1] / 'seed001' / 'ts'
        out.mkdir(parents=True)
        (out / 'complete.json').write_text('{}')
        (out / 'argv.json').write_text(json.dumps(a))
        """))
    (code / "parkour").mkdir(exist_ok=True)
    env = _env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1)
    st = tmp_path / "stamp"
    st.mkdir()
    (st / "run_meta.json").write_text(json.dumps({"env_vars": {"KRABBY_STD_PARAM": "soft"}}))
    script = f"""
        source {RUN_ARM}
        code_guard() {{ echo guard "$1" >> {tmp_path}/guard.calls; return 0; }}
        drivers_idle() {{ :; }}
        write_sidecar() {{ echo '{{}}' > "$1/code_state.json"; }}
        row_eval NAME {tmp_path}/m.pt {st} {tmp_path}/man.yaml field_light__omni 1 7; echo "rc=$? dir=$ROW_DIR"
    """
    r = _bash(script, env)
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    argv = json.loads((tmp_path / "out/lr/NAME/field_light__omni/seed001/ts/argv.json").read_text())
    assert argv[argv.index("--episodes") + 1] == "7" and argv[argv.index("--env-seed") + 1] == "1"
    assert (tmp_path / "guard.calls").read_text().startswith("guard row-eval NAME")


def test_run_ref_refuses_on_the_code_guard_after_preflight(tmp_path):
    env = _env(tmp_path, P2_PY=_stub_py(tmp_path, 7), P2_RUN_ARM_LIB=1)
    script = f"""
        source {RUN_ARM}
        chain_info() {{ echo "P2__A1_s18 18 0"; }}
        chain_placeholders() {{ return 0; }}
        verify_head_sha() {{ echo "sha256 ok"; return 0; }}
        launch() {{ touch {tmp_path}/LAUNCHED; return 0; }}
        run_ref P2__A1_s18; echo "rc=$?"
    """
    r = _bash(script, env)
    assert "rc=7" in r.stdout, r.stdout + r.stderr
    assert not (tmp_path / "LAUNCHED").exists()
    calls = (tmp_path / "stub_py.calls").read_text().splitlines()
    assert any("--preflight" in c for c in calls) and "--code-guard" in calls[-1]      # the guard runs after preflight
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "STOP P2__A1_s18 refused: code-state guard" in log
    # with a passing guard it goes on to launch (the stub launch refuses: rc 6, nothing else runs)
    env["P2_PY"] = str(_stub_py(tmp_path, 0))
    script_ok = f"""
        source {RUN_ARM}
        chain_info() {{ echo "P2__A1_s18 18 0"; }}
        chain_placeholders() {{ return 0; }}
        verify_head_sha() {{ echo "sha256 ok"; return 0; }}
        launch() {{ touch {tmp_path}/LAUNCHED; LAUNCH_WHY=stub; return 6; }}
        run_ref P2__A1_s18; echo "rc=$?"
    """
    r = _bash(script_ok, env)
    assert "rc=6" in r.stdout and (tmp_path / "LAUNCHED").exists(), r.stdout + r.stderr
    assert "P2__A1_s18 code-state guard ok" in (tmp_path / "out" / "queue.log").read_text()


def test_the_watcher_follows_a_driver_retry_to_the_r2_training_log(tmp_path):
    camp = tmp_path / "camp"
    runs = tmp_path / "runs"
    runs.mkdir()
    env = _env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1)
    (camp / "logs").mkdir(parents=True, exist_ok=True)
    (camp / "orchestrator.log").write_text(
        "[2026-10-07 10:00:00] seed18_X01_p2a: Isaac-Crab-Hex-Omni-Teacher-v0 mode 2a | 5000 its\n"
        "[2026-10-07 10:20:00] seed18_X01_p2a: infra death — one retry\n"
        "[2026-10-07 10:22:00] seed18_X01_p2a_r2: Isaac-Crab-Hex-Omni-Teacher-v0 mode 2a | 5000 its\n")
    old = runs / "2026-10-07_10-00-05_seed18_X01_p2a"
    new = runs / "2026-10-07_10-22-04_seed18_X01_p2a"
    old.mkdir()
    new.mkdir()
    os.utime(old, (1_000_000, 1_000_000))
    (camp / "logs" / "seed18_X01_p2a_train.log").write_text("first")
    (camp / "logs" / "seed18_X01_p2a_r2_train.log").write_text("retry")
    os.utime(camp / "logs" / "seed18_X01_p2a_train.log", (1_000_000, 1_000_000))
    script = f"""
        source {RUN_ARM}
        RUNS={runs}; T0S="2026-10-07 09:59:00"
        if retry_target {old}; then echo "RD=$RT_RD"; echo "TL=$RT_TL"; else echo none; fi
        if retry_target {new}; then echo again; else echo "same dir: none"; fi
        echo "POST=$(train_log_for seed18_X01_p2a)"
        echo "PLAIN=$(train_log_for seed99_X09_p2a)"
    """
    r = _bash(script, env)
    out = r.stdout
    assert f"RD={new}" in out, out + r.stderr
    assert f"TL={camp}/logs/seed18_X01_p2a_r2_train.log" in out
    assert "same dir: none" in out
    assert f"POST={camp}/logs/seed18_X01_p2a_r2_train.log" in out
    assert f"PLAIN={camp}/logs/seed99_X09_p2a_train.log" in out


# ===================================================================================== shell: p2_queue.sh
def _queue_env(tmp: Path, smokes_rc: int, cal2_rc: int) -> dict:
    read_stub = tmp / "read_stub.py"
    read_stub.write_text(textwrap.dedent(f"""\
        import sys
        open({str(tmp / 'read.calls')!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')
        a = sys.argv[1:]
        print('{{"stub": true}}')
        if '--smokes-ok-check' in a: sys.exit({smokes_rc})
        if '--cal2-gate' in a: sys.exit({cal2_rc})
        sys.exit(0)
        """))
    run_arm = tmp / "run_arm_stub.sh"
    run_arm.write_text(f"#!/usr/bin/env bash\necho \"$@\" >> {tmp}/run_arm.calls\nexit 5\n")
    run_arm.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2_"))}
    env.update(P2_OUT=str(tmp / "out"), P2_PY=sys.executable, READ=str(read_stub), RUN_ARM=str(run_arm))
    return env


def test_queue_stops_without_smokes_ok(tmp_path):
    env = _queue_env(tmp_path, smokes_rc=4, cal2_rc=0)
    for step in ("anchors", "a1s18", "recipe"):
        env["FROM_STEP"] = step
        r = subprocess.run(["bash", str(QUEUE)], env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode == 4, r.stderr
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "STOP p2_queue: no valid" in log and "SMOKES_OK" in log and "NOTIFY" in log
    assert not (tmp_path / "run_arm.calls").exists()
    assert f"--smokes-ok-check --out {tmp_path / 'out'}" in (tmp_path / "read.calls").read_text()


def test_queue_stops_before_a1_without_a_passed_cal2(tmp_path):
    env = _queue_env(tmp_path, smokes_rc=0, cal2_rc=4)
    env["FROM_STEP"] = "a1s18"
    r = subprocess.run(["bash", str(QUEUE)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 4
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "CAL2 GATE rc 4" in log and "STOP p2_queue: the CAL2 smoke (P2__CALW2_s18) is not a passed calibration smoke" in log
    assert not (tmp_path / "run_arm.calls").exists()


def test_queue_launches_a1_once_both_gates_pass(tmp_path):
    env = _queue_env(tmp_path, smokes_rc=0, cal2_rc=0)
    env["FROM_STEP"] = "a1s18"
    r = subprocess.run(["bash", str(QUEUE)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 4                                       # the stub arm exits 5 -> block_ok stops
    assert (tmp_path / "run_arm.calls").read_text().split() == ["P2__A1_s18"]
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "STOP p2_queue: rc 5 in the arm P2__A1_s18 step" in log


def test_queue_anchors_need_the_smokes_gate_first(tmp_path):
    env = _queue_env(tmp_path, smokes_rc=0, cal2_rc=0)
    env["FROM_STEP"] = "anchors"
    r = subprocess.run(["bash", str(QUEUE)], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 4
    first = (tmp_path / "run_arm.calls").read_text().splitlines()[0]
    assert first.startswith("--row-eval ANCW_s18") and "suite__omni 1" in first     # the wall-era anchor names
    calls = (tmp_path / "read.calls").read_text().splitlines()
    assert calls[0].startswith("--smokes-ok-check")


# ===================================================================================== M4
def test_the_stale_preflight_sentence_is_gone():
    for f in (CAMP / "REPLAY_GATE.md", SCRIPTS / "p2_replay.py"):
        txt = f.read_text()
        assert "It does not read `verdict`" not in txt, f
    assert "records a user `overrule`" in (CAMP / "REPLAY_GATE.md").read_text()


def test_the_queue_header_has_no_e1_step():
    head = QUEUE.read_text().split("\nC=", 1)[0]
    assert "-> E1 ->" not in head and "#   s1|e1|f1 " not in head          # no E1 step (e1 stays a FROM_STEP value)
    assert "E1 was dropped" in head


def test_schedule_row_stamps_count_as_complete_with_the_runner_marker(tmp_path):
    """The harness writes complete.json only for category_suite rows; run_arm.sh marks a clean 8-hold schedule-row
    eval with p2_complete.json, and the reader must accept either marker (S5 smoke, 2026-10-06)."""
    import importlib.util
    p = Path(__file__).resolve().parents[2] / ("parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/"
                                                "2026-10-07_omni_phase2/scripts/p2_read.py")
    spec = importlib.util.spec_from_file_location("p2_read_marker", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    root = tmp_path / "lr" / "X" / "field_light__omni" / "seed001"
    a, b, c = root / "2026-10-06_00-00-01", root / "2026-10-06_00-00-02", root / "2026-10-06_00-00-03"
    for d in (a, b, c):
        d.mkdir(parents=True)
    (a / "complete.json").write_text("{}")
    (b / "p2_complete.json").write_text("{}")
    assert mod.stamp_complete(a) and mod.stamp_complete(b) and not mod.stamp_complete(c)
    assert mod.newest_stamp("X", "field_light__omni", 1, lrout=tmp_path / "lr") == str(b)
    sh = (p.parent / "run_arm.sh").read_text()
    assert "p2_complete.json" in sh and 'erc" = 0' in sh
