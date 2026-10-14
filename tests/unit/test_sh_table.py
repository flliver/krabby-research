"""Stand-height campaign (``2026-10-13_stand_height``): the TABLE / GROUND stand-up test (K3) on CPU -- no Isaac / Kit /
GPU. ``scripts/sh_table_core.py`` (pure pieces), ``scripts/sh_table_test.py`` (Kit script: its Kit-free modes, pre-Kit
refusals and env composition with stubs) and ``scripts/sh_table_test.sh`` (launcher, with fake tools).

Pinned here:

* both scripts are torch- / numpy-free at import, write no bytecode, and import nothing of Isaac before Kit;
* the K3 cells: the frozen tops (gated 0.635 / 0.6985 / 0.762 + the reported 18-in table; CHARTER step-4 decision),
  5 surfaces x 3 poses, the GATED set = exactly the 6 gated-top x knee_folded / random cells (every other cell
  reported, never flipping a K3 line, the R_h floor or its STOP), the fixed poses' spawn / table geometry against
  ``facts.json``,
  the random-fold family (hip grid, knee range, table rejection), commands (>= top + 0.31; ground >= max(h_min, spawn +
  0.14625); stratified; fixed ``--h-cmd`` in range), per-(seed, cell, k) determinism independent of the other cells,
  k = 0 nominal jitter, the pinned K3 episode (16.0 s) covering every env's rise + ramp + settle + window;
* the reset event with stub assets: floor reference = the highest probe point, root z, joints, the table's top by z
  placement (ground: parked), the realised spawn record; the raycast probe refuses a missing mesh and a miss;
* the controllers through sh_core: R_h on the effective stand rows, S1 rows = clamp(S1, +-1) bitwise, posture-only =
  q, stand_net = sh_core's input + router, amask_old = the predecessor's input with clamp(a_net, +-1) (bitwise
  stand_net at a 1.064 command with W[:, 15] zeroed), the env obs never written; R_h at a retracted start = SEAT ->
  q(h_init) on every leg; the stand-checkpoint rule (tags, mask, S1, the W[:, 15] zeroing);
* K3 STOOD on synthetic traces (every clause and its boundary, the 95 % tolerances, 5- / 6-down, falls by cause,
  the H5 hip-contact arming on ground starts, fell-off, homed), the aggregate, the pooled views, the K3 line vs R_h
  (rate - max(5 pp, 2 SE), falls <= max(1, R_h's) per 100) on the gated cells and the step-4 R_h floor per gated top
  pooled over the two gated poses, the merge (duplicates refused; gated vs reported labelled);
* the action-clip check, the crab_failure read before it is disabled, the env composition (time-out only, DR off,
  ``sh_start`` LAST, the tall kinematic table);
* the launcher: ``bash -n``, the busy pattern, ``--dry-run`` (smoke / full / split; nothing written), refusals, and the
  queue with fake tools (facts first, serial jobs, the merge last, stop at the first failure, a failed merge is rc 6);
* the residual deadband (CHARTER 2026-10-14): ``--deadband`` on stand_net only (controller, pre-Kit, launcher jobs),
  its default label suffix, the Kit path records it in the summary and every cell, the merge never pools two deadbands
  under one label (absent = 0).
"""
from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import json
import math
import os
import py_compile
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
TASK = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = TASK / "experiments/2026-10-13_stand_height"
SCRIPTS = CAMP / "scripts"
CORE_PATH = SCRIPTS / "sh_table_core.py"
TEST_PATH = SCRIPTS / "sh_table_test.py"
SH_PATH = SCRIPTS / "sh_table_test.sh"
PRED = TASK / "experiments/2026-10-11_stand_network/scripts"
RSL_MODULES = REPO / "parkour/scripts/rsl_rl/modules"
S1_DIR = REPO / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-07_07-50-10_seed18_X08_p2a_A15pB"
S1_CKPT = S1_DIR / "model_19994.pt"
AMASK = TASK / "policy/stand_net/model_4000.pt"
W, P0, P1, NP = 1149, 207, 216, 75
HIPS = [29 + i for i in range(6)]
CAMS = [18 + 2 * i for i in range(6)]
DT = 0.02


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def C():
    return _load(CORE_PATH, "_t_sht_core")


@pytest.fixture(scope="module")
def T():
    return _load(TEST_PATH, "_t_sht_test")


@pytest.fixture(scope="module")
def SH(C):
    return C.SH


@pytest.fixture(scope="module")
def facts():
    return json.loads((CAMP / "facts.json").read_text())


@pytest.fixture(scope="module")
def plan1(C):
    return C.plan_envs(C.K3_CELLS, 100, 1, jitter=True)


def bits(a):
    return a.contiguous().view(torch.int32) if a.dtype == torch.float32 else a.contiguous().view(torch.int64)


def bitwise_equal(a, b) -> bool:
    return a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(bits(a), bits(b))


# ===================================================================================== import hygiene
@pytest.mark.parametrize("path", [CORE_PATH, TEST_PATH])
def test_import_is_torch_and_numpy_free_and_writes_no_bytecode(path):
    code = ("import sys, importlib.util\n"
            f"spec = importlib.util.spec_from_file_location('m', {str(path)!r})\n"
            "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
            "bad = sorted(k for k in ('torch', 'numpy', 'isaaclab', 'gymnasium', 'parkour_tasks') if k in sys.modules)\n"
            "assert not bad, bad\n"
            "print('ok')\n")
    before = {p for p in SCRIPTS.iterdir()}
    pred_before = {p for p in PRED.iterdir()}
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"}, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["ok"]
    assert not (SCRIPTS / "__pycache__").exists()
    assert {p for p in SCRIPTS.iterdir()} == before
    assert {p for p in PRED.iterdir()} == pred_before


def test_scripts_compile_and_no_top_level_heavy_imports(tmp_path):
    for path in (TEST_PATH, CORE_PATH):
        py_compile.compile(str(path), cfile=str(tmp_path / (path.stem + ".pyc")), doraise=True)
        tree = ast.parse(path.read_text())
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name.split(".")[0] for n in top for a in n.names} | {
            (n.module or "").split(".")[0] for n in top if isinstance(n, ast.ImportFrom)}
        assert not names & {"torch", "numpy", "isaaclab", "gymnasium", "parkour_tasks", "imageio", "yaml"}, (path, names)
    src = TEST_PATH.read_text()
    for needle in ("env.clip_actions = float(SH.HARNESS_CLIP)", "CORE.controller_actions(", "hc.step(stand, h_cmd_t, obs)",
                   "SH.terrain_z_under_feet(", "SH.h_meas(", "dcore.load_strict(", "SN.state_dicts_equal(",
                   "CORE.stand_ckpt_rule(", "SH.zero_height_column(", "save_rng()", "restore_rng(before)",
                   "rng_equal(before", "il_time_out", "kinematic_enabled=True", "imageio.get_writer(",
                   "P2L.arm_slot_latch(", "_install_schedule_override_world", "_apply_env_overrides",
                   "CORE.action_clip_reasons(", "crab_failure_params(", "hip_sid", 'B["h_lim"][t] = hs.h_lim',
                   '"stand_rows_rule"', 'np.zeros_like(A["stand_eff"]) if o.controller == "s1"'):
        assert needle in src, needle
    assert "RecordVideo" not in src
    # order: the probe is registered BEFORE the wrapper's construction reset; the height step BEFORE the actions
    assert src.index('CORE.RUNTIME["terrain_probe"] =') < src.index("ParkourRslRlVecEnvWrapper(raw_env")
    loop = src[src.index("for t in range(T):"):]
    assert loop.index("hc.step(") < loop.index("CORE.controller_actions(") < loop.index("env.step(actions)")
    assert src.index("crab_failure_params(env_cfg.terminations)") < src.index("setattr(term, name, None)")


# ===================================================================================== constants / cells
def test_constants_and_cells(C, SH, facts):
    fr = facts["frozen"]
    assert C.K3_TOPS_M == tuple(fr["k3_eval_tops_m"]) == (0.635, 0.6985, 0.762)            # gated (step-4 decision)
    assert C.K3_REPORTED_TOPS_M == tuple(fr["k3_reported_tops_m"]) == (0.4572,)              # run + read, never gated
    assert C.K3_GATED_POSES == tuple(fr["k3_gated_poses"]) == ("knee_folded", "random")
    assert C.TOP_MIN_M == fr["top_min_m"] == 0.635 and C.TOP_MAX_TRAIN_M == fr["top_max_train_m"] == 0.805
    assert C.SURFACES == ("table_0p4572", "table_0p635", "table_0p6985", "table_0p762", "ground")
    assert len(C.K3_CELLS) == 15 and C.K3_CELLS[0] == "table_0p4572__knee_folded" and C.K3_CELLS[-1] == "ground__random"
    assert C.POSES == ("knee_folded", "rod_min", "random") and C.ENVS_PER_CELL == 100
    assert C.CONTROLLERS == ("r_h", "stand_net", "posture_only", "s1", "amask_old")
    assert C.CMD_MARGIN_EVAL_M == 0.31 and C.GROUND_CMD_RISE_M == pytest.approx(0.31 - 0.15875 - 0.005)
    assert (C.STOOD_WINDOW_S, C.STOOD_RISE_M, C.STOOD_MIN_FEET, C.STOOD_SHARE, C.STOOD_H_TOL_M) == (3.0, 0.10, 4, 0.95,
                                                                                                      0.04)
    assert (C.FOOT_LOADED_N, C.FOOT_FLOOR_Z_M) == (20.0, 0.15)                 # the predecessor table test's rule
    assert (C.HOMED_RAD, C.HOMED_LAST_S, C.SETTLE_HOLD_S) == (0.10, 5.0, 0.5)  # p2s_stand Z2
    assert C.START_CODE == {"table": SH.START_KINDS.index("table"), "ground": SH.START_KINDS.index("ground")}
    for top in C.K3_TOPS_M:                                                     # frozen eval tops are feasible
        assert fr["top_min_m"] <= top <= fr["top_max_train_m"] and top + 0.31 <= SH.H_MAX
    for top in C.K3_REPORTED_TOPS_M:                                            # the 18-in table: below the trained
        assert top < fr["top_min_m"] and top + 0.31 <= SH.H_MAX                 # range, still runnable
    assert C.surface_id(0.635) == "table_0p635" and C.surface_id(None) == "ground"
    assert C.surface_id(0.6985) == "table_0p6985" and C.SURFACE_TOP["table_0p6985"] == 0.6985
    assert C.SURFACE_TOP["table_0p4572"] == 0.4572 and C.SURFACE_TOP["ground"] is None


def test_gated_set_is_exactly_the_six_cells(C):
    """CHARTER step-4 decision: gated = tops {0.635, 0.6985, 0.762} x poses {knee_folded, random}; every other cell (the
    18-in table, every rod_min cell, every ground cell) is reported."""
    assert C.K3_GATED_CELLS == ("table_0p635__knee_folded", "table_0p635__random", "table_0p6985__knee_folded",
                                "table_0p6985__random", "table_0p762__knee_folded", "table_0p762__random")
    assert len(C.K3_REPORTED_CELLS) == 9 and not set(C.K3_GATED_CELLS) & set(C.K3_REPORTED_CELLS)
    assert set(C.K3_GATED_CELLS) | set(C.K3_REPORTED_CELLS) == set(C.K3_CELLS)
    assert set(C.K3_REPORTED_CELLS) == ({c for c in C.K3_CELLS if c.startswith(("ground", "table_0p4572"))}
                                        | {c for c in C.K3_CELLS if c.endswith("__rod_min")})
    assert [c for c in C.K3_CELLS if C.cell_gated(c)] == list(C.K3_GATED_CELLS)
    assert not C.cell_gated("ground__knee_folded") and not C.cell_gated(None) and not C.cell_gated("bogus")
    assert C.START_TYPES == ("tables",) and C.REPORTED_START_TYPES == ("ground",)
    g = C.gating_record()
    assert g["gated_cells"] == list(C.K3_GATED_CELLS) and g["reported_cells"] == list(C.K3_REPORTED_CELLS)
    assert g["gated_tops_m"] == [0.635, 0.6985, 0.762] and g["reported_tops_m"] == [0.4572]
    assert C.cell_record({"seed": 1}, "table_0p6985__random", [], [])["meta"]["gated"] is True
    assert C.cell_record({"seed": 1}, "table_0p6985__rod_min", [], [])["meta"]["gated"] is False
    f = C.build_facts()
    assert f["k3_gated_cells"] == list(C.K3_GATED_CELLS) and f["k3_reported_cells"] == list(C.K3_REPORTED_CELLS)
    assert f["surfaces"]["table_0p635"]["gated_poses"] == ["knee_folded", "random"]
    assert f["surfaces"]["table_0p4572"]["gated_poses"] == [] and f["surfaces"]["ground"]["gated_poses"] == []


def test_parse_cells(C):
    assert C.parse_cells("all") == list(C.K3_CELLS)
    assert C.parse_cells("ground") == ["ground__knee_folded", "ground__rod_min", "ground__random"]
    assert C.parse_cells("table_0p6985") == ["table_0p6985__knee_folded", "table_0p6985__rod_min",
                                             "table_0p6985__random"]
    assert C.parse_cells("table_0p635__rod_min,ground,table_0p635__rod_min") == [
        "table_0p635__rod_min", "ground__knee_folded", "ground__rod_min", "ground__random"]
    for bad in ("table_0p7__rod_min", "ground__folded", "ground_rod_min", "x"):
        with pytest.raises(ValueError):
            C.parse_cells(bad)


def test_fixed_pose_geometry_matches_the_step1_facts(C, facts):
    fp = facts["c_table_ground"]["poses"]
    for pose in C.FIXED_POSES:
        p = C.fixed_pose(pose)
        lp = C.lowest_parts(p["hip_rad"], p["knee_left_rad"])
        assert C.ground_root_rel(p["hip_rad"], p["knee_left_rad"]) == pytest.approx(
            fp[pose]["ground"]["spawn_root_z_m"], abs=1e-9)
        assert C.table_top_min(p["hip_rad"], p["knee_left_rad"]) == pytest.approx(
            fp[pose]["table_top_min_m_legs_0p02_above_floor"], abs=1e-9)
        assert lp["first_contact"] == fp[pose]["ground"]["first_contact"]
        for top in C.K3_TOPS_M + C.K3_REPORTED_TOPS_M:                          # gated and reported tables
            assert C.fold_ok_on_table(p["hip_rad"], p["knee_left_rad"], top), (pose, top)
    assert C.table_spawn_root_z(0.635) == pytest.approx(0.79875)


def test_random_fold_family(C, facts):
    hips = C.random_hip_grid()
    assert len(hips) == 30 and math.degrees(hips[0]) == pytest.approx(-44.5) and math.degrees(hips[-1]) == pytest.approx(
        -30.0)
    for h in hips[::7]:
        k_lo, k_hi = C.random_knee_range(h)
        assert k_lo < k_hi
        assert C.STC.link_clearances(h, k_hi)["min_m"] >= C.STC.PLATE_CLEARANCE_M
    fam = facts["c_table_ground"]["random_fold_family"]
    lo, hi = fam["ground_spawn_root_z_m"][0], fam["ground_spawn_root_z_m"][2]
    for k in range(40):
        d = C.draw_random_fold(7, "ground__random", k, None)
        assert d == C.draw_random_fold(7, "ground__random", k, None)                 # deterministic
        assert d["hip_rad"] in hips and lo - 1e-6 <= C.ground_root_rel(d["hip_rad"], d["knee_left_rad"]) <= hi + 0.01
        t = C.draw_random_fold(7, "table_0p4572__random", k, 0.4572)
        assert C.fold_ok_on_table(t["hip_rad"], t["knee_left_rad"], 0.4572)
    # the table rule rejects: a fold whose legs would touch the floor under an 18-in table
    grid = [(h, float(k)) for h in hips[::3] for k in np.linspace(*C.random_knee_range(h), 9)]
    bad = [(h, k) for h, k in grid if C.table_top_min(h, k) > 0.4572]
    assert bad and not any(C.fold_ok_on_table(h, k, 0.4572) for h, k in bad)
    assert max(C.table_top_min(h, k) for h, k in grid) == pytest.approx(fam["table_top_min_for_legs_clear_m"][2],
                                                                        abs=0.02)


def test_commands_and_ranges(C, SH, plan1):
    for e in plan1:
        lo, hi = e["h_cmd_range"]
        assert lo - 1e-12 <= e["h_cmd"] <= hi + 1e-12 and hi == SH.H_MAX
        if e["top"] is not None:
            assert lo == pytest.approx(e["top"] + 0.31)
        else:
            assert lo == pytest.approx(max(SH.H_MIN, e["spawn_h_plan"] + C.GROUND_CMD_RISE_M))
    for cell in ("table_0p635__knee_folded", "ground__rod_min"):                      # stratified quantiles
        rows = [e for e in plan1 if e["cell"] == cell]
        lo, hi = rows[0]["h_cmd_range"]
        u = sorted((e["h_cmd"] - lo) / (hi - lo) for e in rows)
        assert u == pytest.approx([(k + 0.5) / 100 for k in range(100)], abs=1e-9)
        assert [e["h_cmd"] for e in rows] != sorted(e["h_cmd"] for e in rows)       # not ordered by env index
    assert C.cmd_quantiles(1, "ground__rod_min", 1) == [0.5]
    one = C.plan_envs(["table_0p762__rod_min"], 1, 1)[0]
    assert one["h_cmd"] == pytest.approx(0.5 * (0.762 + 0.31 + SH.H_MAX))
    fixed = C.plan_envs(["table_0p635__knee_folded", "ground__rod_min"], 3, 1, h_cmd=1.2)
    assert all(e["h_cmd"] == 1.2 for e in fixed)
    with pytest.raises(ValueError):
        C.plan_envs(["table_0p762__rod_min"], 2, 1, h_cmd=1.0)                        # < 0.762 + 0.31
    with pytest.raises(ValueError):
        C.plan_envs(["ground__knee_folded"], 2, 1, h_cmd=1.4)                         # > h_max


def test_plan_layout_determinism_and_independence(C, plan1):
    assert [e["env"] for e in plan1] == list(range(1500))
    assert [e["cell"] for e in plan1[::100]] == list(C.K3_CELLS)
    again = C.plan_envs(C.K3_CELLS, 100, 1, jitter=True)
    assert again == plan1 and C.plan_digest(again) == C.plan_digest(plan1)
    assert C.plan_digest(C.plan_envs(C.K3_CELLS, 100, 2, jitter=True)) != C.plan_digest(plan1)
    sub = C.plan_envs(["ground__random", "table_0p4572__random"], 100, 1, jitter=True)
    strip = lambda e: {k: v for k, v in e.items() if k != "env"}  # noqa: E731
    for e in sub:                                                     # a cell's envs do not depend on the other cells
        ref = next(x for x in plan1 if x["cell"] == e["cell"] and x["k"] == e["k"])
        assert strip(e) == strip(ref)
    for e in plan1:
        if e["top"] is None or e["k"] == 0:
            assert e["table_offset"] == [0.0, 0.0, 0.0]                  # nominal: env k = 0, and every ground env
        else:
            assert abs(e["table_offset"][0]) <= 0.03 and abs(e["table_offset"][1]) <= 0.03
            assert abs(e["table_offset"][2]) <= math.radians(10.0) and e["table_offset"] != [0.0, 0.0, 0.0]
        if e["top"] is not None:
            assert e["spawn_root_rel"] == pytest.approx(e["top"] + 0.15875 + 0.005)
        assert e["h_init_plan"] == pytest.approx(min(max(e["spawn_h_plan"], 0.719), 1.346))
    assert all(e["table_offset"] == [0.0, 0.0, 0.0] for e in C.plan_envs(["table_0p635__rod_min"], 5, 1, jitter=False))


def test_episode_sizing(C, SH, plan1):
    K = C.kin()
    lh, lk = C.posture_rod_targets(SH.H_SETTLED)                         # q(1.064) = 0: the mid-stroke rods
    assert lh == pytest.approx(0.5 * (K.HIP_LEN_MIN_M + K.HIP_LEN_MAX_M), abs=1e-9)
    assert lk == pytest.approx(0.5 * (K.KNEE_LEN_MIN_M + K.KNEE_LEN_MAX_M), abs=1e-9)
    assert C.posture_rod_targets(0.3) == C.posture_rod_targets(SH.H_MIN)  # clamped
    n = C.env_need_s(-0.7767, 1.355, 0.719, 1.346)
    assert n["ramp_s"] == pytest.approx((1.346 - 0.719) / 0.08)
    assert n["need_s"] == pytest.approx(max(n["rise_to_init_s"] + n["ramp_s"], n["rise_to_cmd_s"]) + 1.0 + 3.0 + 0.5)
    assert C.k3_episode_s() == 16.0                                      # PINNED: the K3 suite episode
    assert C.round_episode(15.51) == 16.0 and C.round_episode(15.5) == 15.5
    for seed_plan in (plan1, C.plan_envs(C.K3_CELLS, 100, 2, jitter=True)):
        assert max(e["need_s"] for e in seed_plan) <= C.k3_episode_s()
        assert C.plan_episode_s(seed_plan) <= C.k3_episode_s()


def test_joint_rows_probe_rows_and_event_params(C, plan1):
    names = [f"{leg}_{j}" for j in ("Body_Hip_RevoluteJoint", "Body_CamShaft_RevoluteJoint", "Hip_Femur_RevoluteJoint",
                                    "Femur_Tibia_RevoluteJoint") for leg in C.LEGS]
    rows = C.joint_pos_rows(plan1[:3], names)
    for e, row in zip(plan1[:3], rows):
        for name, v in zip(names, row):
            leg = name.split("_")[0]
            if "Hip_Femur" in name:
                assert v == e["hip_rad"]
            elif "Femur_Tibia" in name:
                assert v == (e["knee_left_rad"] if leg in C.LEFT else -e["knee_left_rad"])
            else:
                assert v == 0.0
    with pytest.raises(ValueError):
        C.joint_pos_rows(plan1[:1], ["FL_Mystery_Joint"])
    pts = C.probe_points_body(plan1[0]["hip_rad"], plan1[0]["knee_left_rad"])
    assert len(pts) == 23 and [0.0, 0.0] in pts
    kf = C.fixed_pose("knee_folded")
    toes = [p["toe"] for p in C.STC.foot_points_body(kf["hip_rad"], kf["knee_left_rad"]).values()]
    assert all([t[0], t[1]] in pts for t in toes)
    ev = C.event_params(plan1, names)
    assert len(ev["joint_pos"]) == len(ev["probe_xy_b"]) == len(ev["root_rel"]) == 1500
    assert all(math.isnan(t) for t, e in zip(ev["table_top_rel"], plan1) if e["top"] is None)
    assert all(t == e["top"] for t, e in zip(ev["table_top_rel"], plan1) if e["top"] is not None)
    json.dumps({k: v for k, v in ev.items() if k != "table_top_rel"})                  # plain lists


def test_action_clip_reasons(C):
    good_clip = {".*_Body_CamShaft_RevoluteJoint": (-1.0, 1.0), ".*_Hip_Femur_RevoluteJoint": (-4.8, 4.8),
                 ".*_Femur_Tibia_RevoluteJoint": (-4.8, 4.8)}
    good_scale = {".*_Body_CamShaft_RevoluteJoint": 3.0, ".*_Hip_Femur_RevoluteJoint": 0.25,
                  ".*_Femur_Tibia_RevoluteJoint": 0.25}
    assert C.action_clip_reasons(good_clip, good_scale) == []
    bad = dict(good_clip, **{".*_Hip_Femur_RevoluteJoint": (-1.0, 1.0)})
    assert any("Hip_Femur" in r for r in C.action_clip_reasons(bad, good_scale))
    assert C.action_clip_reasons((-4.8, 4.8), good_scale)
    assert any("scale" in r for r in C.action_clip_reasons(good_clip, dict(good_scale, **{
        ".*_Femur_Tibia_RevoluteJoint": 0.24})))


# ===================================================================================== the reset event (stubs)
class _Asset:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def rec(*a, **k):
            self.calls.append((name, [x.clone() if torch.is_tensor(x) else x for x in a],
                               {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in k.items()}))
        return rec


class _Scene(dict):
    def __init__(self, n):
        super().__init__(robot=_Asset(), sh_table=_Asset())
        self.env_origins = torch.tensor([[10.0 * i, -5.0 * i, 0.03] for i in range(n)])
        self.sensors = {}


class _Env:
    def __init__(self, n):
        self.num_envs, self.device, self.scene = n, "cpu", _Scene(n)


def _floor(xy):
    return 0.01 * xy[:, 0] - 0.004 * xy[:, 1] + 0.02 * torch.sin(3.0 * xy[:, 0])


def test_reset_event_floor_reference_tables_and_ground(C, plan1, capsys):
    pick = [next(e for e in plan1 if e["cell"] == c and e["k"] == 3) for c in
            ("table_0p635__knee_folded", "ground__rod_min", "table_0p4572__random", "ground__random")]
    names = [f"{leg}_{j}" for leg in C.LEGS for j in ("Body_Hip_RevoluteJoint", "Body_CamShaft_RevoluteJoint",
                                                      "Hip_Femur_RevoluteJoint", "Femur_Tibia_RevoluteJoint")]
    ev = C.event_params(pick, names)
    env = _Env(4)
    C.RUNTIME.clear()
    C.RUNTIME["terrain_probe"] = _floor
    C.RUNTIME["probe_label"] = "stub"
    try:
        C.sh_start(env, torch.tensor([0, 1, 2, 3]), **ev)
        rec = {c[0]: c for c in env.scene["robot"].calls}
        pose = rec["write_root_pose_to_sim"][1][0]
        fl = []
        for i, e in enumerate(pick):
            xy = torch.tensor(ev["probe_xy_b"][i], dtype=torch.float32) + env.scene.env_origins[i, :2]
            fl.append(float(_floor(xy).max()))
        fl = torch.tensor(fl)
        assert torch.allclose(C.RUNTIME["last_floor_z"], fl, atol=1e-6)
        assert torch.allclose(pose[:, 2], fl + torch.tensor(ev["root_rel"], dtype=torch.float32), atol=1e-6)
        assert torch.allclose(pose[:, :2], env.scene.env_origins[:, :2]) and torch.equal(
            pose[:, 3:], torch.tensor([[1.0, 0, 0, 0]] * 4))
        assert torch.allclose(C.RUNTIME["last_root_z"], pose[:, 2])
        q, v = rec["write_joint_state_to_sim"][1]
        assert torch.allclose(q, torch.tensor(ev["joint_pos"])) and torch.equal(v, torch.zeros_like(v))
        assert torch.allclose(rec["set_joint_position_target"][1][0], q)
        tp = {c[0]: c for c in env.scene["sh_table"].calls}["write_root_pose_to_sim"][1][0]
        for i, e in enumerate(pick):
            if e["top"] is None:
                assert float(tp[i, 2]) == C.PARK_CENTER_Z_M
            else:                                                        # top = floor + top, by z placement
                assert float(tp[i, 2]) + 0.5 * C.TABLE_HEIGHT_M == pytest.approx(float(fl[i]) + e["top"], abs=1e-6)
                assert float(pose[i, 2]) - (float(tp[i, 2]) + 0.5 * C.TABLE_HEIGHT_M) == pytest.approx(0.16375, abs=1e-6)
            off = e["table_offset"]
            assert float(tp[i, 0]) == pytest.approx(float(env.scene.env_origins[i, 0]) + off[0], abs=1e-6)
        assert C.RUNTIME["probe_kind"] == "stub"
        # an empty reset is a no-op; a partial reset keeps the other rows' records
        n_calls = len(env.scene["robot"].calls)
        C.sh_start(env, torch.tensor([], dtype=torch.long), **ev)
        assert len(env.scene["robot"].calls) == n_calls
        keep = C.RUNTIME["last_root_z"].clone()
        C.RUNTIME["terrain_probe"] = lambda xy: _floor(xy) + 1.0
        C.sh_start(env, torch.tensor([2]), **ev)
        assert torch.allclose(C.RUNTIME["last_root_z"][[0, 1, 3]], keep[[0, 1, 3]])
        assert float(C.RUNTIME["last_root_z"][2]) == pytest.approx(float(keep[2]) + 1.0, abs=1e-5)
        bad = dict(ev, table_top_rel=[float("nan")] * 4)                    # a table start without a top
        with pytest.raises(ValueError):
            C.sh_start(env, torch.tensor([0]), **bad)
    finally:
        C.RUNTIME.clear()
    with pytest.raises(Exception):                                       # no probe, no Isaac: fails closed
        C.sh_start(_Env(4), torch.tensor([1]), **ev)
    capsys.readouterr()


def test_terrain_probe_refuses_missing_mesh_and_misses(C):
    with pytest.raises(RuntimeError):
        C.make_terrain_probe(None, lambda *a, **k: None)
    seen = {}

    def fake_raycast(starts, dirs, mesh):
        seen["starts"], seen["dirs"], seen["mesh"] = starts.clone(), dirs.clone(), mesh
        hits = starts.clone()
        hits[:, 2] = 0.01 * starts[:, 0]
        return (hits,)

    p = C.make_terrain_probe("MESH", fake_raycast)
    z = p(torch.tensor([[1.0, 2.0], [3.0, -1.0]]))
    assert torch.allclose(z, torch.tensor([0.01, 0.03])) and seen["mesh"] == "MESH"
    assert torch.all(seen["starts"][:, 2] == C.PROBE_START_Z_M) and torch.equal(
        seen["dirs"], torch.tensor([[0.0, 0.0, -1.0]] * 2))
    miss = C.make_terrain_probe("MESH", lambda s, d, mesh: (torch.full_like(s, float("inf")),))
    with pytest.raises(RuntimeError):
        miss(torch.zeros(3, 2))


# ===================================================================================== controllers (stubs + real nets)
class _Est:
    def __init__(self, k: int, scale: float):
        self.k, self.scale = k, scale

    def inference(self, prop):
        return (prop.sum(dim=1, keepdim=True) * self.scale + torch.arange(self.k, dtype=prop.dtype)).to(prop.dtype)


class _Net:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def act_inference(self, x, hist_encoding=False):
        self.calls.append((x.clone(), hist_encoding))
        return self.fn(x)


def _obs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    o = torch.randn(n, W, generator=g) * 0.3
    o[:, 3:5] *= 0.1
    o[:, 15] = 0.0
    return o


def _hs(SH, n, h, slot, restoring=None):
    hc = SH.HeightController(n, "cpu", dt=DT)
    hc.reset(None, "walk", h_cmd=h)
    return hc.step(slot, h, _obs(n, 99))._replace(**({} if restoring is None else {"restoring": restoring}))


def test_controller_routes_through_sh_core(C, SH):
    n = 6
    obs = _obs(n, 1)
    obs0 = obs.clone()
    s1_est, st_est = _Est(P1 - P0, 0.01), _Est(P1 - P0, -0.02)
    s1 = _Net(lambda x: 1.7 * torch.tanh(3.0 * x[:, :18]))                # saturating: rows beyond +-1
    st = _Net(lambda x: torch.tanh(x[:, 65:83] + x[:, P0:P0 + 1]) * 1.5 + x[:, 15:16])
    slot = torch.tensor([True, True, False, True, False, True])
    h = torch.tensor([0.8, 1.2, 1.064, 1.064, 0.9, 1.3], dtype=torch.float64)
    hc = SH.HeightController(n, "cpu", dt=DT)
    hc.reset(None, "walk", h_cmd=h)
    hs = hc.step(slot, h, obs)
    harness = obs.clone()
    harness[:, P0:P1] = s1_est.inference(harness[:, :NP])
    a_s1 = s1.fn(harness)
    kw = dict(s1=s1, s1_est=s1_est, p0=P0, p1=P1, num_prop=NP, st_pol=st, st_est=st_est, mask=1)
    a, ex = C.controller_actions("s1", obs, slot, hs, **kw)
    assert bitwise_equal(a, a_s1.clamp(-1, 1)) and a_s1.abs().max() > 1.0
    a, ex = C.controller_actions("posture_only", obs, slot, hs, **kw)
    assert bitwise_equal(a[slot], SH.q18(hs.h_lim)[slot]) and bitwise_equal(a[~slot], a_s1.clamp(-1, 1)[~slot])
    assert float(a[0, 6:12].max()) < -1.0                                 # h = 0.8: hips beyond -1
    a, ex = C.controller_actions("r_h", obs, slot, hs, **kw)
    lab = SH.rh_label(harness, hs.h_lim, check=False)
    assert bitwise_equal(a, SH.route(hs.stand_eff, lab.target, a_s1)) and torch.equal(ex["r_branch"], lab.branch)
    assert float(ex["r_label_excess"]) <= SH.LABEL_TOL
    a, ex = C.controller_actions("stand_net", obs, slot, hs, **kw)
    x = SH.stand_net_input(harness, st_est, p0=P0, p1=P1, num_prop=NP, mask=1, h_cmd_limited=hs.h_lim,
                           restoring=hs.restoring)
    assert torch.equal(st.calls[-1][0], x) and st.calls[-1][1] is True
    assert bitwise_equal(a, SH.router_actions(hs.stand_eff, st.fn(x), a_s1, hs.h_lim).actions)
    a_old, ex_old = C.controller_actions("amask_old", obs, slot, hs, **kw)
    x_old = SH.SN.stand_net_input(harness, st_est, p0=P0, p1=P1, num_prop=NP, mask=1)
    assert torch.equal(st.calls[-1][0], x_old)
    assert bitwise_equal(a_old[slot], st.fn(x_old).clamp(-1, 1)[slot])
    assert bitwise_equal(a_old[~slot], a_s1.clamp(-1, 1)[~slot])
    # at a 1.064 command (col 15 = 0, q = 0) stand_net and amask_old are the same function of the obs
    one = torch.full((n,), 1.064, dtype=torch.float64)
    hs1 = _hs(SH, n, one, slot)
    a1, _ = C.controller_actions("stand_net", obs, slot, hs1, **kw)
    a2, _ = C.controller_actions("amask_old", obs, slot, hs1, **kw)
    assert bitwise_equal(a1, a2)
    assert torch.equal(obs, obs0)                                         # the env's obs is never written
    with pytest.raises(ValueError):
        C.controller_actions("stand_net", obs, slot, hs, s1=s1, s1_est=s1_est, p0=P0, p1=P1, num_prop=NP)
    with pytest.raises(ValueError):
        C.controller_actions("zero", obs, slot, hs, **kw)


def test_rh_at_a_retracted_start_is_seat_to_the_posture(C, SH):
    """PLAN H3 premise on CPU: from every fixed retracted pose, at the limiter's table / ground init heights, every leg
    is SEAT and R_h's target is the posture q(h_init) on every leg (cams 0)."""
    rst = SH.SN.load_rst()
    seat = list(rst.BRANCH_NAMES).index("seat")
    dflt = C.STC.defaults_rad()["hip"]
    for pose in C.FIXED_POSES:
        p = C.fixed_pose(pose)
        inits = [SH.H_MIN] + [min(max(C.table_spawn_root_z(t), SH.H_MIN), SH.H_MAX) for t in C.K3_TOPS_M]
        obs = torch.zeros(len(inits), W)
        obs[:, HIPS] = float(p["hip_rad"] - dflt)
        lab = SH.rh_label(obs, torch.tensor(inits, dtype=torch.float64))
        assert torch.all(lab.branch == seat), (pose, lab.branch)
        assert torch.equal(lab.target, SH.q18(torch.tensor(inits, dtype=torch.float64))), pose


def test_stand_ckpt_rule(C, SH):
    s1 = "a" * 64
    amask = SH.STAND_NET_SHA256
    sn = {"sn_dagger": {"stand_mask": 1, "s1_sha256": s1, "arm": "mask", "seed": 17}}
    sh = {"sh_dagger": {"stand_mask": 1, "s1_sha256": s1}}
    r = C.stand_ckpt_rule("stand_net", amask, sn, mask=1, s1_sha256=s1)
    assert r["tag"] == "sn_dagger" and r["zero_w15"] is True and r["arm"] == "mask"
    assert C.stand_ckpt_rule("amask_old", amask, sn, mask=1, s1_sha256=s1)["zero_w15"] is True
    r = C.stand_ckpt_rule("stand_net", "b" * 64, sh, mask=1, s1_sha256=s1)
    assert r["tag"] == "sh_dagger" and r["zero_w15"] is False
    both = {**sn, **sh}                                   # the campaign trainer writes both blocks: sh_dagger wins
    r = C.stand_ckpt_rule("stand_net", "b" * 64, both, mask=1, s1_sha256=s1)
    assert r["tag"] == "sh_dagger" and r["zero_w15"] is False
    for ctrl, sha, infos, mask in (("amask_old", "b" * 64, sn, 1),                # amask_old: A-mask only
                                   ("amask_old", amask, sh, 1),                    # ... the predecessor's
                                   ("stand_net", "b" * 64, {}, 1),                 # untagged
                                   ("stand_net", "b" * 64, sh, 0),                 # mask mismatch
                                   ("stand_net", "b" * 64, {"sh_dagger": {"stand_mask": True, "s1_sha256": s1}}, 1),
                                   ("stand_net", "b" * 64, {"sh_dagger": {"stand_mask": 1, "s1_sha256": "c" * 64}}, 1)):
        with pytest.raises(ValueError):
            C.stand_ckpt_rule(ctrl, sha, infos, mask=mask, s1_sha256=s1)
    with pytest.raises(ValueError):
        C.stand_ckpt_rule("r_h", amask, sn, mask=1, s1_sha256=s1)


def rsl_modules():
    name = "_sht_test_rslmods"
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(RSL_MODULES)]
            sys.modules[name] = pkg
        crab = importlib.import_module(f"{name}.crab_actor_critic_with_encoder")
        est = importlib.import_module(f"{name}.feature_extractors.estimator")
    finally:
        sys.dont_write_bytecode = old
    return crab.CrabHexActorCriticRMA, est.DefaultEstimator


@pytest.fixture(scope="module")
def nets(SH):
    if not (S1_CKPT.is_file() and AMASK.is_file()):
        pytest.skip("S1 model_19994.pt / the stand net is not on this machine")
    crab_cls, est_cls = rsl_modules()
    agent = yaml.safe_load((S1_DIR / "params/agent.yaml").read_text())

    def policy():
        pc = copy.deepcopy(agent["policy"])
        pc.pop("class_name")
        return crab_cls(W, 18, **pc)

    ec = copy.deepcopy(agent["estimator"])
    ec.pop("class_name")
    s1ck = torch.load(S1_CKPT, map_location="cpu", weights_only=False)
    amck = torch.load(AMASK, map_location="cpu", weights_only=False)
    s1, est, am = policy(), est_cls(**ec), policy()
    s1.load_state_dict(s1ck["model_state_dict"], strict=True)
    est.load_state_dict(s1ck["estimator_state_dict"], strict=True)
    sd, _norm = SH.zero_height_column(amck["model_state_dict"])
    am.load_state_dict(sd, strict=True)
    return {"s1": s1.eval(), "est": est.eval(), "am": am.eval(), "infos": amck.get("infos"),
            "s1_sha": SH.SN.sha256_file(S1_CKPT)}


def test_real_networks_stand_net_vs_amask_old_and_the_posture(C, SH, nets):
    n = 64
    obs = _obs(n, 5)
    slot = torch.ones(n, dtype=torch.bool)
    kw = dict(s1=nets["s1"], s1_est=nets["est"], p0=P0, p1=P1, num_prop=NP, st_pol=nets["am"], st_est=nets["est"],
              mask=1)
    rule = C.stand_ckpt_rule("stand_net", SH.STAND_NET_SHA256, nets["infos"], mask=1, s1_sha256=nets["s1_sha"])
    assert rule["zero_w15"] and rule["tag"] == "sn_dagger"
    with torch.no_grad():
        hs1 = _hs(SH, n, torch.full((n,), 1.064, dtype=torch.float64), slot)
        a1, _ = C.controller_actions("stand_net", obs, slot, hs1, **kw)
        a2, _ = C.controller_actions("amask_old", obs, slot, hs1, **kw)
        assert bitwise_equal(a1, a2)                                     # A-mask + q at 1.064 == the old path
        low = torch.full((n,), 0.8, dtype=torch.float64)
        hc = SH.HeightController(n, "cpu", dt=DT)
        hc.reset(None, "table", h_meas=low)
        hs = hc.step(slot, low, obs)
        a3, ex = C.controller_actions("stand_net", obs, slot, hs, **kw)
    assert bitwise_equal(a3, SH.router_actions(hs.stand_eff, ex["a_net"], a2, hs.h_lim).actions)
    assert float(a3[:, 6:12].max()) < -0.99 and float(a3[:, 6:12].min()) >= -4.8   # hip targets beyond -1


# ===================================================================================== K3 classification
#: the root sits this far above the chassis' lowest point on a level chassis (sh_table_core.BODY_HALF_Z_M)
ROOT_ABOVE_LOW = 0.15875


def _series(T=800, rise=0.25, feet=6, herr=0.0, tilt=0.0, hip=0.0, cam=0.0, h_cmd=1.0, spawn=0.64, term=False,
            h_lim=None, corner_drop=0.0):
    """``rise`` drives both the lowest corner and the root (level chassis); ``corner_drop`` lowers ONLY the corner
    (a tilted chassis: its lowest corner sits below the level belly while the root does not move)."""
    def arr(v, dtype=float):
        return np.full(T, v, dtype=dtype) if np.isscalar(v) else np.asarray(v, dtype=dtype)

    s = {"lowest_z": spawn + arr(rise) - arr(corner_drop), "cos_tilt": np.cos(arr(tilt)),
         "feet_on_floor": arr(feet, int), "h_meas": h_cmd + arr(herr), "hip_force": arr(hip),
         "cam_abs": np.repeat(arr(cam)[:, None], 6, axis=1), "root_z": spawn + ROOT_ABOVE_LOW + arr(rise),
         "terminated": arr(term, bool)}
    if h_lim is not None:
        s["h_lim"] = arr(h_lim)
    return s


def _cls(C, start="table", top=0.635, **kw):
    spawn = kw.pop("spawn", 0.64)
    h_cmd = kw.get("h_cmd", 1.0)
    return C.classify(_series(spawn=spawn, **kw), DT, start=start, spawn_lowest_z=spawn,
                      spawn_root_z=spawn + ROOT_ABOVE_LOW, h_cmd=h_cmd, table_top=top if start == "table" else None)


def test_classify_clean_standup(C):
    T = 800
    t = (np.arange(T) + 1) * DT
    rise = np.clip((t - 1.0) * 0.05, 0.0, 0.25)
    feet = np.where(t < 2.0, 0, 6)
    herr = np.where(t < 8.0, -0.3 + 0.0375 * t, 0.0)
    h_lim = np.where(np.arange(T) < 400, 0.9, 1.0)
    r = _cls(C, rise=rise, feet=feet, herr=herr, h_lim=h_lim)
    assert r["stood"] and r["stood_5down"] and r["stood_6down"] and not r["reasons"], r["reasons"]
    assert r["clauses"] == {"rise": True, "feet4": True, "h_tol": True, "no_fall": True, "not_terminated": True}
    assert r["limiter_arrived_s"] == pytest.approx(401 * DT) and r["limiter_arrived_before_window"]
    assert r["within_2cm_first_s"] is not None and r["within_2cm_first_s"] < 8.0
    assert r["lift_off_s"] == pytest.approx((int(np.nonzero(rise > 0.02)[0][0]) + 1) * DT)
    assert r["homed"] and r["settled"] and r["cams_homed_end"] and not r["fall"]
    assert r["window_shares"] == {"rise": 1.0, "feet4": 1.0, "feet5": 1.0, "feet6": 1.0, "h_tol": 1.0}


def test_classify_boundaries_and_tolerances(C):
    assert _cls(C, rise=0.10)["stood"] and not _cls(C, rise=0.10 - 1e-6)["stood"]
    r = _cls(C, feet=4)
    assert r["stood"] and not r["stood_5down"] and not r["stood_6down"]
    r = _cls(C, feet=5)
    assert r["stood"] and r["stood_5down"] and not r["stood_6down"]
    assert not _cls(C, feet=3)["stood"]
    assert _cls(C, herr=0.04)["stood"] and _cls(C, herr=-0.04)["stood"] and not _cls(C, herr=0.0401)["stood"]
    T, w = 800, 150
    assert C.window_steps(DT) == w and C.steps_needed(w) == 143
    for n_bad, ok in ((7, True), (8, False)):
        feet = np.full(T, 6)
        feet[T - w + 2: T - w + 2 + 2 * n_bad: 2] = 3
        herr = np.zeros(T)
        herr[T - w + 3: T - w + 3 + 2 * n_bad: 2] = 0.08
        assert _cls(C, feet=feet)["stood"] is ok and _cls(C, herr=herr)["stood"] is ok, n_bad
    for n_bad, ok in ((7, True), (8, False)):                              # the rise clause: >= 95 % too
        rise = np.full(T, 0.25)
        rise[T - w + 4: T - w + 4 + 2 * n_bad: 2] = 0.09
        r = _cls(C, rise=rise)
        assert r["stood"] is ok, n_bad
        assert r["rise_only_fail"] is (not ok) and r["window_shares"]["rise"] == pytest.approx((w - n_bad) / w)
    assert any("root rise" in x for x in r["reasons"])
    early = np.where(np.arange(T) < T - w, 0.0, 0.25)                     # only the window counts for (1)-(3)
    assert _cls(C, rise=early, feet=np.where(np.arange(T) < T - w, 0, 6))["stood"]
    r = C.classify(_series(T=w - 1), DT, start="table", spawn_lowest_z=0.64, spawn_root_z=0.64 + ROOT_ABOVE_LOW,
                   h_cmd=1.0, table_top=0.635)
    assert not r["stood"] and any("window" in x for x in r["reasons"])


def test_classify_falls_and_h5_arming(C):
    T = 800
    spike = np.where(np.arange(T) == 10, 600.0, 0.0)
    r = _cls(C, hip=spike)                                                # table: armed from step 0
    assert r["fall_hip_contact"] and r["fall"] and not r["stood"] and r["hip_contact_armed_s"] == pytest.approx(DT)
    rise = np.where(np.arange(T) < 300, 0.0, 0.25)
    r = _cls(C, start="ground", rise=rise, hip=spike)                     # ground, before the 0.10 m rise: reported
    assert r["stood"] and not r["fall"] and r["hip_contact_pre_arm_steps"] == 1
    assert r["hip_force_pre_arm_max_n"] == 600.0 and r["hip_contact_armed_s"] == pytest.approx(301 * DT)
    r = _cls(C, start="ground", rise=rise, hip=np.where(np.arange(T) == 350, 600.0, 0.0))
    assert r["fall_hip_contact"] and not r["stood"]
    sag = np.where(np.arange(T) < 300, 0.0, np.where(np.arange(T) < 320, 0.12, 0.05))
    armed = C.hip_contact_armed(sag, "ground")
    assert not armed[:300].any() and armed[300:].all()                    # stays armed after the rise drops
    assert C.hip_contact_armed(sag, "table").all()
    assert _cls(C, hip=np.full(T, 500.0))["stood"]                        # > threshold, not >=
    r = _cls(C, tilt=np.where(np.arange(T) == 50, 0.51, 0.0))
    assert r["fall_tilt"] and not r["stood"] and _cls(C, tilt=0.5)["stood"]
    low = np.full(T, 0.25)
    low[400:] = 0.635 - 0.06 - 0.64                                       # 6 cm below the table-top plane
    r = _cls(C, rise=low)
    assert r["fell_off_table"] and r["fall"] and not r["stood"]
    r = _cls(C, start="ground", rise=low)                                 # no table on a ground start
    assert not r["fell_off_table"]
    r = _cls(C, term=np.arange(T) == 100)
    assert r["terminated"] and not r["stood"] and not r["fall"]
    for kw in ({"start": "table", "table_top": None}, {"start": "ground", "table_top": 0.6}, {"start": "air",
                                                                                              "table_top": None}):
        with pytest.raises(ValueError):
            C.classify(_series(), DT, spawn_lowest_z=0.64, spawn_root_z=0.64 + ROOT_ABOVE_LOW, h_cmd=1.0, **kw)
    bad = _series()
    bad["h_meas"] = bad["h_meas"][:-1]
    with pytest.raises(ValueError):
        C.classify(bad, DT, start="table", spawn_lowest_z=0.64, spawn_root_z=0.64 + ROOT_ABOVE_LOW, h_cmd=1.0,
                   table_top=0.635)


def test_classify_rise_reads_the_root_not_the_tilted_corner(C):
    """Blocking-fix pin (review 2026-10-13): the K3 rise clause reads the ROOT's rise on >= 95 % of the window. A robot
    standing inside the 4-cm height tolerance at the lowest command (top + 0.31: nominal rise 0.31 - 0.16375 = 0.146
    m), 3.5 cm low, with a 1-cm max-vs-mean floor offset and 2 cm of tilt drop at its lowest corner, has a root rise of
    0.1013 m and a corner rise of 0.0813 m: it STOOD (the old every-step corner rule said no)."""
    top, T = 0.635, 800
    h_cmd = top + C.CMD_MARGIN_EVAL_M
    nominal = C.CMD_MARGIN_EVAL_M - C.BODY_HALF_Z_M - C.SPAWN_CLEARANCE_M
    assert nominal == pytest.approx(0.14625)
    rise = nominal - 0.035 - 0.010
    r = _cls(C, top=top, h_cmd=h_cmd, rise=rise, herr=-0.035, corner_drop=0.02)
    assert r["stood"] and r["clauses"]["rise"] and not r["rise_only_fail"], r["reasons"]
    assert r["rise_min_window_m"] == pytest.approx(rise) and r["rise_corner_min_window_m"] == pytest.approx(rise - 0.02)
    assert r["rise_corner_min_window_m"] < C.STOOD_RISE_M < r["rise_min_window_m"]
    # a root that never rose 0.10 fails on the rise clause ONLY (feet / height / falls fine): reported per cell
    r = _cls(C, top=top, h_cmd=h_cmd, rise=0.09)
    assert not r["stood"] and r["rise_only_fail"] and r["clauses"] == {
        "rise": False, "feet4": True, "h_tol": True, "no_fall": True, "not_terminated": True}
    a = C.aggregate([r, _cls(C), _cls(C, feet=3, rise=0.09)])
    assert a["n_fail_rise_only"] == 1 and a["n_stood"] == 1
    assert C.pooled_views({"table_0p635__knee_folded": a})["tables"]["n_fail_rise_only"] == 1
    # the hip-contact arming keeps the lowest-corner rise (the trainer's re-arm rule): a ground start whose corner
    # never rises 0.10 is not armed although its root did
    r = _cls(C, start="ground", rise=0.12, corner_drop=0.05, hip=np.where(np.arange(T) == 500, 600.0, 0.0))
    assert r["hip_contact_armed_s"] is None and not r["fall_hip_contact"] and r["hip_contact_pre_arm_steps"] == 1
    with pytest.raises(TypeError):
        C.classify(_series(), DT, start="table", spawn_lowest_z=0.64, h_cmd=1.0, table_top=0.635)


def test_classify_homed_and_standing_since(C):
    T = 800
    cam = np.zeros(T)
    cam[T - 200] = 0.2                                                    # inside the last 5 s: not homed
    r = _cls(C, cam=cam)
    assert not r["homed"] and r["cams_homed_end"] and r["cam_max_last5_rad"] == pytest.approx(0.2)
    cam = np.zeros(T)
    cam[T - 300] = 0.2                                                    # before the last 5 s
    assert _cls(C, cam=cam)["homed"]
    cam[-1] = 0.15
    assert not _cls(C, cam=cam)["cams_homed_end"]
    assert not _cls(C, cam=0.2)["settled"]
    rise = np.where(np.arange(T) < 300, 0.0, 0.25)
    feet = np.where(np.arange(T) < 310, 0, 6)
    r = _cls(C, rise=rise, feet=feet)
    assert r["stood"] and r["standing_since_s"] == pytest.approx(311 * DT)


def test_aggregate(C):
    rs = [_cls(C), _cls(C, feet=4), _cls(C, rise=0.0), _cls(C, tilt=0.7), _cls(C, cam=0.2)]
    a = C.aggregate(rs)
    assert a["n_envs"] == 5 and a["n_stood"] == 3 and a["stood_share"] == pytest.approx(0.6)
    assert a["n_stood_5down"] == 2 and a["falls"] == 1 and a["falls_tilt"] == 1 and a["falls_per100"] == 20.0
    assert a["homed"] == 4 and a["homed_share"] == pytest.approx(0.8)
    assert "FALL" in a["fail_reasons"] and a["h_err_window_median_m"]["n"] == 5


# ===================================================================================== pooled views / K3 / floor
def _agg(n=100, stood=95, falls=0):
    return {"n_envs": n, "n_stood": stood, "falls": falls, "n_stood_5down": stood, "n_stood_6down": stood,
            "homed": stood}


def test_k3_line_formula(C):
    k = C.k3_line(_agg(stood=85), _agg(stood=95))
    se = math.sqrt(0.85 * 0.15 / 100 + 0.95 * 0.05 / 100)
    assert k["se_diff"] == pytest.approx(se) and k["rate_line"] == pytest.approx(0.95 - 2 * se) and not k["rate_pass"]
    k = C.k3_line(_agg(stood=90), _agg(stood=95))
    assert k["rate_pass"] and k["pass"]
    k = C.k3_line(_agg(stood=100), _agg(stood=100))                       # 5 pp floor when the SE is 0
    assert k["rate_line"] == pytest.approx(0.95) and k["pass"]
    assert C.k3_line(_agg(stood=95, falls=1), _agg(stood=95))["falls_pass"]
    k = C.k3_line(_agg(stood=95, falls=2), _agg(stood=95))
    assert not k["falls_pass"] and k["falls_line_per100"] == 1.0 and not k["pass"]
    assert C.k3_line(_agg(stood=95, falls=3), _agg(stood=95, falls=3))["falls_pass"]   # max(1, ref's)
    assert C.k3_line(_agg(n=0, stood=0), _agg())["pass"] is None


def test_pooled_views_k3_read_and_rh_floor(C):
    ref = {c: _agg(stood=95) for c in C.K3_CELLS}
    subj = {c: _agg(stood=93) for c in C.K3_CELLS}
    bad = dict(subj, **{c: _agg(stood=60) for c in C.K3_CELLS if c.startswith("ground")})
    pv = C.pooled_views(ref)
    assert pv["tables"]["n_envs"] == 600 and pv["tables"]["cells"] == 6 and pv["ground"]["n_envs"] == 300
    assert set(pv["per_top"]) == {"table_0p635", "table_0p6985", "table_0p762"}
    assert pv["per_top"]["table_0p635"]["n_envs"] == 200 and pv["per_top"]["table_0p635"]["cells"] == 2
    assert set(pv["per_top_all_poses"]) == set(C.SURFACES[:-1]) and pv["per_top_all_poses"]["table_0p635"][
        "n_envs"] == 300
    assert set(pv["ground_per_pose"]) == set(C.POSES)
    assert pv["gated_cells"] == list(C.K3_GATED_CELLS) and pv["missing_gated_cells"] == []
    assert pv["reported_cells"] == list(C.K3_REPORTED_CELLS)
    assert all(pv["cells"][c]["gated"] == C.cell_gated(c) for c in C.K3_CELLS)
    pooled = {"1": {"r_h": pv, "x": C.pooled_views(subj), "y": C.pooled_views(bad),
                    "z": C.pooled_views({c: a for c, a in subj.items() if c != "table_0p6985__random"}),
                    "zr": C.pooled_views({c: a for c, a in subj.items() if "rod_min" not in c})}}
    k3 = C.k3_read(pooled)
    assert "r_h" not in k3["1"]
    assert k3["1"]["x"]["tables"]["pass"] and k3["1"]["x"]["tables"]["gated"] is True
    assert k3["1"]["x"]["ground"]["pass"] and k3["1"]["x"]["ground"]["gated"] is False          # reported line
    assert k3["1"]["y"]["tables"]["pass"] and k3["1"]["y"]["ground"]["pass"] is False
    assert k3["1"]["z"]["tables"]["pass"] is None and "gated cell set incomplete" in k3["1"]["z"]["tables"]["why"]
    assert k3["1"]["zr"]["tables"]["pass"] is True                     # dropping REPORTED cells never refuses the line
    assert C.k3_read({"2": {"x": C.pooled_views(subj)}})["2"]["x"]["tables"]["pass"] is None   # no reference
    fl = C.rh_floor(pv)
    assert fl["_all_gated_pass"] and fl["_complete"] and fl["_missing_gated_cells"] == []
    gated = {k for k, v in fl.items() if not k.startswith("_") and v["gated"]}
    assert gated == {"top table_0p635", "top table_0p6985", "top table_0p762"}
    assert {k for k, v in fl.items() if not k.startswith("_") and not v["gated"]} == {
        f"cell {c}" for c in C.K3_REPORTED_CELLS}
    low = dict(ref, **{"ground__rod_min": _agg(stood=85), "ground__knee_folded": _agg(stood=0, falls=31)})
    fl = C.rh_floor(C.pooled_views(low))
    assert fl["_all_gated_pass"] and not fl["cell ground__rod_min"]["floor_pass"]   # reported: listed, never a miss
    falls = dict(ref, **{"table_0p762__knee_folded": _agg(stood=95, falls=3)})
    fl = C.rh_floor(C.pooled_views(falls))                                       # 3 of 200 on that top: 1.5 / 100
    assert not fl["top table_0p762"]["floor_pass"] and not fl["_all_gated_pass"]


def test_k3_read_compares_gated_cell_sets_not_counts(C, tmp_path):
    """Review fix: equal cell COUNTS with different (or both short) gated cell SETS never read as a ``tables`` line --
    in k3_read, the merge summary and the printed table; the reported ``ground`` line compares sets too."""
    ref = {c: _agg(stood=95) for c in C.K3_CELLS}
    subj = {c: _agg(stood=93) for c in C.K3_CELLS}
    a, b = "table_0p635__knee_folded", "table_0p762__random"
    pv = C.pooled_views(ref)
    assert pv["tables"]["cell_ids"] == list(C.K3_GATED_CELLS)
    assert pv["ground"]["cell_ids"] == [c for c in C.K3_CELLS if c.startswith("ground")]
    drop = lambda d, *cs: {c: v for c, v in d.items() if c not in cs}  # noqa: E731
    cases = {"different": (drop(ref, a), drop(subj, b)),      # 5 vs 5 cells, different sets
             "same": (drop(ref, a), drop(subj, a)),           # both short the same gated cell
             "ref_short": (drop(ref, a), subj)}
    for name, (r, s) in cases.items():
        k = C.k3_read({"1": {"r_h": C.pooled_views(r), "x": C.pooled_views(s)}})["1"]["x"]["tables"]
        assert k["pass"] is None and "gated cell set incomplete" in k["why"], name
        assert k["gated"] is True and a in k["why"], name
    g1, g2 = "ground__knee_folded", "ground__random"
    k = C.k3_read({"1": {"r_h": C.pooled_views(drop(ref, g1)), "x": C.pooled_views(drop(subj, g2))}})["1"]["x"]
    assert k["ground"]["pass"] is None and "cell sets differ" in k["ground"]["why"]
    assert k["tables"]["pass"] is True                         # a reported gap never refuses the gated line
    # through the per-cell files, the merge summary and the printed table: no tables PASS over 5 cells
    _write_cells(C, tmp_path, "r_h", 1, [c for c in C.K3_CELLS if c != a], stood=97)
    _write_cells(C, tmp_path, "stand_net", 1, [c for c in C.K3_CELLS if c != b], stood=96)
    m = C.merge_cells(sorted(tmp_path.glob("*.json")))
    assert m["k3"]["1"]["stand_net"]["tables"]["pass"] is None
    row = next(ln for ln in C.format_table(m).splitlines() if "stand_net" in ln and " tables " in ln)
    assert "PASS" not in row and "FAIL" not in row


def test_rh_floor_is_per_gated_top_pooled_over_the_two_gated_poses(C):
    ref = {c: _agg(stood=100) for c in C.K3_CELLS}
    # 0.635: knee_folded 85 + random 96 = 181 / 200 = 0.905 -- one pose alone would miss, the pooled top passes
    mix = dict(ref, **{"table_0p635__knee_folded": _agg(stood=85), "table_0p635__random": _agg(stood=96)})
    t = C.rh_floor(C.pooled_views(mix))["top table_0p635"]
    assert (t["n_envs"], t["n_stood"], t["cells"], t["poses"]) == (200, 181, 2, ["knee_folded", "random"])
    assert t["stood_share"] == pytest.approx(0.905) and t["floor_pass"]
    # the top's rod_min cell is not in its pool: a terrible one moves nothing
    t2 = C.rh_floor(C.pooled_views(dict(mix, **{"table_0p635__rod_min": _agg(stood=0, falls=100)})))["top table_0p635"]
    assert {k: t2[k] for k in ("n_envs", "n_stood", "falls", "floor_pass")} == {
        k: t[k] for k in ("n_envs", "n_stood", "falls", "floor_pass")}
    # 179 / 200 = 0.895 < 0.90: MISS on that top only
    fl = C.rh_floor(C.pooled_views(dict(ref, **{"table_0p635__knee_folded": _agg(stood=83),
                                                "table_0p635__random": _agg(stood=96)})))
    assert not fl["top table_0p635"]["floor_pass"] and fl["top table_0p6985"]["floor_pass"] and not fl["_all_gated_pass"]
    # falls pooled over both poses: 2 / 200 = 1.0 per 100 passes, 3 / 200 = 1.5 misses
    f2 = dict(ref, **{"table_0p6985__knee_folded": _agg(stood=99, falls=1), "table_0p6985__random": _agg(stood=99,
                                                                                                         falls=1)})
    assert C.rh_floor(C.pooled_views(f2))["top table_0p6985"]["falls_per100"] == pytest.approx(1.0)
    assert C.rh_floor(C.pooled_views(f2))["_all_gated_pass"]
    f3 = dict(f2, **{"table_0p6985__random": _agg(stood=98, falls=2)})
    assert not C.rh_floor(C.pooled_views(f3))["top table_0p6985"]["floor_pass"]
    # a gated cell missing: incomplete, never a pass (the merge prints INCOMPLETE, not PASS)
    fl = C.rh_floor(C.pooled_views({c: a for c, a in ref.items() if c != "table_0p762__random"}))
    assert not fl["_complete"] and fl["_missing_gated_cells"] == ["table_0p762__random"] and not fl["_all_gated_pass"]
    assert fl["top table_0p762"]["cells"] == 1 and fl["top table_0p762"]["poses"] == ["knee_folded"]


def test_a_terrible_reported_cell_never_flips_a_k3_line_or_the_floor(C, tmp_path):
    ref = {c: _agg(stood=96) for c in C.K3_CELLS}
    subj = {c: _agg(stood=93) for c in C.K3_CELLS}
    terrible = _agg(stood=0, falls=100)

    def read(r, s):
        k = C.k3_read({"1": {"r_h": C.pooled_views(r), "x": C.pooled_views(s)}})["1"]["x"]["tables"]
        fl = C.rh_floor(C.pooled_views(r))
        return k, {k_: v["floor_pass"] for k_, v in fl.items() if k_.startswith("top ")}, fl["_all_gated_pass"]

    base = read(ref, subj)
    assert base[0]["pass"] is True and base[2] is True
    for cell in C.K3_REPORTED_CELLS:
        for r, s in ((ref, dict(subj, **{cell: terrible})), (dict(ref, **{cell: terrible}), subj),
                     (dict(ref, **{cell: terrible}), dict(subj, **{cell: terrible}))):
            assert read(r, s) == base, cell
    # the same injection in a GATED cell flips the line (subject) and the floor (R_h)
    for cell in C.K3_GATED_CELLS:
        assert read(ref, dict(subj, **{cell: terrible}))[0]["pass"] is False, cell
        assert read(dict(ref, **{cell: terrible}), subj)[2] is False, cell
    # end to end through the per-cell files, the merge and the printed table
    _write_cells(C, tmp_path, "r_h", 1, C.K3_CELLS, stood=97)
    _write_cells(C, tmp_path, "stand_net", 1, C.K3_GATED_CELLS, stood=96)
    _write_cells(C, tmp_path, "stand_net", 1, C.K3_REPORTED_CELLS, stood=0)
    for cell in C.K3_REPORTED_CELLS:                                       # R_h too: every reported cell terrible
        f = tmp_path / f"r_h_s1__{cell}.json"
        rec = json.loads(f.read_text())
        rec["aggregate"].update(n_stood=0, falls=100, falls_per100=100.0, stood_share=0.0)
        f.write_text(json.dumps(rec))
    m = C.merge_cells(sorted(tmp_path.glob("*.json")))
    assert m["k3"]["1"]["stand_net"]["tables"]["pass"] is True and m["rh_floor"]["1"]["_all_gated_pass"] is True
    assert m["gating"]["gated_cells"] == list(C.K3_GATED_CELLS)
    txt = C.format_table(m)
    assert "K3 GATED cells (6" in txt and "K3 REPORTED cells (9" in txt and "-> PASS" in txt
    assert "R_h reported es1 (never gated)" in txt and "cell ground__knee_folded 0/100 falls 100" in txt
    row = next(ln for ln in txt.splitlines() if "stand_net" in ln and "ground__rod_min" in ln)
    assert " reported " in row
    row = next(ln for ln in txt.splitlines() if "stand_net" in ln and "table_0p6985__random" in ln)
    assert " GATED " in row


def _write_cells(C, d: Path, label, seed, cells, stood=95, controller=None, video=False):
    for cell in cells:
        per = [_cls(C)] * stood + [_cls(C, rise=0.0)] * (100 - stood)
        meta = {"label": label, "controller": controller or label, "seed": seed, "video": video, "smoke": False,
                "episode_s": 16.0, "mp4": "x.mp4" if video else None}
        rec = C.cell_record(meta, cell, [], per)
        (d / f"{label}_s{seed}{'_video' if video else ''}__{cell}.json").write_text(json.dumps(rec))


def test_merge_cells_and_format(C, tmp_path):
    _write_cells(C, tmp_path, "r_h", 1, C.K3_CELLS, stood=96)
    _write_cells(C, tmp_path, "stand_net_abcd1234", 1, C.K3_CELLS, stood=92, controller="stand_net")
    _write_cells(C, tmp_path, "posture_only", 1, C.K3_CELLS, stood=90)            # 0.90 < 0.96 - 5 pp
    _write_cells(C, tmp_path, "r_h", 1, ["ground__knee_folded"], video=True)
    (tmp_path / "junk.json").write_text(json.dumps({"kind": "other"}))
    m = C.merge_cells(sorted(tmp_path.glob("*.json")))
    assert set(m["cells"]["1"]) == {"r_h", "stand_net_abcd1234", "posture_only"} and len(m["videos"]) == 1
    assert m["cells"]["1"]["r_h"]["table_0p635__rod_min"]["n_stood"] == 96
    k = m["k3"]["1"]["stand_net_abcd1234"]
    assert k["tables"]["pass"] is True and k["ground"]["n"] == 300
    assert m["k3"]["1"]["posture_only"]["tables"]["pass"] is False
    assert m["k3"]["1"]["posture_only"]["tables"]["rate_line"] == pytest.approx(0.96 - 0.05)
    assert m["rh_floor"]["1"]["_all_gated_pass"] and m["k3_episode_s"] == 16.0
    txt = C.format_table(m)
    assert "table_0p635__rod_min" in txt and "PASS" in txt and "R_h floor es1" in txt and "ref" in txt
    dup = tmp_path / "dup"
    dup.mkdir()
    _write_cells(C, dup, "r_h", 1, ["ground__rod_min"])
    (dup / "copy__ground__rod_min.json").write_text((dup / "r_h_s1__ground__rod_min.json").read_text())
    with pytest.raises(ValueError):
        C.merge_cells(sorted(dup.glob("*.json")))


# ===================================================================================== the Kit script (Kit-free parts)
def test_crab_failure_params_and_env_composition(T, C, plan1):
    sc = types.SimpleNamespace(name="contact_forces", body_names=[".*_Hip"])
    crab = types.SimpleNamespace(func=lambda: None, params={"limit_angle": 0.5, "contact_force_threshold": 500.0,
                                                            "hip_contact_sensor_cfg": sc}, time_out=False)
    with pytest.raises(SystemExit):
        T.crab_failure_params(types.SimpleNamespace(total_terminates=None))
    assert T.crab_failure_params(types.SimpleNamespace(crab_failure=crab)) == {
        "limit_angle": 0.5, "contact_force_threshold": 500.0, "hip_body_names": [".*_Hip"], "sensor": "contact_forces"}

    class Rec:
        def __init__(self, **k):
            self.__dict__.update(k)

    class ROCfg(Rec):
        InitialStateCfg = Rec

    sim = types.SimpleNamespace(CuboidCfg=Rec, RigidBodyPropertiesCfg=Rec, MassPropertiesCfg=Rec,
                                CollisionPropertiesCfg=Rec, PreviewSurfaceCfg=Rec)
    ev = types.SimpleNamespace(
        reset_root_state=types.SimpleNamespace(func=None, mode="reset", params={"yaw_range": (0.0, 0.0), "offset": 0.0}),
        reset_robot_joints=types.SimpleNamespace(func=None, mode="reset", params={"position_range": (0.95, 1.05)}),
        physics_material=types.SimpleNamespace(func=None, mode="startup", params={"friction_range": (0.6, 1.0)}),
        push_robot=types.SimpleNamespace(func=None, mode="interval", params={}),
        randomize_mass=types.SimpleNamespace(func=None, mode="startup", params={}))
    env_cfg = types.SimpleNamespace(
        terminations=types.SimpleNamespace(total_terminates=types.SimpleNamespace(func=None, params={}),
                                           crab_failure=crab),
        events=ev, scene=types.SimpleNamespace(terrain=types.SimpleNamespace(physics_material={"f": 1})),
        commands=types.SimpleNamespace(base_velocity=types.SimpleNamespace(debug_vis=True)),
        viewer=types.SimpleNamespace(), observations=types.SimpleNamespace(policy=types.SimpleNamespace(
            enable_corruption=False)))
    o, _ = T.parse(["--controller", "r_h", "--out", "/tmp/x"])
    names = [f"{leg}_{j}" for leg in C.LEGS for j in ("Body_Hip_RevoluteJoint", "Body_CamShaft_RevoluteJoint",
                                                      "Hip_Femur_RevoluteJoint", "Femur_Tibia_RevoluteJoint")]
    params = C.event_params(plan1[:4], names)
    changes, fail = T._compose_env_cfg(env_cfg, o, event_params=params, friction=None, sim_utils=sim, RigidObjectCfg=ROCfg,
                                       EventTermCfg=Rec, TerminationTermCfg=Rec, il_time_out="TIMEOUT", ep_s=16.0)
    assert fail["contact_force_threshold"] == 500.0 and fail["limit_angle"] == 0.5
    assert env_cfg.terminations.crab_failure is None and env_cfg.terminations.total_terminates.func == "TIMEOUT"
    assert ev.reset_robot_joints is None and ev.push_robot is None and ev.randomize_mass is None
    assert ev.physics_material.params["friction_range"] == (0.8, 0.8)
    assert list(vars(ev))[-1] == "sh_start" and ev.sh_start.func is T.CORE.sh_start and ev.sh_start.params is params
    tab = env_cfg.scene.sh_table
    assert tab.prim_path == "{ENV_REGEX_NS}/ShTable" and tab.spawn.size == (0.90, 0.70, C.TABLE_HEIGHT_M)
    assert tab.spawn.rigid_props.kinematic_enabled is True and env_cfg.commands.base_velocity.debug_vis is False
    assert any(c["what"] == "terminations.crab_failure" for c in changes)
    ev.weird = types.SimpleNamespace(func=None, mode="reset", params={})
    env_cfg.terminations.crab_failure = crab
    with pytest.raises(SystemExit):
        T._compose_env_cfg(env_cfg, o, event_params=params, friction=None, sim_utils=sim, RigidObjectCfg=ROCfg,
                           EventTermCfg=Rec, TerminationTermCfg=Rec, il_time_out="TIMEOUT", ep_s=16.0)


def test_parse_labels_and_prefixes(T, C, SH):
    o, rest = T.parse(["--controller", "r_h", "--out", "/tmp/x", "--seed", "2", "--headless"])
    assert o.label == "r_h" and o.prefix == "r_h_s2" and rest == ["--headless"]
    if AMASK.is_file():
        o, _ = T.parse(["--controller", "stand_net", "--out", "/tmp/x"])
        assert o.label == "stand_net_" + SH.STAND_NET_SHA256[:8]
    o, _ = T.parse(["--controller", "s1", "--out", "/tmp/x", "--video", "--cells", "ground__rod_min"])
    assert o.prefix == "s1_s1_video_ground__rod_min"
    with pytest.raises(SystemExit):
        T.parse(["--controller", "r_h", "--cells=all", "--envs-per-cell=x"])
    assert T.episode_s(o) == C.k3_episode_s()


def test_pre_kit_refusals(T, tmp_path):
    def reasons(*args):
        o, rest = T.parse(list(args))
        return "\n".join(T.pre_kit_reasons(o, rest))

    r = reasons("--controller", "r_h", "--out", "relative/x", "--video", "--envs-per-cell", "3")
    assert "not absolute" in r and "--enable_cameras" in r and "exactly one cell" in r
    r = reasons("--controller", "r_h", "--out", str(tmp_path), "--envs-per-cell", "20")
    assert "n >= 100" in r
    r = reasons("--controller", "r_h", "--out", str(tmp_path), "--episode-s", "8")
    assert "pinned K3 length" in r
    r = reasons("--controller", "r_h", "--out", str(tmp_path), "--cells", "table_0p762__rod_min", "--h-cmd", "1.0",
                "--smoke", "--envs-per-cell", "2")
    assert "plan:" in r
    r = reasons("--controller", "r_h", "--out", str(tmp_path), "--cells", "nope")
    assert "cell" in r
    if T.DEFAULT_S1.is_file() and AMASK.is_file() and (Path(T.SPEC.REFS_ENV_STAMP) / "run_meta.json").is_file():
        assert reasons("--controller", "r_h", "--out", str(tmp_path), "--headless", "--jitter") == ""
        assert reasons("--controller", "stand_net", "--out", str(tmp_path), "--smoke", "--envs-per-cell", "2",
                       "--episode-s", "2", "--cells", "ground__rod_min,table_0p635__knee_folded") == ""
        assert reasons("--controller", "r_h", "--out", str(tmp_path), "--video", "--enable_cameras", "--envs-per-cell",
                       "1", "--cells", "ground__knee_folded") == ""
        o, rest = T.parse(["--controller", "r_h", "--out", str(tmp_path), "--cells", "ground__rod_min"])
        (tmp_path / f"{o.prefix}__ground__rod_min.json").write_text("{}")
        assert any("exists" in x for x in T.pre_kit_reasons(o, rest))
        o.overwrite = True
        assert not any("exists" in x for x in T.pre_kit_reasons(o, rest))
        other = tmp_path / "other.pt"
        other.write_bytes(b"x")
        assert "A-mask s17 only" in reasons("--controller", "amask_old", "--out", str(tmp_path), "--stand-ckpt",
                                            str(other))
    assert T.main(["sh_table_test.py", "--controller", "s1", "--out", "rel"]) == T.REFUSED_RC


def test_emit_facts_and_merge_end_to_end(T, C, tmp_path, capsys):
    assert T.main(["sh_table_test.py", "--emit-facts", "--out", str(tmp_path / "f")]) == 0
    f = json.loads((tmp_path / "f" / "facts.json").read_text())
    assert f["k3_episode_s"] == 16.0 and len(f["k3_cells"]) == 15 and f["h_min"] == 0.719 and f["h_max"] == 1.346
    assert len(f["k3_gated_cells"]) == 6 and len(f["k3_reported_cells"]) == 9 and f["k3_reported_tops_m"] == [0.4572]
    assert f["_sources"]["r_knob"] == T.SN.RESTANCE_D3 and set(f["poses"]) == {"knee_folded", "rod_min", "random"}
    assert f["clip"]["harness_clip_actions"] == 4.8 and "floor_rule" in f
    d = tmp_path / "m"
    d.mkdir()
    _write_cells(C, d, "r_h", 1, C.K3_CELLS, stood=97)
    _write_cells(C, d, "s1", 1, C.K3_CELLS, stood=5)
    (d / "s1_s1_summary.json").write_text(json.dumps({"kind": "sh_table_job"}))
    (d / "facts.json").write_text("{}")
    assert T.main(["sh_table_test.py", "--merge", str(d)]) == 0
    m = json.loads((d / "table_test_summary.json").read_text())
    assert m["k3"]["1"]["s1"]["tables"]["pass"] is False and m["rh_floor"]["1"]["_all_gated_pass"]
    assert "FAIL" in capsys.readouterr().out


# ===================================================================================== the launcher
def _pat(name: str) -> str:
    m = re.search(rf"^{name}='([^']*)'$", SH_PATH.read_text(), flags=re.M)
    assert m, f"{name} not found"
    return m.group(1)


def test_launcher_busy_pattern():
    pat, nonkit = re.compile(_pat("ISAAC_PAT")), re.compile(_pat("NONKIT_PAT"))
    pred = re.search(r"^ISAAC_PAT='([^']*)'$", (PRED / "sn_table_test.sh").read_text(), flags=re.M).group(1)
    names = lambda p: set(re.search(r"\(([^()]*\[a-z_\]\+[^()]*)\)\\\.py", p).group(1).split("|"))  # noqa: E731
    assert names(pred) <= names(_pat("ISAAC_PAT"))                       # never forgets a predecessor busy name
    py = "/home/nickmagus/krabby/isaac_venv/bin/python"
    busy = [f"{py} /x/sh_table_test.py --headless --controller r_h --prefix a", f"{py} /x/sh_dagger.py --mode dagger",
            f"{py} /x/sh_eval_crab_hex_gait.py --headless", f"{py} /x/sn_table_test.py --headless",
            f"{py} -u /x/scripts/rsl_rl/train.py --task X", f"{py} /x/sn_record_video.py --headless"]
    idle = [f"systemd-run --user --scope -q timeout 1 {py} /x/sh_table_test.py --headless",
            f"bash -c {py} /x/sh_table_test.py", f"{py} -m pytest -q tests/unit/test_sh_table.py"]
    for line in busy:
        assert pat.search(line) and not nonkit.search(line), line
    for line in idle:
        assert not pat.search(line), line
    for line in (f"{py} -B /x/sh_table_test.py --emit-facts --out /y", f"{py} -B /x/sh_table_test.py --merge /y",
                 f"{py} /x/scripts/sh_facts.py", f"{py} -B /x/sn_dagger.py --read /y"):
        assert pat.search(line) and nonkit.search(line), line


def test_launcher_cell_lists_match_the_core(C):
    """The launcher's surfaces / smoke / video cells follow sh_table_core after the step-4 re-scope: the full run's
    ``--cells all`` holds the reported cells too (the 18-in table and the ground), the smoke exercises a gated and a
    reported cell, the default videos are gated cells."""
    src = SH_PATH.read_text()
    surf = re.search(r'^\s*SURFACES="([^"]*)"$', src, flags=re.M).group(1).split()
    assert tuple(surf) == C.SURFACES and "table_0p4572" in surf and "ground" in surf
    vc = re.search(r'^\s*VCELLS="([^"]*)"$', src, flags=re.M).group(1).split()
    sc = re.search(r'^\s*SC="([^"]*)"$', src, flags=re.M).group(1).split(",")
    for c in vc + sc:
        C.parse_cell(c)
    assert all(C.cell_gated(c) for c in vc)
    assert any(C.cell_gated(c) for c in sc) and any(not C.cell_gated(c) for c in sc)
    assert set(C.parse_cells("all")) == set(C.K3_GATED_CELLS) | set(C.K3_REPORTED_CELLS)


def _dry(env, *args):
    r = subprocess.run(["bash", str(SH_PATH), "--dry-run", *args], env=env, capture_output=True, text=True, timeout=60)
    return r, [ln for ln in r.stdout.splitlines() if ln.startswith("systemd-run")]


def test_launcher_syntax_and_dry_runs(tmp_path):
    subprocess.run(["bash", "-n", str(SH_PATH)], check=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "LAUNCHED"
    for tool in ("systemd-run", "setsid", "nohup"):
        (fake_bin / tool).write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
        (fake_bin / tool).chmod(0o755)
    out = tmp_path / "out"
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ.get('PATH', '')}", SH_DDC=str(tmp_path / "ddc"),
               SH_TABLE_OUT=str(out), CUDA_VISIBLE_DEVICES="")
    r, jobs = _dry(env, "--smoke")
    assert r.returncode == 0, r.stderr
    for needle in ("OMNI_KIT_ACCEPT_EULA=yes", "MemoryMax=24G", "MemorySwapMax=0", "--headless", "--emit-facts",
                   "timeout --signal=TERM 3600 ", "--merge"):
        assert needle in r.stdout, needle
    assert " -k " not in r.stdout and "--kill-after" not in r.stdout and "KILL" not in r.stdout   # never SIGKILL Kit
    assert len(jobs) == 3 and ["--enable_cameras" in j for j in jobs] == [False, False, True]
    assert all("--smoke" in j and "--episode-s 2" in j for j in jobs)
    assert "--controller r_h --cells table_0p635__knee_folded,ground__rod_min,table_0p6985__random" in jobs[0]
    assert "--controller stand_net" in jobs[1] and "--envs-per-cell 2" in jobs[1]
    assert "--controller r_h --cells table_0p635__knee_folded --envs-per-cell 1" in jobs[2] and "--video" in jobs[2]
    r, jobs = _dry(env, "--smoke", "--smoke-no-video")
    assert r.returncode == 0 and len(jobs) == 2 and "--enable_cameras" not in r.stdout
    r, jobs = _dry(env)
    assert r.returncode == 0, r.stderr
    stats = [j for j in jobs if "--enable_cameras" not in j]
    vids = [j for j in jobs if "--enable_cameras" in j]
    assert len(stats) == 10 and len(vids) == 6                            # 2 seeds x 5 controllers; 2 x 3 videos
    for s in (1, 2):
        for c in ("r_h", "stand_net", "posture_only", "s1", "amask_old"):
            assert sum(f"--controller {c} --cells all --envs-per-cell 100 --seed {s} --jitter" in j and
                       f"--prefix stats_{c}_s{s} " in j + " " for j in stats) == 1, (c, s)
    assert all("--video" in j and "--envs-per-cell 1" in j and "--seed 1" in j for j in vids)
    assert not any("--stand-ckpt" in j for j in jobs)
    r, jobs = _dry(env, "--split", "surface", "--seeds", "1", "--stand-ckpt", "/abs/ck.pt", "--label", "pilot18",
                   "--videos", "r_h")
    stats = [j for j in jobs if "--enable_cameras" not in j]
    assert r.returncode == 0 and len(stats) == 25                         # 5 controllers x 5 surfaces
    for surf in ("table_0p4572", "table_0p635", "table_0p6985", "table_0p762", "ground"):   # = sh_table_core.SURFACES
        assert sum(f"--cells {surf} " in j for j in stats) == 5, surf
    for j in jobs:
        assert ("--stand-ckpt /abs/ck.pt --label pilot18" in j) == ("--controller stand_net" in j), j
    assert not marker.exists() and not out.exists()                      # a dry run launches and writes nothing
    for bad in (["--controllers", "walk"], ["--envs-per-cell", "0"], ["--split", "cell"], ["--bogus"], ["--seeds", "a"],
                ["--label", "a/b"], ["--cells", "a b"]):
        r, _ = _dry(env, *bad)
        assert r.returncode == 5, (bad, r.stderr)
    r, _ = _dry(dict(env, SH_TABLE_TIMEOUT="0"))
    assert r.returncode == 5 and "SH_TABLE_TIMEOUT" in r.stderr


def _fake_tools(tmp_path, fail_first: bool = False, fail_merge: bool = False):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "systemd-run").write_text('#!/bin/bash\nwhile [ $# -gt 0 ] && [ "$1" != timeout ]; do shift; done\n'
                                          'exec "$@"\n')
    (fake_bin / "systemd-run").chmod(0o755)
    calls = tmp_path / "calls.txt"
    py = tmp_path / "fakepy"
    py.write_text(f"""#!/bin/bash
args="$*"
out=""; prefix=""
while [ $# -gt 0 ]; do
  case "$1" in --out) out=$2; shift 2 ;;
    --merge) if [ "{int(fail_merge)}" = 1 ]; then echo "merge boom" >&2; exit 2; fi
             echo merged > "$2/table_test_summary.json"; exit 0 ;;
    --prefix) prefix=$2; shift 2 ;; *) shift ;; esac
done
case "$args" in *--emit-facts*) echo '{{}}' > "$out/facts.json"; exit 0 ;; esac
test -f "$out/facts.json" || exit 8
echo "$args" >> {calls}
if [ "{int(fail_first)}" = 1 ] && [ "$(grep -c . {calls})" = 1 ]; then exit 9; fi
echo '{{}}' > "$out/${{prefix}}_summary.json"
""")
    py.chmod(0o755)
    return fake_bin, py, calls


def _queue_env(tmp_path, fake_bin, py):
    return dict(os.environ, PATH=f"{fake_bin}:{os.environ.get('PATH', '')}", SH_PY=str(py), SH_DDC=str(tmp_path / "ddc"),
                SH_TABLE_OUT=str(tmp_path / "out"), CUDA_VISIBLE_DEVICES="", SH_TABLE_TIMEOUT="60")


def _isaac_busy() -> bool:
    pat, nonkit = re.compile(_pat("ISAAC_PAT")), re.compile(_pat("NONKIT_PAT"))
    lines = subprocess.run(["ps", "-ww", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return any(pat.search(ln) and not nonkit.search(ln) for ln in lines)


def test_launcher_queue_runs_jobs_serially_and_merges(tmp_path):
    if _isaac_busy():
        pytest.skip("an Isaac process is running on this machine (the launcher would refuse)")
    fake_bin, py, calls = _fake_tools(tmp_path)
    r = subprocess.run(["bash", str(SH_PATH), "--smoke", "--foreground"], env=_queue_env(tmp_path, fake_bin, py),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (r.stdout, r.stderr)
    stamp = Path((tmp_path / "out" / "LATEST").read_text().strip())
    assert stamp.parent == tmp_path / "out" and stamp.name.endswith("_smoke")
    q = (stamp / "queue.log").read_text()
    names = ("smoke_r_h", "smoke_stand_net", "smoke_video_r_h_table_0p635__knee_folded")
    for name in names:
        assert f"SHTABLE JOB START {name}" in q and f"SHTABLE JOB END {name} rc=0" in q
        assert (stamp / f"{name}_summary.json").is_file() and (stamp / f"{name}.unit").is_file()
    assert [q.index(f"SHTABLE JOB START {n}") for n in names] == sorted(q.index(f"SHTABLE JOB START {n}") for n in names)
    assert "SHTABLE QUEUE END rc=0" in q and (stamp / "facts.json").is_file()
    assert (stamp / "table_test_summary.json").read_text().strip() == "merged"
    jobs = calls.read_text().splitlines()
    assert len(jobs) == 3 and all("--headless" in j and f"--out {stamp}" in j for j in jobs)
    assert "--enable_cameras" in jobs[2] and "--video" in jobs[2]


def test_launcher_failures(tmp_path):
    if _isaac_busy():
        pytest.skip("an Isaac process is running on this machine (the launcher would refuse)")
    fake_bin, py, calls = _fake_tools(tmp_path, fail_first=True)
    r = subprocess.run(["bash", str(SH_PATH), "--smoke", "--foreground"], env=_queue_env(tmp_path, fake_bin, py),
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 1, (r.stdout, r.stderr)
    q = (Path((tmp_path / "out" / "LATEST").read_text().strip()) / "queue.log").read_text()
    assert "SHTABLE JOB END smoke_r_h rc=9" in q and "SHTABLE JOB START smoke_stand_net" not in q
    assert "SHTABLE QUEUE END rc=1" in q and len(calls.read_text().splitlines()) == 1
    (tmp_path / "b").mkdir()
    fb2, py2, _ = _fake_tools(tmp_path / "b", fail_merge=True)
    r = subprocess.run(["bash", str(SH_PATH), "--smoke", "--smoke-no-video", "--foreground"],
                       env=_queue_env(tmp_path / "b", fb2, py2), capture_output=True, text=True, timeout=120)
    assert r.returncode == 6, (r.stdout, r.stderr)
    q = (Path((tmp_path / "b" / "out" / "LATEST").read_text().strip()) / "queue.log").read_text()
    assert "merge failed" in q and "SHTABLE QUEUE END rc=6" in q and "SHTABLE QUEUE END rc=0" not in q


# ===================================================================================== the residual deadband
# CHARTER section 3, 2026-10-14 "probe a residual deadband": --deadband D on stand_net only, recorded per cell, the
# merge never pools two deadbands under one label.
def test_controller_deadband_stand_net_only(C, SH):
    n = 6
    obs = _obs(n, 3)
    s1_est, st_est = _Est(P1 - P0, 0.01), _Est(P1 - P0, -0.02)
    s1 = _Net(lambda x: 1.7 * torch.tanh(3.0 * x[:, :18]))
    st = _Net(lambda x: 0.4 * torch.tanh(x[:, 65:83] + x[:, P0:P0 + 1]))   # small residuals: many below 0.25
    slot = torch.tensor([True, True, False, True, False, True])
    h = torch.tensor([0.8, 1.2, 1.064, 1.064, 0.9, 1.3], dtype=torch.float64)
    hc = SH.HeightController(n, "cpu", dt=DT)
    hc.reset(None, "walk", h_cmd=h)
    hs = hc.step(slot, h, obs)
    harness = obs.clone()
    harness[:, P0:P1] = s1_est.inference(harness[:, :NP])
    a_s1 = s1.fn(harness)
    kw = dict(s1=s1, s1_est=s1_est, p0=P0, p1=P1, num_prop=NP, st_pol=st, st_est=st_est, mask=1)
    x = SH.stand_net_input(harness, st_est, p0=P0, p1=P1, num_prop=NP, mask=1, h_cmd_limited=hs.h_lim,
                           restoring=hs.restoring)
    a0, _ = C.controller_actions("stand_net", obs, slot, hs, **kw)
    a00, _ = C.controller_actions("stand_net", obs, slot, hs, deadband=0.0, **kw)
    a1, ex = C.controller_actions("stand_net", obs, slot, hs, deadband=0.25, **kw)
    assert bitwise_equal(a0, a00) and bitwise_equal(a0, SH.router_actions(hs.stand_eff, st.fn(x), a_s1, hs.h_lim).actions)
    assert bitwise_equal(a1, SH.router_actions(hs.stand_eff, st.fn(x), a_s1, hs.h_lim, deadband=0.25).actions)
    assert not torch.equal(a1[slot], a0[slot]) and bitwise_equal(a1[~slot], a0[~slot])   # S1 rows untouched
    assert torch.equal(ex["a_net"], st.fn(x))                                             # the raw output recorded
    for ctl in ("r_h", "posture_only", "s1", "amask_old"):
        with pytest.raises(ValueError, match="deadband"):
            C.controller_actions(ctl, obs, slot, hs, deadband=0.25, **kw)
        C.controller_actions(ctl, obs, slot, hs, deadband=0.0, **kw)                     # 0 = off: fine
    assert C.DEADBAND_CONTROLLER == "stand_net"


def test_parse_label_and_pre_kit_deadband(T, tmp_path):
    assert (T.deadband_tag(0.25), T.deadband_tag(0.0), T.deadband_tag(0.1), T.deadband_tag(2.0)) == \
        ("_db0p25", "", "_db0p1", "")
    ck = tmp_path / "head.pt"
    ck.write_bytes(b"head")
    sha8 = T.SN.sha256_file(ck)[:8]
    o, _ = T.parse(["--controller", "stand_net", "--out", "/tmp/x", "--stand-ckpt", str(ck), "--deadband", "0.25"])
    assert o.deadband == 0.25 and o.label == f"stand_net_{sha8}_db0p25" and o.prefix == f"stand_net_{sha8}_db0p25_s1"
    o, _ = T.parse(["--controller", "stand_net", "--out", "/tmp/x", "--stand-ckpt", str(ck)])
    assert o.deadband == 0.0 and o.label == f"stand_net_{sha8}"                       # unchanged without it
    o, _ = T.parse(["--controller", "stand_net", "--out", "/tmp/x", "--stand-ckpt", str(ck), "--deadband", "0.25",
                    "--label", "SH_P18db"])
    assert o.label == "SH_P18db"

    def reasons(*args):
        o, rest = T.parse(list(args))
        return "\n".join(T.pre_kit_reasons(o, rest))

    assert "stand_net only" in reasons("--controller", "r_h", "--out", str(tmp_path), "--deadband", "0.25")
    assert "stand_net only" in reasons("--controller", "amask_old", "--out", str(tmp_path), "--deadband", "0.25")
    r = reasons("--controller", "stand_net", "--out", str(tmp_path), "--stand-ckpt", str(ck), "--deadband", "1.5")
    assert "--deadband:" in r
    assert "deadband" not in reasons("--controller", "r_h", "--out", str(tmp_path))
    with pytest.raises(SystemExit):
        T.parse(["--controller", "stand_net", "--deadbandx", "0.2"])


def test_kit_script_records_the_deadband_in_the_summary_and_every_cell():
    """The Kit path (not runnable on CPU) passes --deadband to the controller and writes it into the job meta (+ its
    clip / provenance) and every cell's meta, from which the merge and sh_read read it."""
    src = TEST_PATH.read_text()
    assert "st_pol=st, st_est=st_est, mask=int(o.stand_mask), deadband=float(o.deadband))" in src
    assert "SH.RESIDUAL_DEADBAND_KEY: float(o.deadband)," in src
    assert "SH.clip_record(float(o.deadband))" in src and "SH.provenance(float(o.deadband))" in src
    cm = re.search(r"cell_meta = \{k: meta\[k\] for k in \(([^)]*)\)\}", src, flags=re.S)
    assert cm and "SH.RESIDUAL_DEADBAND_KEY" in cm.group(1)


def _db_cells(C, d: Path, label, seed, cells, deadband=None, controller="stand_net"):
    _write_cells(C, d, label, seed, cells, stood=95, controller=controller)
    if deadband is not None:
        for cell in cells:
            f = d / f"{label}_s{seed}__{cell}.json"
            rec = json.loads(f.read_text())
            rec["meta"]["residual_deadband"] = deadband
            f.write_text(json.dumps(rec))


def test_merge_never_pools_two_deadbands_under_one_label(C, tmp_path):
    ok = tmp_path / "ok"
    ok.mkdir()
    _db_cells(C, ok, "r_h", 1, C.K3_CELLS, controller="r_h")                         # written before the option
    _db_cells(C, ok, "SH_P18db", 1, C.K3_CELLS, deadband=0.25)
    _db_cells(C, ok, "SH_P18", 1, C.K3_CELLS, deadband=0.0)
    m = C.merge_cells(sorted(ok.glob("*.json")))
    assert m["deadbands"] == {"r_h": 0.0, "SH_P18db": 0.25, "SH_P18": 0.0}
    assert m["cells"]["1"]["SH_P18db"]["table_0p635__knee_folded"]["residual_deadband"] == 0.25
    assert m["cells"]["1"]["r_h"]["table_0p635__knee_folded"]["residual_deadband"] == 0.0
    assert m["k3"]["1"]["SH_P18db"]["tables"]["pass"] is True
    txt = C.format_table(m)
    assert "residual deadband" in txt and "SH_P18db 0.25" in txt
    for name, writes in (
            ("one_cell", [("X", 1, C.K3_CELLS[:1], 0.25), ("X", 1, C.K3_CELLS[1:], None)]),
            ("two_seeds", [("X", 1, C.K3_CELLS, 0.25), ("X", 2, C.K3_CELLS, 0.0)]),
            ("two_values", [("X", 1, C.K3_CELLS[:2], 0.25), ("X", 1, C.K3_CELLS[2:4], 0.3)])):
        dd = tmp_path / name
        dd.mkdir()
        for label, seed, cells, db in writes:
            _db_cells(C, dd, label, seed, cells, deadband=db)
        with pytest.raises(ValueError, match="mixes residual deadbands"):
            C.merge_cells(sorted(dd.glob("*.json")))
    bad = tmp_path / "bad"
    bad.mkdir()
    _db_cells(C, bad, "X", 1, C.K3_CELLS[:1], deadband=True)
    with pytest.raises(ValueError, match="residual deadband"):
        C.merge_cells(sorted(bad.glob("*.json")))


def test_launcher_passes_the_deadband_to_stand_net_jobs_only(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "LAUNCHED"
    for tool in ("systemd-run", "setsid", "nohup"):
        (fake_bin / tool).write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
        (fake_bin / tool).chmod(0o755)
    out = tmp_path / "out"
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ.get('PATH', '')}", SH_DDC=str(tmp_path / "ddc"),
               SH_TABLE_OUT=str(out), CUDA_VISIBLE_DEVICES="")
    r, jobs = _dry(env, "--controllers", "r_h stand_net", "--videos", "r_h stand_net", "--stand-ckpt", "/abs/ck.pt",
                   "--label", "SH_P18db", "--deadband", "0.25", "--seeds", "1 2")
    assert r.returncode == 0, r.stderr
    assert len(jobs) == 4 + 6
    for j in jobs:
        assert ("--stand-ckpt /abs/ck.pt --label SH_P18db --deadband 0.25" in j) == ("--controller stand_net" in j), j
    r, jobs = _dry(env, "--controllers", "stand_net", "--videos", "stand_net")
    assert r.returncode == 0 and jobs and not any("--deadband" in j for j in jobs)   # off unless asked
    for bad in (["--deadband", "0"], ["--deadband", "1"], ["--deadband", "abc"], ["--deadband", ""],
                ["--deadband", "0.2", "--deadband", "0.2"],
                ["--controllers", "r_h", "--videos", "r_h", "--deadband", "0.25"]):
        r, _ = _dry(env, *bad)
        assert r.returncode == 5, (bad, r.stderr)
    assert not marker.exists() and not out.exists()
