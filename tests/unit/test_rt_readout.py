"""Rough-terrain campaign read-out (``crab_hex_omni_task/experiments/2026-11-10_rough_terrain/scripts``), on CPU.

The read-out decides the campaign's verdict, so a wrong statistic is quiet. Pinned here, on synthetic stamps:

* the generated manifest / env / row list are in sync with the spec (``make_manifest.py --check``) and follow the
  step-2 / step-4 FIXED INTERFACE (21 rows, the site env block, friction on the gravel rows only, no DR key);
* crossed = the HORIZONTAL max root distance from the SPAWN (the first raw sample) >= the kind's radius;
  success = complete AND crossed, stuck = complete AND not crossed;
* rows are keyed by checkpoint sha256 and composed env, never by the row-name prefix; duplicates and mismatches
  are reported;
* the paired statistics are the repo's (thresholds.paired_completion_se, select_student.completion_margin /
  completion_pairs), the G gain / regression and R lines follow PLAN Gates, k = 2;
* make_thresholds' step-7 rule (lowest in-envelope band <= 0.80, solved -> highest in-envelope band, 0 -> close,
  1 -> stop and ask);
* the hardening of the step-2 / step-4 reviews: rt_spec's radii / ladders pinned to the library's, per-seed n == 100 on
  kind rows, a failed launch never shadows an OK relaunch, the R tracking read fails closed (track_err / v1.2,
  side22 absolute), ``gates`` reads its rule from the frozen file and refuses changed scripts, a missing --spec is refused;
* on this machine only (skipped without the recorded stamps): the read-out reproduces TDR's recorded G1 numbers
  and passes the radius calibration.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "parkour" / "parkour_tasks" / "parkour_tasks" / "crab_hex_omni_task" / "experiments" / \
    "2026-11-10_rough_terrain"
SCRIPTS = CAMP / "scripts"
if not (SCRIPTS / "readout.py").is_file():
    pytest.skip("the rough-terrain campaign scripts are not in this checkout", allow_module_level=True)
pytest.importorskip("scipy")


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def M():
    S = _load("rt_spec", SCRIPTS / "rt_spec.py")
    RO = _load("rt_readout", SCRIPTS / "readout.py")
    MT = _load("rt_make_thresholds", SCRIPTS / "make_thresholds.py")
    MM = _load("rt_make_manifest", SCRIPTS / "make_manifest.py")
    return type("M", (), {"S": S, "RO": RO, "MT": MT, "MM": MM})


# ============================================================================== generated files
def test_generated_files_are_in_sync_with_the_spec(M, capsys):
    assert M.MM.main(["--check"]) == 0, capsys.readouterr().out


def test_kind_rows_follow_the_fixed_interface(M):
    S = M.S
    doc = yaml.safe_load(S.MANIFEST_PATH.read_text())
    base_doc = yaml.safe_load((S.OMNI / "experiments/2026-10-07_omni_phase2/scenarios_p2.yaml").read_text())
    fh = next(s for s in base_doc["scenarios"] if s["id"] == "field_hard__omni")
    assert {k: v for k, v in doc.items() if k != "scenarios"} == {k: v for k, v in base_doc.items() if k != "scenarios"}
    ids = [s["id"] for s in doc["scenarios"]]
    assert ids == [f"rt_{k}_{b}__omni" for k in ("slopes", "hills", "ledge_up", "ledge_down", "pits", "gravel",
                                                  "rubble") for b in ("b1", "b2", "b3")]
    bands = {"b1": "0:0.3", "b2": "0.3:0.6", "b3": "0.6:1"}
    for sc in doc["scenarios"]:
        kind, band = sc["id"][len("rt_"):-len("__omni")].rsplit("_", 1)
        want = {"KRABBY_FLAT_TERRAIN_MODE": "light", "KRABBY_FLAT_TERRAIN_FLAT_FRAC": "0.0",
                "KRABBY_FLAT_TERRAIN_GEOM": "site", "KRABBY_SITE_LADDER": "rt", "KRABBY_SITE_MIX": f"{kind}=20",
                "KRABBY_SITE_DIFF": bands[band], "KRABBY_FLAT_TERRAIN_DIFF": "0.20:1.0"}
        if kind == "gravel":
            want["KRABBY_RT_FRICTION"] = "gravel:0.2:0.5"
        assert sc["env"] == want, sc["id"]
        assert sc["schedule"] == fh["schedule"] and sc["episodes"] == 100 and sc["episode_length_s"] == 80.0
        assert sc["role"] == "teacher" and sc["spawn_yaw"] == fh["spawn_yaw"]
        assert not any(k.startswith("KRABBY_DR_") for k in sc["env"])


def test_kind_env_is_tdr_field_hard_minus_its_block(M):
    S = M.S
    src = S.read_env_file(S.OMNI / "experiments/2026-10-16_task4_pipeline/F19/envs/p3b.field_hard.env")
    block = {"KRABBY_FIELD_BLOCKS", "KRABBY_FIELD_EDGES", "KRABBY_FIELD_KINDS", "KRABBY_FIELD_PAD_R",
             "KRABBY_FLAT_TERRAIN_DIFF", "KRABBY_FLAT_TERRAIN_FLAT_FRAC", "KRABBY_FLAT_TERRAIN_GEOM",
             "KRABBY_FLAT_TERRAIN_MODE"}
    assert S.read_env_file(S.KIND_ENV_PATH) == {k: v for k, v in src.items() if k not in block}


def _lib_literal(name: str):
    """A top-level literal assignment of the library's ``site_terrains.py`` (by AST: no trimesh / Isaac import)."""
    import ast

    path = REPO / "parkour" / "parkour_isaaclab" / "terrains" / "omni" / "site_terrains.py"
    for node in ast.parse(path.read_text()).body:
        tgt = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else getattr(node, "target", None)
        if isinstance(tgt, ast.Name) and tgt.id == name and getattr(node, "value", None) is not None:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a literal top-level assignment of {path}")


def test_rt_spec_radii_and_ladders_cannot_drift_from_the_library(M):
    """The envelope rule and the crossing radii (rt_spec) are the library's own (site_terrains RT_CROSS_R / RT_LADDERS)."""
    S = M.S
    assert S.CROSS_R == _lib_literal("RT_CROSS_R")
    assert tuple(S.KINDS) == _lib_literal("RT_KINDS")
    lib, prim = _lib_literal("RT_LADDERS"), _lib_literal("RT_PRIMARY_PARAM")
    assert set(S.LADDERS) == set(lib) == set(S.KINDS)
    for kind, (_name, _unit, (bottom, top)) in S.LADDERS.items():
        param = prim.get(kind, "grade_deg")                     # slopes / hills have no metres parameter
        assert (bottom, top) == tuple(lib[kind][param]), (kind, param)
    assert {k for k in S.KINDS if k in prim} == set(prim)
    assert S.ENVELOPE_M == 0.14 and S.above_envelope_set() == tuple(S.ABOVE_ENVELOPE_EXPECTED)


def test_load_spec_refuses_a_missing_explicit_path(M, tmp_path):
    S = M.S
    with pytest.raises(FileNotFoundError, match="explicit --spec"):
        S.load_spec(tmp_path / "nope.yaml")
    assert S.load_spec(S.SPEC_PATH) == S.load_spec()
    with pytest.raises(FileNotFoundError):
        M.MM.main(["--spec", str(tmp_path / "nope.yaml"), "--check"])


def test_envelope_radii_and_retention_rows(M):
    S = M.S
    assert S.above_envelope_set() == (("ledge_up", "b3"), ("ledge_down", "b3"), ("pits", "b3"))
    assert S.CROSS_R == {"ledge_up": 3.2, "ledge_down": 3.2, "pits": 3.2, "rubble": 3.2, "slopes": 3.0,
                         "hills": 3.0, "gravel": 3.0}
    assert max(S.CROSS_R.values()) <= 3.2
    rows = S.load_rows()
    assert len(rows["kind_rows"]) == 21
    assert sum(r["above_envelope"] for r in rows["kind_rows"]) == 3
    assert [r["rt_friction"] is not None for r in rows["kind_rows"]].count(True) == 3
    ret = {r["id"]: r for r in rows["retention_rows"]}
    assert ret["R_field_hard"]["env_seeds"] == [1, 2, 3, 4]
    assert set(S.COMPLETION_R) == {"R_field_hard", "R_field_light", "R_ring_obst", "R_field_turn",
                                   "R_field_light_L090", "R_field_light_L135", "R_x07_L090_field_light"}
    for r in rows["retention_rows"]:
        env = S.expected_env(r)
        assert not any(k.startswith("KRABBY_DR_") or k == "KRABBY_RT_FRICTION" for k in env), r["id"]
    assert S.expected_env(ret["R_x07_L090_field_light"])["KRABBY_STRESS_ROD_SPEED"] == "0.7"
    assert S.expected_env(ret["R_field_light_L135"])["KRABBY_PAYLOAD_KG"] == "135:135"
    assert sum(S.TRAIN_MIX.values()) == 20


# ============================================================================== per-episode math
def test_r_max_is_horizontal_and_from_the_spawn(M):
    RO = M.RO
    tr = np.array([[10.0, 5.0, 1.0], [10.0, 5.0, 9.0], [11.0, 5.0, -3.0], [10.0, 6.5, 1.0]])
    assert RO.r_max_from(tr[:, :2], tr[0, :2]) == pytest.approx(1.5)          # z ignored, origin = spawn
    assert RO.r_max_from(tr[:, :2], (0.0, 0.0)) > 11.0                          # not the world origin
    rm = {0: 3.2, 1: 3.2 - 1e-6, 2: 4.0, 3: 0.1}
    assert RO.crossed_from(rm, 3.2) == {0: True, 1: False, 2: True, 3: False}
    comp = {0: True, 1: True, 2: False, 3: False}
    cro = RO.crossed_from(rm, 3.2)
    assert RO.success_from(comp, cro) == {0: True, 1: False, 2: False, 3: False}
    assert RO.stuck_from(comp, cro) == {0: False, 1: True, 2: False, 3: False}
    with pytest.raises(RO.KeySetError):
        RO.success_from({0: True}, {1: True})


def _maps(n: int, pa: float, pb: float, seed: int = 0):
    g = np.random.default_rng(seed)
    keys = [(es, i) for es in (1, 2) for i in range(n // 2)]
    a = {k: bool(x) for k, x in zip(keys, g.permutation(np.arange(n) < round(pa * n)))}
    b = {k: bool(x) for k, x in zip(keys, g.permutation(np.arange(n) < round(pb * n)))}
    return a, b


def test_g_line_gain_and_regression_rules(M):
    RO = M.RO
    TH = RO.th()
    a, b = _maps(200, 0.80, 0.50)
    ln = RO.g_line(a, b)
    se = TH.paired_completion_se({k: float(v) for k, v in a.items()}, {k: float(v) for k, v in b.items()})
    assert ln["se"] == se and ln["diff"] == pytest.approx(0.30)
    assert ln["gain_need"] == max(0.10, 3 * se) and ln["ci_lb"] == pytest.approx(0.30 - 1.96 * se)
    assert ln["gain"] and not ln["regress"]
    small = RO.g_line(*_maps(200, 0.53, 0.50, 1))                               # 0.03: below the noise floor
    assert small["le_noise"] and not small["gain"] and not small["regress"]
    reg = RO.g_line(*_maps(200, 0.40, 0.60, 2))
    assert reg["regress"] and not reg["gain"]
    assert reg["regress_margin"] == max(0.05, 2 * reg["se"])
    with pytest.raises(RO.KeySetError):
        RO.g_line({(1, 0): True}, {(1, 1): True})


def test_retention_line_is_select_students_completion_pairs(M):
    RO = M.RO
    a, b = _maps(200, 0.90, 0.95, 3)
    ln = RO.retention_line(a, b, "field_light")
    ref = next(x for x in RO.ss().completion_pairs({"field_light": a}, {"field_light": b}, RO.R_COMPLETION_CFG)
               if x["row"] == "field_light")
    assert ln["head"] == ref["student"] and ln["ref"] == ref["tstar"] and ln["se_diff"] == ref["se_diff"]
    assert ln["margin"] == ref["margin"] == max(0.05, 2 * ref["se_diff"]) and ln["pass"] == ref["pass"]
    assert ln["line"] == ref["line"]


def test_g_gate_k2_and_decision_rules(M):
    RO = M.RO
    g = RO.g_gate({"slopes": {"gain": True}, "pits": {"gain": True}, "hills": {"gain": False}})
    assert g["status"] == "PASS" and g["n_gain"] == 2
    assert RO.g_gate({"slopes": {"gain": True}, "pits": {"gain": False}})["status"] == "FAIL"
    assert RO.g_gate({"slopes": {"gain": True}, "pits": {"gain": True}, "hills": {"regress": True}})["status"] == "FAIL"
    assert RO.decide("PASS", "PASS")["verdict"] == "ACCEPT"
    assert RO.decide("PASS", "FAIL")["verdict"] == "HALT_R"
    assert RO.decide("FAIL", "PASS")["verdict"] == "NOT_ACCEPTED"
    assert RO.decide("FAIL", "INCOMPLETE")["verdict"] == "NOT_ACCEPTED"
    assert RO.decide("PASS", "INCOMPLETE")["verdict"] == "INCOMPLETE"
    assert RO.decide("INVALID", "PASS")["verdict"] == "INVALID"


def _zk(**over):
    zk = {"rows": [{"statistic": "track_err fwd", "row": "suite es1", "watch": False, "fails": False},
                   {"statistic": "field_hard completion", "row": "field_hard", "watch": False, "fails": True},
                   {"statistic": "ratio fwd", "row": "suite es1", "watch": True, "fails": True}],
          "side22": {"head": ["FAST", "FAST"], "p2a": ["FAST", "FAST"], "fast_fast_lost": False}, "fails": [],
          "translate_statistic": "track_err", "thresholds_version": "v1.2"}
    zk.update(over)
    return zk


def test_r_tracking_reads_suite_lines_and_side22_only(M):
    RO = M.RO
    zk = _zk()
    assert RO.r_tracking(zk)["status"] == "PASS"                    # the field line is report-only
    zk["rows"][0]["fails"] = True
    assert RO.r_tracking(zk)["status"] == "FAIL"
    zk["rows"][0]["fails"] = False
    zk["side22"]["fast_fast_lost"] = True
    assert RO.r_tracking(zk)["status"] == "FAIL"


def test_r_tracking_fails_closed_on_the_statistic_and_reads_side22_absolutely(M):
    RO = M.RO
    assert "relative" in RO.r_tracking(_zk())["side22_reading"] and "absolute" in RO.r_tracking(_zk())["side22_reading"]
    for over in ({"translate_statistic": "ratio_drop"}, {"thresholds_version": "v1.1"}, {"translate_statistic": None},
                 {"thresholds_version": None}):
        t = RO.r_tracking(_zk(**over))
        assert t["status"] == "INVALID" and t["fails"] and "fail closed" in t["fails"][0], over
    # TDR itself not FAST/FAST: 'not lost relative to TDR' would pass, the absolute check on the new head fails it
    slow = _zk(side22={"head": ["SLOW", "FAST"], "p2a": ["SLOW", "FAST"], "fast_fast_lost": False})
    t = RO.r_tracking(slow)
    assert t["status"] == "FAIL" and any("side22 absolute" in f for f in t["fails"])
    assert RO.r_tracking(_zk(side22={"fast_fast_lost": False}))["status"] == "FAIL"     # no head class on record


# ============================================================================== synthetic stamps
SHA_A = "a" * 64
SHA_B = "b" * 64


def _stamp(lr: Path, name: str, row: dict, es: int, sha: str, eps: list, *, S, env=None, t="2026-11-11_00-00-00",
           code=("c" * 64, "e" * 64), args=None):
    st = lr / name / row["scenario"] / f"seed{es:03d}" / t
    (st / "raw").mkdir(parents=True)
    env = dict(S.expected_env(row) if env is None else env)
    episodes = []
    for e in eps:
        tr = np.asarray(e["track"], np.float32)
        np.savez(st / "raw" / f"episode_{e['i']:02d}.npz", root_pos_w=tr,
                 terrain_z=np.full((len(tr), 6), e.get("pad", 0.0), np.float32))
        episodes.append({"env_index": e["i"], "n_steps": len(tr), "termination_reason": e["term"]})
    (st / "run_meta.json").write_text(json.dumps({
        "checkpoint_sha256": sha, "scenario_id": row["scenario"], "env_overrides": {"seed": es}, "role": "teacher",
        "task": S.TEACHER_TASK, "env_vars": env, "num_envs": len(eps), "checkpoint": f"/x/{name}.pt"}))
    (st / "scenario_metrics.json").write_text(json.dumps({"episodes": episodes, "aggregate": {
        "n_episodes": len(eps), "schedule_completion_rate": float(np.mean([e["term"] == "schedule_complete"
                                                                           for e in eps]))}}))
    (st / "summary.md").write_text("x\n")
    (st / "code_state.json").write_text(json.dumps({"code_sha": code[0], "eval_sha": code[1], "dirty": "False"}))
    (st / "tp_row.json").write_text(json.dumps({"rc": 0, "env_seed": es, "stripped_keys": [],
                                                "args": list(row["args"] if args is None else args)}))
    return st


def _eps(spawn=(5.0, -5.0), reach=(3.5, 3.0, 2.0, 4.0), terms=("schedule_complete", "schedule_complete",
                                                                  "schedule_complete", "fall"), pad=0.0):
    out = []
    for i, (r, t) in enumerate(zip(reach, terms)):
        x0, y0 = spawn[0] + 10 * i, spawn[1]
        tr = [[x0, y0, 1.085 + pad], [x0 + r * 0.5, y0, 1.3], [x0 + r, y0 + 0.0, 0.2], [x0 + 0.2, y0, 1.0]]
        out.append({"i": i, "track": tr, "term": t, "pad": pad})
    return out


def test_rows_are_keyed_by_sha256_and_composition_not_by_name(M, tmp_path):
    S, RO = M.S, M.RO
    doc = S.load_rows()
    row = next(r for r in doc["kind_rows"] if r["id"] == "rt_ledge_up_b1__omni")
    lr = tmp_path / "lr"
    _stamp(lr, f"RT_{SHA_A[:8]}_decoy", row, 1, SHA_B, _eps(), S=S)              # A's name, B's checkpoint
    good = _stamp(lr, "anything", row, 1, SHA_A, _eps(), S=S)
    bad_env = dict(S.expected_env(row), KRABBY_SITE_DIFF="0.3:0.6")
    _stamp(lr, "wrong_band", row, 1, SHA_A, _eps(), S=S, env=bad_env, t="2026-11-10_00-00-00")
    R = RO.Rows([lr], doc)
    f = R.found(row["id"], 1, SHA_A)
    assert f["status"] == "OK" and Path(f["stamp"]) == good and not f["duplicates"]
    assert R.found(row["id"], 1, SHA_B)["stamp"].split("/lr/")[1].startswith(f"RT_{SHA_A[:8]}_decoy")
    assert R.found(row["id"], 2, SHA_A)["status"] == "MISSING"
    later = _stamp(lr, "relaunch", row, 1, SHA_A, _eps(), S=S, t="2026-11-12_00-00-00")
    R = RO.Rows([lr], doc)
    f = R.found(row["id"], 1, SHA_A)
    assert Path(f["stamp"]) == good and f["duplicates"] == [str(later)]
    assert Path(RO.Rows([lr], doc, dup="latest").found(row["id"], 1, SHA_A)["stamp"]) == later
    assert RO.Rows([lr], doc, dup="error").found(row["id"], 1, SHA_A)["status"] == "BAD"


def test_gravel_rows_need_the_friction_contract_in_the_eval_env(M, tmp_path):
    S, RO = M.S, M.RO
    doc = S.load_rows()
    row = next(r for r in doc["kind_rows"] if r["id"] == "rt_gravel_b2__omni")
    assert S.expected_env(row)["KRABBY_RT_FRICTION"] == "gravel:0.2:0.5"
    stripped = {k: v for k, v in S.expected_env(row).items() if k != "KRABBY_RT_FRICTION"}
    lr = tmp_path / "lr"
    _stamp(lr, "stripped", row, 1, SHA_A, _eps(), S=S, env=stripped)
    assert RO.Rows([lr], doc).found(row["id"], 1, SHA_A)["status"] == "MISSING"
    dr = dict(S.expected_env(row), KRABBY_DR_ROD_SPEED="0.6:1.35:12000")
    _stamp(lr, "dr", row, 2, SHA_A, _eps(), S=S, env=dr)
    assert RO.Rows([lr], doc).found(row["id"], 2, SHA_A)["status"] == "MISSING"


def test_kind_row_statistics_and_the_paired_g_line(M, tmp_path, monkeypatch):
    S, RO = M.S, M.RO
    monkeypatch.setattr(RO, "KIND_ROW_EPISODES", 4)                              # the synthetic stamps hold 4 episodes
    doc = S.load_rows()
    row = next(r for r in doc["kind_rows"] if r["id"] == "rt_ledge_up_b2__omni")      # radius 3.2
    lr = tmp_path / "lr"
    for es in (1, 2):
        _stamp(lr, "new", row, es, SHA_A, _eps(reach=(3.5, 3.3, 3.25, 4.0)), S=S)
        _stamp(lr, "tdr", row, es, SHA_B, _eps(reach=(3.5, 3.0, 2.0, 4.0)), S=S)
    R = RO.Rows([lr], doc)
    rep = RO.row_report(R, row["id"], (1, 2), SHA_B)
    st = rep["stats"]
    assert st["completion"] == 0.75 and st["success"] == 0.25 and st["stuck"] == 0.5 and st["crossed"] == 0.5
    assert st["r_max_completers"]["max"] == pytest.approx(3.5, abs=1e-5)
    assert st["termination_reasons"] == {"fall": 2, "schedule_complete": 6}
    pt = RO.pair_tables(R, row["id"], (1, 2), SHA_A, SHA_B, "success")
    assert pt["status"] == "OK" and len(pt["a"]) == 8 and RO.share(pt["a"]) == 0.75 and RO.share(pt["b"]) == 0.25
    ln = RO.g_line(pt["a"], pt["b"])
    assert ln["diff"] == pytest.approx(0.5)


def test_pairing_refuses_moved_spawns_and_mixed_builds(M, tmp_path, monkeypatch):
    S, RO = M.S, M.RO
    monkeypatch.setattr(RO, "KIND_ROW_EPISODES", 4)
    doc = S.load_rows()
    row = next(r for r in doc["kind_rows"] if r["id"] == "rt_slopes_b1__omni")
    lr = tmp_path / "lr"
    _stamp(lr, "new", row, 1, SHA_A, _eps(spawn=(5.0, -5.0), pad=-0.1), S=S)
    _stamp(lr, "tdr", row, 1, SHA_B, _eps(spawn=(5.0, -4.0), pad=0.1), S=S)
    R = RO.Rows([lr], doc)
    moved = RO.pair_tables(R, row["id"], (1,), SHA_A, SHA_B, "success")
    assert moved["status"] == "INVALID" and "spawn apart" in moved["why"][0]
    rep = RO.row_report(R, row["id"], (1,), SHA_A)
    assert rep["stats"]["slopes_variant"]["up_sunk_pad"] == 4
    lr2 = tmp_path / "lr2"
    _stamp(lr2, "new", row, 1, SHA_A, _eps(), S=S, code=("1" * 64, "e" * 64))
    _stamp(lr2, "tdr", row, 1, SHA_B, _eps(), S=S)
    pt = RO.pair_tables(RO.Rows([lr2], doc), row["id"], (1,), SHA_A, SHA_B, "success")
    assert pt["status"] == "INVALID" and "builds" in pt["why"][0]


# ============================================================================== make_thresholds (STOP 0)
def _baseline(M, succ: dict) -> dict:
    S = M.S
    kinds = {}
    for k in S.KINDS:
        for b in ("b1", "b2", "b3"):
            v = succ[k][b]
            kinds.setdefault(k, {})[b] = {
                "status": "OK", "stamps": [{"stamp": f"/x/{k}/{b}"}],
                "stats": {"n": 100, "completion": min(1.0, v + 0.1), "success": v, "stuck": 0.1, "crossed": v,
                          "r_max_all": {"p50": 4.0}, "r_max_completers": {"n": 90}, "termination_reasons": {}}}
    return {"kind_env_seeds": [3], "tdr_sha256": "t" * 64, "kinds": kinds, "retention": {}}


def test_make_thresholds_step7_rule(M):
    MT, S = M.MT, M.S
    succ = {k: {"b1": 0.95, "b2": 0.95, "b3": 0.95} for k in S.KINDS}
    succ["slopes"] = {"b1": 0.90, "b2": 0.79, "b3": 0.30}          # gated at b2 (lowest <= 0.80)
    succ["pits"] = {"b1": 0.85, "b2": 0.81, "b3": 0.05}            # b3 above the envelope: solved, R on b2
    succ["ledge_up"] = {"b1": 0.70, "b2": 0.20, "b3": 0.0}         # gated at b1
    doc = MT.freeze(_baseline(M, succ))
    assert doc["kinds"]["slopes"]["status"] == "gated" and doc["kinds"]["slopes"]["gate_band"] == "b2"
    assert doc["kinds"]["slopes"]["borderline"]                     # 0.79 in 0.74-0.86 is recorded
    assert doc["kinds"]["pits"]["status"] == "solved" and doc["kinds"]["pits"]["r_band"] == "b2"
    assert doc["kinds"]["pits"]["above_envelope_bands"] == ["b3"]
    assert doc["kinds"]["hills"]["status"] == "solved" and doc["kinds"]["hills"]["r_band"] == "b3"
    assert doc["kinds"]["ledge_up"]["gate_band"] == "b1"
    assert doc["gated"] == ["slopes", "ledge_up"] and doc["decision"] == "proceed"
    assert doc["rule"]["k"] == 2 and doc["rule"]["gate_env_seeds"] == [1, 2]
    one = {k: {"b1": 0.95, "b2": 0.95, "b3": 0.95} for k in S.KINDS}
    one["gravel"]["b3"] = 0.80                                      # <= 0.80 exactly: gated
    d1 = MT.freeze(_baseline(M, one))
    assert d1["gated"] == ["gravel"] and d1["decision"] == "stop_and_ask"
    none = {k: {"b1": 0.95, "b2": 0.95, "b3": 0.0 if k in ("ledge_up", "ledge_down", "pits") else 0.9}
            for k in S.KINDS}
    assert MT.freeze(_baseline(M, none))["decision"] == "close"     # above-envelope failures do not gate
    bad = _baseline(M, succ)
    bad["kinds"]["rubble"]["b2"]["status"] = "INCOMPLETE"
    with pytest.raises(ValueError, match="incomplete"):
        MT.freeze(bad)
    with pytest.raises(ValueError, match="es3"):
        MT.freeze({**_baseline(M, succ), "kind_env_seeds": [1]})


def _sources(M) -> dict:
    return {"sha256": {n: M.RO.sha256_file(SCRIPTS / n) for n in M.RO.FROZEN_SCRIPTS}}


def test_gates_end_to_end_on_synthetic_rows(M, tmp_path, monkeypatch):
    """G on two gated kinds + R completion rows + a solved kind's R band, the tracking read stubbed (its inputs are
    suite / side22 stamps; zero_kg_check itself is the repo's and is exercised by the recorded-G1 test)."""
    S, RO, MT = M.S, M.RO, M.MT
    monkeypatch.setattr(RO, "KIND_ROW_EPISODES", 4)
    succ = {k: {"b1": 0.95, "b2": 0.95, "b3": 0.95} for k in S.KINDS}
    succ["slopes"]["b1"] = 0.5
    succ["pits"]["b2"] = 0.6
    thr = MT.freeze({**_baseline(M, succ), "tdr_sha256": SHA_B}, sources=_sources(M))
    assert thr["gated"] == ["slopes", "pits"] and thr["kinds"]["hills"]["r_band"] == "b3"
    doc = S.load_rows()
    rows = {r["id"]: r for r in doc["kind_rows"] + doc["retention_rows"]}
    lr = tmp_path / "lr"
    win, lose = (3.5, 3.5, 3.5, 3.5), (3.5, 2.0, 2.0, 2.0)          # crossed by all vs one of four
    allc = ("schedule_complete",) * 4
    for es in (1, 2):
        for rid in ("rt_slopes_b1__omni", "rt_pits_b2__omni"):
            _stamp(lr, "new", rows[rid], es, SHA_A, _eps(reach=win, terms=allc), S=S)
            _stamp(lr, "tdr", rows[rid], es, SHA_B, _eps(reach=lose, terms=allc), S=S)
        for kind in thr["solved"]:
            rid = S.row_id(kind, thr["kinds"][kind]["r_band"])
            _stamp(lr, "new", rows[rid], es, SHA_A, _eps(reach=win, terms=allc), S=S)
            _stamp(lr, "tdr", rows[rid], es, SHA_B, _eps(reach=win, terms=allc), S=S)
    for rid in S.COMPLETION_R:
        for es in rows[rid]["env_seeds"]:
            for who, sha in (("new", SHA_A), ("tdr", SHA_B)):
                _stamp(lr, f"{who}_{rid}", rows[rid], es, sha, _eps(terms=allc), S=S)
    trk = {"status": "OK", "rows": [{"statistic": "track_err fwd", "row": "suite es1", "watch": False,
                                     "fails": False}],
           "side22": {"head": ["FAST", "FAST"], "p2a": ["FAST", "FAST"], "fast_fast_lost": False}, "fails": [],
           "translate_statistic": "track_err", "thresholds_version": "v1.2"}
    monkeypatch.setattr(RO, "zero_kg", lambda *a, **k: dict(trk))
    R = RO.Rows([lr], doc)
    out = RO.gates(R, SHA_A, SHA_B, thr)
    assert out["G"]["status"] == "PASS" and out["G"]["gain_kinds"] == ["pits", "slopes"], out["G"]
    assert out["G"]["kinds"]["slopes"]["diff"] == pytest.approx(0.75)
    assert all(v["pass"] for v in out["R"]["completion"].values())
    assert sorted(out["R"]["solved"]) == sorted(thr["solved"]) and all(v["pass"] for v in out["R"]["solved"].values())
    assert out["R"]["status"] == "PASS" and out["verdict"]["verdict"] == "ACCEPT"
    trk["rows"][0]["fails"] = True
    out = RO.gates(R, SHA_A, SHA_B, thr)
    assert out["R"]["tracking"]["status"] == "FAIL" and out["verdict"]["verdict"] == "HALT_R"
    assert out["R"]["tracking"]["drift_reading"]["its_past_p2a"] == 6000
    trk["rows"][0]["fails"] = False
    # the rule is read from the FROZEN file, not from the module constants
    tight = json.loads(json.dumps(thr))
    tight["rule"]["G"]["gain_floor"] = 0.9
    assert RO.gates(R, SHA_A, SHA_B, tight)["G"]["status"] == "FAIL"                  # diff 0.75 < the frozen 0.9
    wide = json.loads(json.dumps(thr))
    wide["rule"]["R_completion"]["floor"] = 0.33
    assert all(v["margin"] >= 0.33 for v in RO.gates(R, SHA_A, SHA_B, wide)["R"]["completion"].values())
    gone = json.loads(json.dumps(thr))
    del gone["rule"]["G"]["ci_lb"]
    with pytest.raises(ValueError, match="rule.G lacks"):
        RO.gates(R, SHA_A, SHA_B, gone)
    # ... and the recorded script sha256s must be the current scripts' (the freeze enforces itself)
    moved = json.loads(json.dumps(thr))
    moved["sources"]["sha256"]["readout.py"] = "0" * 64
    with pytest.raises(ValueError, match="script sha256s differ.*readout.py"):
        RO.gates(R, SHA_A, SHA_B, moved)
    with pytest.raises(ValueError, match="script sha256s differ"):
        RO.gates(R, SHA_A, SHA_B, {**thr, "sources": {}})
    out = RO.gates(R, SHA_B, SHA_A, {**thr, "tdr_sha256": SHA_A})    # roles swapped: both gated kinds regress
    assert out["G"]["regressed"] == ["pits", "slopes"] and out["verdict"]["verdict"] == "NOT_ACCEPTED"


def test_kind_rows_must_hold_100_episodes_per_seed_on_both_sides(M, tmp_path):
    """Fail closed: 4-episode stamps (a short-envs relaunch) are INVALID at the real n == 100, never read short."""
    S, RO = M.S, M.RO
    assert RO.KIND_ROW_EPISODES == 100
    doc = S.load_rows()
    row = next(r for r in doc["kind_rows"] if r["id"] == "rt_pits_b1__omni")
    lr = tmp_path / "lr"
    _stamp(lr, "new", row, 1, SHA_A, _eps(), S=S)
    _stamp(lr, "tdr", row, 1, SHA_B, _eps(), S=S)
    pt = RO.pair_tables(RO.Rows([lr], doc), row["id"], (1,), SHA_A, SHA_B, "success")
    assert pt["status"] == "INVALID" and any("4 episodes (want 100)" in w for w in pt["why"]), pt
    ret = next(r for r in doc["retention_rows"] if r["id"] == "R_field_light")          # not a kind row: no n rule
    for es in (1, 2):
        _stamp(lr, "new_r", ret, es, SHA_A, _eps(), S=S)
        _stamp(lr, "tdr_r", ret, es, SHA_B, _eps(), S=S)
    assert RO.pair_tables(RO.Rows([lr], doc), "R_field_light", (1, 2), SHA_A, SHA_B, "completion")["status"] == "OK"


def test_a_failed_launch_never_shadows_an_ok_relaunch(M, tmp_path):
    S, RO = M.S, M.RO
    doc = S.load_rows()
    row = next(r for r in doc["kind_rows"] if r["id"] == "rt_hills_b2__omni")
    lr = tmp_path / "lr"
    first = _stamp(lr, "launch1", row, 1, SHA_A, _eps(), S=S, t="2026-11-11_00-00-00")
    tp = json.loads((first / "tp_row.json").read_text())
    (first / "tp_row.json").write_text(json.dumps({**tp, "rc": 1}))                    # failed, still wrote tp_row.json
    f = RO.Rows([lr], doc).found(row["id"], 1, SHA_A)
    assert f["status"] == "BAD" and any("rc 1" in p for p in f["problems"])           # alone: read BAD, not hidden
    good = _stamp(lr, "launch2", row, 1, SHA_A, _eps(), S=S, t="2026-11-12_00-00-00")
    for dup in ("earliest", "latest", "error"):
        f = RO.Rows([lr], doc, dup=dup).found(row["id"], 1, SHA_A)
        assert f["status"] == "OK" and Path(f["stamp"]) == good and not f["duplicates"], dup
        assert any("rc != 0" in n for n in f["notes"])
    later_ok = _stamp(lr, "launch3", row, 1, SHA_A, _eps(), S=S, t="2026-11-13_00-00-00")
    f = RO.Rows([lr], doc).found(row["id"], 1, SHA_A)                                  # two OK: the dup rule applies
    assert Path(f["stamp"]) == good and f["duplicates"] == [str(later_ok)]


def _site_stamp(lr: Path, S, sha: str, *, reach: float, cross_r: float = 3.2):
    """A SITE9_tstar-shaped stamp (``<lr>/SITE9_tstar/site_x/seed001/<t>``, raw + site_rec/layers.npz); each
    episode's tile centre is its own spawn (x0 = 5 + 10 i, y0 = -5)."""
    row = next(r for r in S.load_rows()["kind_rows"] if r["id"] == "rt_pits_b1__omni")
    st = _stamp(lr, "SITE9_tstar", {**row, "scenario": "site_x"}, 1, sha, _eps(reach=(reach,) * 4), S=S, env={})
    (st / "site_rec").mkdir()
    origins = np.array([[5.0 + 10 * i, -5.0, 0.0] for i in range(4)])
    np.savez(st / "site_rec" / "layers.npz", terrain_origins=origins, tile_cross_r=np.full(4, cross_r))
    return st


def test_site_frame_crossed_uses_the_campaigns_own_spawn_path(M, tmp_path):
    S, RO = M.S, M.RO
    st = _site_stamp(tmp_path / "lr", S, SHA_A, reach=3.5)
    cro, off, cro_spawn = RO.crossed_tile_frame(st)
    own = {i: RO.crossed_from({i: v}, 3.2)[i] for i, v in RO.r_max_of(st).items()}     # r_max_of + crossed_from
    assert cro_spawn == own == cro and all(cro.values()) and max(off.values()) < 1e-6
    st2 = _site_stamp(tmp_path / "lr2", S, SHA_A, reach=3.0)
    assert not any(RO.crossed_tile_frame(st2)[2].values())


def test_validate_site_skips_a_stamp_that_is_not_tstars_and_reports_it(M, tmp_path):
    S, RO = M.S, M.RO
    lr = tmp_path / "lr"
    _site_stamp(lr, S, SHA_A, reach=3.5)                                               # dir SITE9_tstar, not T*'s sha256
    camp = tmp_path / "camp"
    (camp / "scripts").mkdir(parents=True)
    (camp / "scripts" / "readout.py").write_text("def crossed_of(a):\n    return {}\ndef success_of(a):\n    return {}\n")
    res = RO.validate_site(lr, "SITE9_tstar", camp)
    assert res["rows_read"] == 0 and not res["pass"]
    assert len(res["skipped_wrong_sha256"]) == 1 and "!= T*" in res["skipped_wrong_sha256"][0]


def test_gates_refuse_a_closed_or_foreign_thresholds_file(M, tmp_path):
    RO = M.RO
    R = RO.Rows([tmp_path / "none"])
    out = RO.gates(R, SHA_A, SHA_B, {"decision": "close"})
    assert out["verdict"]["verdict"] == "NO_GATE_READ"
    with pytest.raises(ValueError, match="frozen on TDR"):
        RO.gates(R, SHA_A, SHA_B, {"decision": "proceed", "tdr_sha256": "c" * 64})


# ============================================================================== recorded stamps (this machine only)
_RSD_LR = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/rod_speed_dr/lr")


@pytest.mark.skipif(not (_RSD_LR / "RSDT_03c3e472_T00_nominal_L000_A").is_dir(),
                    reason="TDR's recorded rod-speed G1 stamps are not on this machine")
def test_readout_reproduces_tdrs_recorded_g1_numbers(M):
    res = M.RO.validate_g1()
    assert res["pass"], res["differences"][:10]
    assert res["numbers_compared"] > 400 and res["R_verdict"] == "PASS"


@pytest.mark.skipif(not (_RSD_LR / "RSDSELT_03c3e472_L000").is_dir(),
                    reason="TDR's recorded selection rows are not on this machine")
def test_radius_calibration_on_tdrs_band_free_rows(M):
    res = M.RO.calibrate()
    assert res["pass"] and min(res["min_share"].values()) >= 0.95
