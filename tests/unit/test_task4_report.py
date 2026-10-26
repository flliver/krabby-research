"""M18 Task 4 step 11: the baseline-vs-final report (``TP/report.py``), CPU only.

Pinned:
* never invent a number: an absent source is ``pending`` (a step-11 row: ``pending (row <id>)``; a run record:
  ``pending step 11``) or ``N/A`` WITH its reason -- never 0 / a filled cell without a value; an unreadable source
  never reads as filled; a SKIPPED / failed step-11 row reads N/A with the queue's reason;
* the step-11 rows are located by NAME / SCENARIO / env seed (the newest COMPLETE stamp: a completion marker);
* the derived numbers (omni vy / wz RMS over the own hold x steady_mask, alive steps only; the forward actual / cmd
  ratio) and the labels (sim plant on Task 3, derived (step 11), the 1 N misread caveat);
* the 45-kg ladder: max_stable is never claimed over an unread level;
* the measured wall-clock: stage segments from the queue sentinels, PARKED -> START a park (0 GPU-h);
* HALT chance rates: modelled -> the rate; unmodelled / degenerate -> "rate unknown"; driver / guard -> n/a;
* the README generator names only scripts / flags that exist; report.py carries no dated path;
* the .md never cuts a number: a long table cell is clipped only at a clause boundary (never inside a token) and
  its full text is printed under the table; the cell index and the JSON keep the full strings;
* on the real records (skipped when absent): the F19 sources, the T* record, the wall-clock port, the keep-set guard.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

yaml = pytest.importorskip("yaml")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
TP = OMNI / "experiments/tools/pipeline"
PL = OMNI / "experiments/2026-10-16_task4_pipeline"
SPEC = PL / "pipeline.yaml"
HAVE_F19 = (PL / "F19" / "pipeline_state.json").is_file()
needs_f19 = pytest.mark.skipif(not HAVE_F19, reason="the F19 run records are not on this machine")


def _needs_log(*logs: Path):
    """The queue logs (``*.log``) are gitignored -- local records, absent on any other checkout (CI): skip, naming the
    missing ones, before the assertions that read them."""
    missing = [str(p.relative_to(PL)) for p in logs if not p.is_file()]
    if missing:
        pytest.skip(f"the gitignored queue log(s) {missing} are not on this machine")


def _c7_failed_logs() -> list[Path]:
    return sorted((PL / "preflight" / "stepC" / "dryspec").glob("dry.c7failed_*/pipeline_queue.log")) or \
        [PL / "preflight" / "stepC" / "dryspec" / "dry.c7failed_*" / "pipeline_queue.log"]


def _load(path: Path, name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = old
    return mod


@pytest.fixture(scope="module")
def R():
    _load(TP / "pipeline.py", "tp_pipeline")
    return _load(TP / "report.py", "tp_report")


# ============================================================================== fixture stamps
HOLDS = ["settle", "stand", "fwd"]
CATS = {"stand": "stand", "fwd": "fwd"}


def mk_omni_stamp(d: Path, *, n_per=4, vy_err=0.1, wz_err=0.2, fall_env=None, duty_missing=(), marker=True,
                  cmd=(0.25, 0.0, 0.0)) -> Path:
    """A synthetic omni suite stamp: run_meta / scenario_metrics / metrics / raw (+ tp_row.json)."""
    d.mkdir(parents=True)
    (d / "metrics").mkdir()
    (d / "raw").mkdir()
    emap, eps = [], []
    i = 0
    for lab in ("stand", "fwd"):
        for _ in range(n_per):
            emap.append({"env_index": i, "env": i, "label": lab, "category": CATS[lab], "hand": None, "wave": 0})
            T = 200
            seg = np.full(T, HOLDS.index(lab), np.int16)
            seg[:20], seg[20:50] = -1, 0
            steady = seg == HOLDS.index(lab)
            c = np.zeros((T, 3), np.float32)
            if lab == "fwd":
                c[50:] = cmd
            vb = c.copy()
            vb[:, 1] += vy_err
            wb = np.zeros((T, 3), np.float32)
            wb[:, 2] = c[:, 2] + wz_err
            done = np.zeros(T, bool)
            if fall_env == i:
                done[150] = True
                vb[150:, 1] += 50.0          # garbage after the fall: must not enter the RMS
            np.savez(d / "raw" / f"episode_{i:02d}.npz", segment_id=seg, steady_mask=steady, cmd_applied=c,
                     root_lin_vel_b=vb, root_ang_vel_b=wb, done=done,
                     root_pos_w=np.tile([0.0, 0.0, 1.1], (T, 1)).astype(np.float32),
                     terrain_z=np.zeros((T, 6), np.float32))
            hold = {"label": lab, "tripod": {} if i in duty_missing else {"duty_factor": {f"F{k}": 0.5 for k in range(6)}},
                    "air_time": {"pooled": {"p50": 0.3}, "tippy_tap_fraction": 0.4}}
            ep = {"env_index": i, "category": CATS[lab], "hold": lab, "holds": {lab: hold},
                  "tippy_tap_fraction": 0.45, "actions": {"action_rate_mean_all": 0.2, "action_sign_reversals_total": 100},
                  "orientation": {"roll_rms": 0.01, "pitch_rms": 0.02},
                  "slip_episode": {"per_foot": {f"F{k}": {"duty_factor": 0.6} for k in range(6)}},
                  "termination_reason": "fall" if fall_env == i else "schedule_complete", "n_steps": T}
            (d / "metrics" / f"episode_{i:02d}.json").write_text(json.dumps(ep))
            eps.append(ep)
            i += 1
    n = n_per
    by_cat = {c: {"ratio": {"median": 1.02 if c == "fwd" else None}, "steps_per_m": {"median": 10.0 if c == "fwd" else None},
                  "tripod": {"median": 0.7}} for c in ("stand", "fwd")}
    agg = {"n_episodes": len(eps), "schedule_completion_rate": 1.0,
           "orientation": {"roll_rms": {"median": 0.01}, "pitch_rms": {"median": 0.02}},
           "suite": {"by_category": by_cat, "n_by_hold": {"stand": n, "fwd": n},
                     "survival_by_hold": {"stand": 1.0, "fwd": (n - (1 if fall_env is not None else 0)) / n},
                     "category_of_hold": {"stand": "stand", "fwd": "fwd"}, "hand_of_hold": {"stand": None, "fwd": None},
                     "hold_of_wave": ["stand", "fwd"], "waves": 1, "episodes_total": len(eps),
                     "trained_categories": ["stand", "fwd"]}}
    (d / "scenario_metrics.json").write_text(json.dumps({"aggregate": agg, "episodes": eps}))
    (d / "run_meta.json").write_text(json.dumps({"hold_labels": HOLDS, "env_index_map": emap, "dt": 0.02,
                                                  "trained_categories": ["stand", "fwd"]}))
    (d / "summary.md").write_text("x\n")
    if marker:
        (d / "tp_row.json").write_text("{}")
    return d


# ============================================================================== pure helpers
def test_fmt_never_turns_a_missing_value_into_a_number(R):
    assert R.fmt(None) == "N/A" and R.fmt(None) != "0"
    assert R.fmt(0.0) == "0.000" and R.fmt(0) == "0"
    assert R.pair(None, 0.5) == "N/A / 0.500"
    assert R.med([None, None]) is None and R.med([None, 1.0, 3.0]) == 2.0


def test_newest_complete_stamp_by_name_scenario_seed(R, tmp_path):
    base = tmp_path / "lr" / "T4S11_tstar" / "suite__omni_body" / "seed001"
    old = mk_omni_stamp(base / "2026-10-24_01-00-00")
    mk_omni_stamp(base / "2026-10-24_02-00-00", marker=False)      # still running: no completion marker
    assert R.newest_stamp(tmp_path / "lr", "T4S11_tstar", "suite__omni_body", 1) == old
    assert R.newest_stamp(tmp_path / "lr", "T4S11_tstar", "suite__omni_body", 2) is None
    assert R.newest_stamp(tmp_path / "lr", "nope", "suite__omni_body", 1) is None


def test_step11_row_ids_map_to_the_queue_names(R):
    assert R.S11_ROWS["ST_suite_body_1"] == ("T4S11_c28000", "suite__omni_body", 1)
    assert R.S11_ROWS["T1_dirsweep_body_2"] == ("T4S11_tstar", "dirsweep__omni_body", 2)
    assert R.S11_ROWS["ST_L135_fl_2"] == ("T4S11_c28000_L135", "field_light__omni", 2)
    assert R.S11_ROWS["F_step_pin"] == ("T4S11_fwd3a_pin", "step__A15pB", 1)
    assert R.S11_ROWS["F_m270"] == ("T4S11_fwd3a_m270", "slow__A15pB", 1)


def test_rows_log_parses_row_done_skip(R, tmp_path):
    lg = tmp_path / "q.log"
    lg.write_text("S11 START 2026-10-24T01:00:00-04:00\n"
                  "S11 ROW id=ST_suite_body_1 2026-10-24T01:00:00-04:00\n"
                  "S11 DONE id=ST_suite_body_1 rc=0 2026-10-24T01:30:00-04:00\n"
                  "S11 ROW id=T1_grid_body_1 2026-10-24T01:30:00-04:00\n"
                  "S11 DONE id=T1_grid_body_1 rc=8 2026-10-24T02:00:00-04:00\n"
                  "S11 SKIP id=ST_grid_body_1 why=ST_suite_body_1_failed 2026-10-24T02:00:00-04:00\n")
    r = R.rows_log(lg)
    assert r["rows"]["ST_suite_body_1"] == {"state": "done", "start": "2026-10-24T01:00:00-04:00", "rc": 0,
                                            "end": "2026-10-24T01:30:00-04:00"}
    assert r["rows"]["T1_grid_body_1"]["rc"] == 8
    assert r["rows"]["ST_grid_body_1"]["state"] == "skip" and r["rows"]["ST_grid_body_1"]["why"] == "ST_suite_body_1_failed"
    assert abs(r["measured_h"] - 1.0) < 1e-9
    assert R.rows_log(tmp_path / "absent.log")["rows"] == {}


def test_wallclock_separates_parks(R, tmp_path):
    lg = tmp_path / "pipeline_queue.log"
    lg.write_text("\n".join([
        "T4PIPE START run=F19 pid=1 2026-10-21T10:00:00",
        "T4PIPE STAGE run=F19 stage=s1 kind=driver 2026-10-21T10:00:00",
        "T4PIPE LAUNCH run=F19 stage=s1 chain=x 2026-10-21T10:00:00",
        "T4PIPE ROW run=F19 stage=s1 row=suite@1 2026-10-21T11:00:00",
        "T4PIPE GATE run=F19 stage=s1 verdict=PASS 2026-10-21T11:30:00",
        "T4PIPE STAGE run=F19 stage=p3a kind=driver 2026-10-21T11:30:00",
        "T4PIPE HALT run=F19 stage=p3a class=FAIL 2026-10-21T12:00:00",
        "T4PIPE PARKED run=F19 open_halts=p3a 2026-10-21T12:00:00",
        "T4PIPE START run=F19 pid=2 2026-10-21T15:00:00",
        "T4PIPE STAGE run=F19 stage=p3b kind=driver 2026-10-21T15:00:00",
        "T4PIPE GATE run=F19 stage=p3b verdict=CEILING 2026-10-21T16:00:00",
        "T4PIPE END run=F19 2026-10-21T16:00:00", ""]))
    w = R.wallclock(lg)
    assert abs(w["stages"]["s1"]["active_h"] - 1.5) < 1e-9
    assert abs(w["stages"]["s1"]["train_to_first_row_h"] - 1.0) < 1e-9
    assert abs(w["stages"]["p3a"]["active_h"] - 0.5) < 1e-9
    assert abs(w["stages"]["p3b"]["active_h"] - 1.0) < 1e-9
    assert len(w["parks"]) == 1 and abs(w["parks"][0]["h"] - 3.0) < 1e-9
    assert abs(w["total_active_h"] - 3.0) < 1e-9 and abs(w["span_h"] - 6.0) < 1e-9


def test_curve_ladder_never_claims_over_an_unread_level(R):
    def reads(d):
        return {lv: ({"stable": s} if s is not None else {"status": "pending", "why": "row"}) for lv, s in d.items()}

    s = R.curve_summary(reads({0: True, 45: None, 90: True}))
    assert s["max_stable"] is None and s["open_from"] == 45 and s["stable_through"] == 0
    assert s["pending_levels"] == [45]
    s = R.curve_summary(reads({0: True, 45: True, 90: True, 135: True, 180: False}))
    assert s["max_stable"] == 135 and s["first_unstable"] == 180 and s["open_from"] is None
    s = R.curve_summary(reads({0: True, 45: True, 90: False}))
    assert s["max_stable"] == 45 and s["first_unstable"] == 90
    s = R.curve_summary(reads({lv: True for lv in R.LEVELS}))
    assert s["max_stable"] == 270 and s["first_unstable"] is None
    s = R.curve_summary(reads({0: False}))
    assert s["max_stable"] is None and s["first_unstable"] == 0 and s["open_from"] is None


def test_status_of_pending_dominates_and_none_is_pending(R):
    ok, pend = R._src("/x"), R._src(status="pending", row="ST_grid_body_1")
    na = R._src(status="na", why="not obtainable: y")
    assert R.Report.status_of([ok, ok])[0] == "filled"
    st, why, rows = R.Report.status_of([ok, pend])
    assert st == "pending" and "pending (row ST_grid_body_1)" in why and rows == ["ST_grid_body_1"]
    assert R.Report.status_of([na, na])[:2] == ("na", "not obtainable: y")
    st, why, _ = R.Report.status_of([None, ok])
    assert st == "pending" and R.PENDING in why
    assert R.Report.status_of([])[0] == "pending"


def test_an_unreadable_source_never_reads_filled(R):
    rep = R.Report.__new__(R.Report)
    rep.cells, rep.errors = [], []
    rep.cell("x", "S", "m", "T0", R.FILLED, value={1: {"unreadable": "boom"}, 2: {"a": 1.0}})
    c = rep.cells[0]
    assert c["status"] == R.NOTAPP and "boom" in c["why"]


# ============================================================================== readers on fixture stamps
def test_vel_rms_own_hold_steady_alive_only(R, tmp_path):
    st = mk_omni_stamp(tmp_path / "s", vy_err=0.1, wz_err=0.2, fall_env=5)
    r = R.read_vel_rms(st)
    assert set(r) == {"stand", "fwd"}
    assert abs(r["fwd"]["vy_rms"] - 0.1) < 1e-6 and abs(r["fwd"]["wz_rms"] - 0.2) < 1e-6
    assert abs(r["stand"]["vy_rms"] - 0.1) < 1e-6
    assert r["fwd"]["n"] == 4           # the fallen episode still has a 100-step alive own-hold window


def test_episode_extras_duty_is_none_on_discarded_windows(R, tmp_path):
    st = mk_omni_stamp(tmp_path / "s", duty_missing=(4, 5, 6))
    r = R.read_episode_extras(st)
    assert r["fwd"]["n"] == 4 and r["fwd"]["n_duty_valid"] == 1 and r["fwd"]["duty_hold"] == 0.5
    assert r["stand"]["n_duty_valid"] == 4
    assert abs(r["fwd"]["duty_episode"] - 0.6) < 1e-9 and r["fwd"]["tippy_hold"] == 0.4
    r2 = R.read_episode_extras(mk_omni_stamp(tmp_path / "t", duty_missing=tuple(range(8))))
    assert r2["fwd"]["duty_hold"] is None      # never 0


def test_categories_and_derived_stride(R, tmp_path):
    st = mk_omni_stamp(tmp_path / "s", fall_env=5)
    c = R.read_categories(st)["by_category"]
    assert c["fwd"]["ratio"] == 1.02 and abs(c["fwd"]["stride_m"] - 0.6) < 1e-12
    assert c["stand"]["stride_m"] is None
    assert c["fwd"]["misses"] == 1 and c["fwd"]["survival"] == 0.75


def test_forward_ratio_is_derived_only_when_the_harness_wrote_none(R, tmp_path):
    d = tmp_path / "v1"
    (d / "metrics").mkdir(parents=True)
    (d / "scenario_metrics.json").write_text(json.dumps({"aggregate": {"tripod_score": {"median": 0.05},
                                                                       "schedule_completion_rate": 0.7}}))
    (d / "run_meta.json").write_text(json.dumps({"checkpoint": "model_21100.pt", "hold_labels": ["low"]}))
    (d / "metrics" / "episode_00.json").write_text(json.dumps({"holds": {"low": {"tracking": {
        "vx": {"actual_mean": 0.2, "cmd_mean": 0.25, "rms": 0.1}, "vy": {"rms": 0.05}, "wz": {"rms": 0.3}}}}}))
    (d / "metrics" / "episode_01.json").write_text(json.dumps({"holds": {"low": {"tracking": {
        "vx": {"actual_mean": 0.3, "cmd_mean": 0.25, "ratio": 1.2, "rms": 0.1}}}}}))
    r = R.read_forward(d)
    h = r["by_hold"]["low"]
    assert h["vx_ratio"] == 1.2 and abs(h["vx_ratio_derived"] - 0.8) < 1e-12
    assert r["tripod_score"] == 0.05 and r["slip_ratio"] is None


def test_sweep_pools_env_seed_medians_and_reads_circular_jumps(R, tmp_path):
    def st(name, ratios):
        d = tmp_path / name
        d.mkdir()
        bc = {f"d{i:03d}": {"ratio": {"median": r}, "tripod": {"median": 0.7}, "steps_per_m": {"median": 10.0}}
              for i, r in zip((0, 90, 180, 270), ratios)}
        (d / "scenario_metrics.json").write_text(json.dumps({"aggregate": {"suite": {"by_category": bc}}}))
        return d

    th = {"lines": [{"id": "p2a.watch.sweep.abs_dev.d000", "line": 0.05},
                    {"id": "p2a.watch.sweep.max_adjacent_jump.ratio", "line": 0.1}]}
    r = R.sweep_read({1: st("a", [1.0, 1.1, 1.0, 1.2]), 2: st("b", [1.1, 1.1, 1.0, 1.0])}, th)
    assert abs(r["bins"]["d000"]["ratio"] - 1.05) < 1e-12 and abs(r["bins"]["d000"]["abs_dev"] - 0.05) < 1e-12
    assert r["bins"]["d000"]["pass"] is True and r["bins"]["d090"]["line"] is None
    j = r["max_adjacent_jump"]["ratio"]
    assert abs(j["value"] - 0.1) < 1e-9 and j["pass"] is True     # incl. the circular d270 -> d000 jump
    assert r["watch"] is True


def test_grid_cells_flag_the_envelope_and_derive_the_errors(R, tmp_path):
    d = tmp_path / "g"
    (d / "metrics").mkdir(parents=True)
    cats = [{"name": "g_v0p22_fwd_w0", "kind": "translate", "holds": [{"label": "g_v0p22_fwd_w0", "vx": 0.22, "vy": 0.0, "wz": 0.0}]},
            {"name": "g_v0p3_fwd_w0p3", "kind": "move_turn", "holds": [{"label": "g_v0p3_fwd_w0p3", "vx": 0.3, "vy": 0.0, "wz": 0.3}]}]
    (d / "run_meta.json").write_text(json.dumps({"suite": {"categories": cats}, "env_index_map": []}))
    bc = {"g_v0p22_fwd_w0": {"achieved_speed": {"median": 0.24}, "net_yaw_rate": {"median": 0.004},
                             "heading_drift": {"median": 0.005}},
          "g_v0p3_fwd_w0p3": {"achieved_speed": {"median": 0.2}, "net_yaw_rate": {"median": 0.1}}}
    agg = {"suite": {"by_category": bc, "n_by_hold": {}, "survival_by_hold": {}, "category_of_hold": {},
                     "hand_of_hold": {}, "hold_of_wave": [], "waves": 1, "episodes_total": 0}}
    (d / "scenario_metrics.json").write_text(json.dumps({"aggregate": agg}))
    for i, hd in enumerate((0.01, 0.02, 0.03)):
        (d / "metrics" / f"episode_{i:02d}.json").write_text(json.dumps(
            {"category": "g_v0p22_fwd_w0", "hold": "g_v0p22_fwd_w0", "holds": {"g_v0p22_fwd_w0": {"body": {"heading_drift": hd}}}}))
    g = R.read_grid(d)
    a, b = g["cells"]["g_v0p22_fwd_w0"], g["cells"]["g_v0p3_fwd_w0p3"]
    assert a["in_envelope"] is True and b["in_envelope"] is False
    assert abs(a["abs_speed_err"] - 0.02) < 1e-12 and abs(b["abs_wz_err"] - 0.2) < 1e-12
    assert g["drift_pooled_in_envelope"]["n"] == 3 and abs(g["drift_pooled_in_envelope"]["median"] - 0.02) < 1e-12
    assert g["drift_pooled_in_envelope"]["label"] == R.DERIVED
    assert len(R.GRID_IN_ENVELOPE) == 9


# ============================================================================== the whole report on an empty run
@pytest.fixture()
def empty_spec(tmp_path):
    """The real spec in a temp campaign dir whose row roots are empty: every run record is absent."""
    d = tmp_path / "camp"
    d.mkdir()
    for n in ("thresholds_v1_2.json", "thresholds_v1_1.json"):
        if (PL / n).is_file():
            (d / n).write_text((PL / n).read_text())
    doc = yaml.safe_load(SPEC.read_text())
    doc["row_out"] = str(tmp_path / "out" / "task4")
    doc["stand"]["p2_lr"] = str(tmp_path / "out" / "phase2" / "lr")
    (d / "pipeline.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    return d / "pipeline.yaml", tmp_path / "out" / "task4"


def test_report_on_empty_records_invents_nothing(R, empty_spec, tmp_path):
    spec, row_out = empty_spec
    # one step-11 row present (es1 of T*'s body suite), es2 SKIPPED by the queue
    mk_omni_stamp(row_out / "step11" / "lr" / "T4S11_tstar" / "suite__omni_body" / "seed001" / "2026-10-24_03-00-00")
    lg = tmp_path / "s11.log"
    lg.write_text("S11 SKIP id=T1_grid_body_1 why=ST_suite_body_1_failed 2026-10-24T02:00:00-04:00\n"
                  "S11 SKIP id=T1_grid_body_2 why=ST_suite_body_1_failed 2026-10-24T02:00:00-04:00\n")
    doc = R.build(spec, "F19", None, lg)
    cells = {c["id"]: c for c in doc["cells"]}
    assert len(cells) == len(doc["cells"])                                  # ids unique
    assert sum(doc["counts"].values()) == len(doc["cells"]) == 67          # 72 fill-plan cells minus the 5 R18 ones
    for c in doc["cells"]:
        assert c["status"] in (R.FILLED, R.PEND, R.NOTAPP)
        if c["status"] != R.FILLED:
            assert c.get("why"), c["id"]                                    # every N/A / pending says why
        else:
            assert c.get("value") is not None, c["id"]                      # a filled cell carries its value
    # run records absent -> pending step 11, never a number
    assert cells["t0.cat.T1"]["status"] == R.PEND and R.PENDING in cells["t0.cat.T1"]["why"]
    assert cells["stand.T1.sh"]["status"] == R.PEND
    assert cells["st.pairing"]["status"] == R.PEND
    # the step-11 rows: es1 read, es2 pending by row id; the skipped grid reads N/A with the queue's reason
    sb = cells["t2.suite_body.T1"]
    assert sb["status"] == R.PEND and "pending (row T1_suite_body_2)" in sb["why"] and 1 in sb["value"]
    assert cells["t2.grid_body.T1"]["status"] == R.NOTAPP and "skipped" in cells["t2.grid_body.T1"]["why"]
    assert "pending (row ST_grid_body_1)" in cells["t2.grid_body.St"]["why"]
    # structural N/A cells
    for cid in ("t2.F.na", "t2.H.na", "t3.curve.H", "stand.na", "t2.field_suite.na", "t0.H.na"):
        assert cells[cid]["status"] == R.NOTAPP, cid
    # labels
    m = R.md(doc)
    assert "pending (row T1_suite_body_2)" in m and R.PENDING in m
    assert R.SIM in m and R.DERIVED in m and "GE/metrics.py:1271-1275" in m
    assert "## Disclosures" in m and "## Grant reconciliation" in m and "230.06" in m
    assert re.search(r"Cells:\*\* \d+ filled, \d+ pending, \d+ N/A", m)
    json.dumps(doc, default=str)                                            # the machine record serialises


def test_halt_chance_rates_unknown_when_unmodelled(R, tmp_path):
    rep = R.Report.__new__(R.Report)
    rep.P = sys.modules["tp_pipeline"]
    rep.th_doc = {"lines": [{"id": "stand.D1.G1.taps_per_s suite es1", "value_set": "recommended", "class": "fixed_t",
                             "sigma_hat": 0.0, "rate": None, "rate_status": "degenerate/unvalidated"}]}
    rep.memo, rep.errors = {}, []
    halts = {
        "p3a": {"class": "FAIL", "lines": [{"id": "p3a.capability.zero_kg_paired", "value": 2.7, "line": 1.0}],
                "chance_rate_sum": None, "chance_rate_unmodelled": [{"id": "p3a.capability.zero_kg_paired"}],
                "resolution": {"kind": "accept-chance"}},
        "sh": {"class": "FAIL", "lines": [{"id": "stand.D2.K6.x", "value": 0.935, "line": 0.945, "rate": 6.4e-12}],
               "chance_rate_sum": 6.4e-12, "chance_rate_unmodelled": [], "resolution": {"kind": "accept-chance"}},
        "sn": {"class": "FAIL", "lines": [{"id": "stand.D1.G1.taps_per_s suite es1", "value": 1.04, "line": 1.0, "rate": 0.0}],
               "chance_rate_sum": 0.0, "resolution": {"kind": "amend"}},
        "st": {"class": "DRIVER", "lines": [], "chance_rate_sum": None, "resolution": {"kind": "amend"}},
    }
    ctx = SimpleNamespace(state={"halts": {k: {"class": v["class"], "status": "x", "record": f"{k}.json"} for k, v in halts.items()},
                                 "stages": {}, "pin": None, "amendments": []},
                          qlog=tmp_path / "none.log", dir=tmp_path)
    out = rep.run_provenance("R18", {"ctx": ctx, "gates": {}, "halts": halts})
    by = {h["stage"]: h for h in out["halts"]}
    assert by["p3a"]["chance_rate"] is None and by["p3a"]["chance_rate_text"] == "rate unknown (unmodelled)"
    assert by["sh"]["chance_rate_text"] == "6.4e-12" and by["sh"]["resolution"] == "accepted as chance"
    assert by["sn"]["chance_rate"] is None and by["sn"]["chance_rate_text"].startswith("rate unknown (degenerate")
    assert by["st"]["chance_rate_text"].startswith("not applicable")


def test_report_carries_no_dated_path_and_uses_spec_roots(R):
    for i, ln in enumerate((TP / "report.py").read_text().splitlines(), 1):
        code = ln.split("#", 1)[0]
        assert not re.search(r"[\"'/]20\d\d-\d\d-\d\d_[a-z]", code) or '"""' in ln, f"report.py:{i}: {ln.strip()[:100]}"


def test_readme_checks_find_every_named_script_and_flag(R, tmp_path):
    f = tmp_path / "x.py"
    f.write_text('ap.add_argument("--alpha")\n  --beta) B=$2 ;;\n')
    assert R._flag_present(f, "--alpha") and R._flag_present(f, "--beta") and not R._flag_present(f, "--gamma")
    P = sys.modules["tp_pipeline"]
    checks = R.readme_checks(P.Spec(SPEC))
    bad = [c for c in checks if not c["exists"] or c["missing_flags"]]
    assert not bad, bad


# ============================================================================== the .md never cuts a number
NUM = re.compile(r"\d+(?:\.\d+)?")
# the recorded F19 p3a 'why' shape (F19/deliverable.json): a raw [:160] once printed 'margin 0.0' for 'margin 0.0291'
P3A_WHY = ("eligible, not chosen: max_stable 90 < T*'s 135 (p3b); paired 0-kg check FAIL (max margin use 2.71; "
           "fails: track_err diag suite es1: worse by 0.0380 > margin 0.0291 (R); track_err move_turn suite es1: "
           "worse by 0.0994 > margin 0.0367 (R))")


def _md_cells(m):
    for ln in m.splitlines():
        if ln.startswith("| ") and ln.endswith(" |"):
            for c in re.split(r"(?<!\\)\|", ln)[1:-1]:
                yield c.strip().replace("\\|", "|")


def _strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _strings(v)


def _cut_cells(R, m, doc):
    """md table cells that end INSIDE a token of a longer source string (a silent mid-number slice), or that are
    clipped without a clause boundary / without the full text printed under the table."""
    tail = R.CLIP_MARK + " (full below)"
    notes = [ln.split("(full): ", 1)[1] for ln in m.splitlines() if "(full): " in ln]
    srcs = {s for s in _strings(doc) if len(s) >= 24} | set(notes)
    bad = []
    for c in _md_cells(m):
        clipped = c.endswith(tail)
        core = c[: -len(tail)] if clipped else c
        if len(core) < 24:
            continue
        longer = [s for s in srcs if len(s) > len(core) and s.startswith(core)]
        if clipped:
            if not any(s[len(core)] in " ;," and s in notes for s in longer):
                bad.append(("clipped without a boundary + full note", c))
        elif any(core[-1].isalnum() and (s[len(core)].isalnum() or s[len(core)] == ".") for s in longer):
            bad.append(("cut inside a token", c))
    return bad


def test_clip_never_cuts_inside_a_token(R):
    for s in (P3A_WHY, "stand.D2.K6.homed share bin4 es1 >= R_h's - max(0.05, 2 SE) 0.935 vs 0.945",
              "filled from the recorded 2026-09-08 build, labelled; switches to the pin re-read"):
        for n in range(4, len(s) + 2):
            c = R._clip(s, n)
            if c == s:
                continue
            assert c.endswith(R.CLIP_MARK), (n, c)
            core = c[: -len(R.CLIP_MARK)]
            assert s.startswith(core) and s[len(core)] in " ;,", (n, c)       # cut only at a boundary
            assert set(NUM.findall(core)) <= set(NUM.findall(s)), (n, c)      # every number in it is whole
    assert R._clip("0.0291", 3) == "0.0291"                                    # one unbroken token: never cut
    assert R._clip(None, 10) == "" and R._clip("short", 10) == "short"


def test_md_tables_never_cut_a_number(R, empty_spec):
    """The T* 'why', the HALT 'failed lines' / CHARTER and the cell index keep every number whole in the .md:
    a long cell is clipped only at a clause boundary and its full text is printed under the table."""
    spec, _ = empty_spec
    doc = R.build(spec, "F19", None, None)
    doc["task3"]["tstar"]["considered"] = [
        {"stage": "p3a", "status": "accepted", "max_stable": 90, "zero_kg": "FAIL", "max_margin_used": 2.71,
         "chosen": False, "why": P3A_WHY},
        {"stage": "p3b", "status": "ceiling", "max_stable": 135, "zero_kg": "PASS", "max_margin_used": 0.66,
         "chosen": True, "why": "CHOSEN: the highest max_stable (135)"}]
    lines = [{"id": f"stand.D2.K6.homed share bin{b} es{e} >= R_h's - max(0.05, 2 SE)", "value": 0.83 + b / 1000,
              "line": 0.9451} for b in range(1, 6) for e in (1, 2)]
    charter = ("user 2026-10-21: accept, disclose -- lazy R_h control beside T*: K6 bins 1/4 walker-caused (control "
               "fails the same lines); residual K4 miss 13/10 vs control 7/3; re-anchored head 0.8945 vs 0.9451")
    doc["provenance"]["F19"]["halts"] = [{"stage": "sh", "class": "FAIL", "resolution": "accepted as chance",
                                          "chance_rate_text": "0.148", "lines": lines, "charter": charter}]
    long_why = "recorded build " + "x " * 50 + "dated 2026-09-08 (step 11) with 0.0291 margin"
    doc["cells"][0]["why"] = long_why
    m = R.md(doc)
    assert "margin 0.0291 (R)" in m and "margin 0.0 |" not in m                       # the reported misprint
    for x in lines:                                                                      # every failed line, whole
        assert f"{x['id']} {R.fmt(x['value'])} vs {R.fmt(x['line'])}" in m
    assert charter in m and long_why in m                                              # CHARTER + cell index: full
    assert "p3a why (full): " + P3A_WHY in m and "p3b why (full)" not in m             # only clipped cells get a note
    assert not _cut_cells(R, m, doc), _cut_cells(R, m, doc)


# ============================================================================== the real records (skipped when absent)
@needs_f19
def test_real_md_tables_print_whole_numbers(R, empty_spec):
    """On the recorded F19 T* candidates and the F19 / R18 HALTs: no table cell ends mid-number."""
    spec, _ = empty_spec
    doc = R.build(spec, "F19", None, None)
    rep = R.Report(SPEC, "F19")
    rep.rec = R.run_records(SPEC, "F19")
    doc["task3"]["tstar"] = rep.tstar_block(rep.run_sources("F19"), rep.rec)
    doc["provenance"]["F19"] = rep.run_provenance("F19", rep.rec)
    if (PL / "R18" / "pipeline_state.json").is_file():
        doc["other"] = "R18"
        doc["provenance"]["R18"] = rep.run_provenance("R18", R.run_records(SPEC, "R18"))
    m = "\n".join(R.md_task3(doc) + R.md_provenance(doc))
    assert "worse by 0.0380 > margin 0.0291" in m and "margin 0.0 |" not in m
    for run in ("F19", "R18"):
        for h in (doc["provenance"].get(run) or {}).get("halts") or []:
            for x in h["lines"]:
                assert f"{x['id']} {R.fmt(x['value'])} vs {R.fmt(x['line'])}" in m, (run, h["stage"], x["id"])
            assert (h.get("charter") or h.get("resolution_why") or "") in m, (run, h["stage"])
    assert not _cut_cells(R, m, doc), _cut_cells(R, m, doc)

@needs_f19
def test_real_f19_sources_and_tstar_record(R):
    rep = R.Report(SPEC, "F19")
    src = rep.run_sources("F19")
    assert src["tstage"] == "p3b" and src["sel"].get("cand") == "c28000"
    assert R._ok(src["T1"]["suite@1"]) and R._ok(src["St"]["field_hard@1"]) and R._ok(src["TS"]["field_hard@1"])
    assert sorted(src["levels"]) == [0, 45, 90, 135, 180]
    assert sorted(src["st_levels"]) == [0, 90]                      # level 0 = the unarmed sel rows
    rep.rec = R.run_records(SPEC, "F19")
    tb = rep.tstar_block(src, rep.rec)
    assert tb["heads"]["p3b"]["zero_kg"]["verdict"] == "PASS"
    assert tb["heads"]["p3b"]["zero_kg_before_regate"]["verdict"] == "FAIL"     # v1.1 excluded it (history/p3b.1)
    assert [x["stage"] for x in tb["considered"] if x.get("chosen")] == ["p3b"]


@needs_f19
def test_real_wallclock_matches_the_fill_plan(R):
    _needs_log(PL / "F19" / "pipeline_queue.log", PL / "preflight" / "stepA_queue.log",
               PL / "preflight" / "stepC" / "stepC_queue.log", *_c7_failed_logs())
    w = R.wallclock(PL / "F19" / "pipeline_queue.log")
    assert abs(w["total_active_h"] - 46.89) < 0.01 and abs(w["total_park_h"] - 9.23) < 0.01
    assert abs(w["span_h"] - 56.12) < 0.01 and len(w["parks"]) == 3
    if (PL / "R18" / "pipeline_queue.log").is_file():
        r = R.wallclock(PL / "R18" / "pipeline_queue.log")
        assert abs(r["total_active_h"] - 44.78) < 0.01 and abs(r["total_park_h"] - 13.87) < 0.01
    pf = R.preflight_wallclock(PL)
    assert abs(pf["A"]["h"] - 0.90) < 0.01 and abs(pf["C"]["h"] - 52.62) < 0.01


@needs_f19
def test_real_keep_set_cites_the_student_by_sha_over_the_guard(R):
    rep = R.Report(SPEC, "F19")
    rep.F = rep.run_sources("F19")
    ks = rep.keep_set()
    if ks["student_bytes"] is None:
        pytest.skip("the student checkpoint is not on this machine")
    assert ks["head_max_bytes"] == 64 * 1024 * 1024 and ks["exceeds_guard"] is True
    assert ks["recorded_sha_matches"] is True
    b = yaml.safe_load((PL / "bundle.yaml").read_text()) if (PL / "bundle.yaml").is_file() else None
    if b:
        dests = [b["head"]["dest"]] + [x["dest"] for x in b["head"]["extra_heads"]]
        assert "head/model_25992.pt" in dests and not any("model_28000" in x for x in dests)
        assert b["keep_globs"] == ["summary.md", "complete.json", "tp_row.json", "code_state.json"]


# ============================================================================== step-11 polish (wf_580da618 verifiers)
def test_small_n_rule_prints_n_beside_a_thin_median(R, tmp_path):
    assert R.fmt_n(0.5, 3) == "0.500 (n=3)" and R.fmt_n(0.5, R.N_FLOOR) == "0.500" and R.fmt_n(0.5, None) == "0.500"
    assert R.fmt_n(None, 3) == "N/A" and R.fmt_n(0.5, 0) == "N/A"            # n = 0 is N/A, never a number
    st = mk_omni_stamp(tmp_path / "s")
    sm = json.loads((st / "scenario_metrics.json").read_text())
    sm["aggregate"]["suite"]["by_category"]["fwd"]["ratio"]["n"] = 4
    (st / "scenario_metrics.json").write_text(json.dumps(sm))
    c = R.read_categories(st)["by_category"]
    assert c["fwd"]["_n"]["ratio"] == 4
    assert R._pair_es({1: c["fwd"], 2: c["fwd"]}, None, "ratio", "") == "1.020 (n=4) / 1.020 (n=4)"
    e = R.read_episode_extras(mk_omni_stamp(tmp_path / "t", duty_missing=(4, 5, 6)))
    assert e["fwd"]["_n"]["duty_hold"] == 1 and R._fn(e["fwd"], "duty_hold") == "0.500 (n=1)"
    assert R.N_RULE.startswith("A median computed from fewer than 10 episodes")


def test_rows_log_counts_only_rows_with_both_sentinels(R, tmp_path):
    lg = tmp_path / "q.log"
    lg.write_text("S11 ROW id=ST_suite_body_1 2026-10-24T01:00:00-04:00\n"
                  "S11 DONE id=ST_suite_body_1 rc=0 2026-10-24T01:30:00-04:00\n"
                  "S11 DONE id=ST_suite_body_2 rc=0 2026-10-24T01:40:00-04:00\n"      # no ROW sentinel
                  "S11 ROW id=T1_grid_body_1 2026-10-24T02:00:00-04:00\n")             # still running
    r = R.rows_log(lg)
    assert r["n_rows_measured"] == 1 and abs(r["measured_h"] - 0.5) < 1e-9 and len(r["rows"]) == 3
    assert r["n_rows_by_state"] == {"done": 2, "running": 1, "skip": 0}


def test_wallclock_segments_name_their_launch_and_mark_the_superseded_one(R, tmp_path):
    lg = tmp_path / "pipeline_queue.log"
    lg.write_text("\n".join([
        "T4PIPE START run=F19 pid=1 2026-10-22T15:00:00",
        "T4PIPE STAGE run=F19 stage=st kind=distill 2026-10-22T15:00:00",
        "T4PIPE LAUNCH run=F19 stage=st distill name=T4F19_st from=p3a 2026-10-22T15:00:00",
        "T4PIPE HALT run=F19 stage=st class=DRIVER 2026-10-22T22:39:00",
        "T4PIPE STOPPED run=F19 STOP_PIPELINE present 2026-10-22T22:39:00",
        "T4PIPE START run=F19 pid=2 2026-10-22T23:39:00",
        "T4PIPE STAGE run=F19 stage=st kind=distill 2026-10-22T23:39:00",
        "T4PIPE LAUNCH run=F19 stage=st distill name=T4F19_st_a2 from=p3b 2026-10-22T23:39:00",
        "T4PIPE STAGE run=F19 stage=sel kind=select 2026-10-23T09:39:00",
        "T4PIPE END run=F19 2026-10-23T09:39:00", ""]))
    w = R.wallclock(lg)
    info = w["stages"]["st"]["segment_info"]
    assert [x["launch"]["name"] for x in info] == ["T4F19_st", "T4F19_st_a2"]
    assert abs(info[0]["h"] - (7 + 39 / 60)) < 1e-9 and abs(info[1]["h"] - 10.0) < 1e-9
    state = {"stages": {"st": {"superseded": [{"run_name": "T4F19_st", "charter": "CHARTER x", "status": "halt",
                                               "records": {"live": {"last": 26833, "run_dir": None}}}]}}}
    m = R.Report.mark_superseded("st", w["stages"]["st"], state)
    assert m["segment_info"][0]["superseded"]["last_iteration"] == 26833 and m["segment_info"][1]["superseded"] is None
    assert "superseded" not in (w["stages"]["st"]["segment_info"][0])        # the cached read is not mutated
    doc = {"run": "F19", "provenance": {"selection_disclosure": {"text": "x"}, "F19": {"wallclock": {**w, "stages": {**w["stages"], "st": m}},
                                                "gpu_h": {"total": 17.65, "superseded_h": info[0]["h"]}}}}
    md = "\n".join(R.md_provenance(doc))
    assert "st (total)" in md and "st [1/2] T4F19_st from p3a -- **SUPERSEDED**" in md
    assert "st [2/2] T4F19_st_a2 from p3b -- the run's distill of record" in md


def mk_preflight(pl: Path, *, dry_tail="T4PIPE INJECT run=dry stage=st kind=LIVE (dry pass) 2026-10-17T18:30:50"):
    c = pl / "preflight" / "stepC"
    (c / "dryspec" / "dry.c7failed_2026-10-18").mkdir(parents=True)
    (c / "stepC_queue.log").write_text("\n".join([
        "T4C JOB C1_x gpu_h=0.10 script=a 2026-10-16T22:00:00-04:00",
        "T4C DONE C1_x rc=0 wall_s=3600  2026-10-16T23:00:00-04:00",
        "T4C JOB C1 gpu_h=0.00 script=b 2026-10-16T23:00:00-04:00",
        "T4C DONE C1 rc=0 wall_s=360  PASS 2026-10-16T23:06:00-04:00",
        "T4C JOB C7_dry gpu_h=6.00 script=c 2026-10-17T16:26:55-04:00",
        "T4C DONE C7_dry rc=1 wall_s=108062  2026-10-18T22:27:57-04:00",
        "T4C JOB C7_dry gpu_h=6.00 script=c 2026-10-18T23:08:57-04:00",
        "T4C DONE C7_dry rc=1 wall_s=20009  2026-10-19T04:42:26-04:00", ""]))
    (c / "dryspec" / "dry.c7failed_2026-10-18" / "pipeline_queue.log").write_text(
        "T4PIPE STAGE run=dry stage=st kind=distill status=pending 2026-10-17T18:30:20\n" + dry_tail + "\n")
    return pl


def test_preflight_c_measures_the_c7_hang_from_the_records(R, tmp_path):
    pf = R.preflight_wallclock(mk_preflight(tmp_path / "pl"))
    c = pf["C"]
    tot = (3600 + 360 + 108062 + 20009) / 3600
    hang = (datetime_(2026, 10, 18, 22, 27, 57) - datetime_(2026, 10, 17, 18, 30, 50)).total_seconds() / 3600
    assert abs(c["h"] - tot) < 1e-9 and abs(c["hang"]["h"] - hang) < 1e-9 and abs(c["h_excl_hang"] - (tot - hang)) < 1e-9
    assert abs(c["cpu_check_jobs_h"] - 0.1) < 1e-9 and "excluded from the GPU-h figure" in c["note"]
    # the failed dry log not ending on st: no hang is claimed (and no GPU-h figure without it)
    pf2 = R.preflight_wallclock(mk_preflight(tmp_path / "pl2", dry_tail="T4PIPE GATE run=dry stage=p3a x 2026-10-17T18:30:50"))
    assert pf2["C"]["hang"] is None and pf2["C"]["h_excl_hang"] is None and "no GPU-h figure" in pf2["C"]["note"]


def datetime_(*a):
    from datetime import datetime
    return datetime(*a)


def test_gpu_total_names_every_term_and_shows_the_hang_both_ways(R):
    rep = R.Report.__new__(R.Report)
    rep.run, rep.other = "F19", "R18"
    pv = {"F19": {"gpu_h": {"total": 47.3, "stages_active_h": 46.89, "superseded_h": 7.65, "lazy_control_h": 0.42}},
          "R18": {"gpu_h": {"total": 45.16, "stages_active_h": 44.78, "superseded_h": 0.0, "lazy_control_h": 0.38}},
          "step11_rows": {"measured_h": 2.0, "n_rows_measured": 14, "n_rows_seen": 15},
          "preflight": {"A": {"h": 0.9}, "C": {"h": 52.62, "h_excl_hang": 24.67, "hang": {"h": 27.95, "rule": "r"},
                                             "cpu_check_jobs_h": 0.09}}}
    t = rep.gpu_total(pv)
    assert abs(t["total_excl_hang"] - (47.3 + 45.16 + 2.0 + 0.9 + 24.67)) < 1e-9
    assert abs(t["total_incl_hang"] - t["total_excl_hang"] - 27.95) < 1e-9 and t["missing"] == []
    line = R.md_gpu_total(t)
    assert "Total GPU-h (measured): 120.03" in line and "14 rows with both S11 ROW and DONE" in line
    assert "minus the C7 stop hang 27.95 h" in line and "With the 27.95-h C7 hang counted: 147.98" in line
    assert "7.65 h of superseded segments" in line and "superseded" not in line.split("R18 45.16")[1].split(")")[0]
    pv["step11_rows"] = {}
    t2 = rep.gpu_total(pv)
    assert [x["term"] for x in t2["missing"]] == ["step-11 rows"] and "Not counted (no measurement): step-11 rows" in R.md_gpu_total(t2)


def test_halt_k6_caveat_and_recorded_rate(R, tmp_path):
    rep = R.Report.__new__(R.Report)
    rep.P = sys.modules["tp_pipeline"]
    rep.th_doc = {"lines": [{"id": "stand.D1.G1.taps_per_s suite es1", "value_set": "recommended", "class": "fixed_t",
                             "sigma_hat": 0.0, "rate": None, "rate_status": "degenerate/unvalidated"}]}
    rep.memo, rep.errors = {}, []
    halts = {
        "sh": {"class": "FAIL", "lines": [{"id": "stand.D2.K4.miss es2", "value": 10, "line": 3, "rate": 0.14},
                                          {"id": "stand.D2.K6.homed share bin4 es1", "value": 0.89, "line": 0.945,
                                           "rate": 6.4e-12}],
               "chance_rate_sum": 0.14, "chance_rate_unmodelled": [], "resolution": {"kind": "accept-chance"}},
        "sn": {"class": "FAIL", "lines": [{"id": "stand.D1.G1.taps_per_s suite es1", "value": 1.04, "line": 1.0, "rate": 0.0}],
               "chance_rate_sum": 0.0, "resolution": {"kind": "amend"}},
    }
    ctx = SimpleNamespace(state={"halts": {k: {"class": v["class"], "status": "x", "record": f"{k}.json"} for k, v in halts.items()},
                                 "stages": {}, "pin": None, "amendments": []}, qlog=tmp_path / "none.log", dir=tmp_path)
    by = {h["stage"]: h for h in rep.run_provenance("R18", {"ctx": ctx, "gates": {}, "halts": halts})["halts"]}
    assert "S1-era R_h" in by["sh"]["chance_rate_caveat"] and "sum 6.4e-12" in by["sh"]["chance_rate_caveat"]
    assert R.halt_rate_text(by["sh"]).startswith("0.14 (its K6 line rates") and "pre-re-anchor" in R.halt_rate_text(by["sh"])
    assert by["sn"]["chance_rate"] is None and by["sn"]["chance_rate_recorded"] == 0.0      # the recorded 0.0 kept
    assert by["sn"]["chance_rate_caveat"] is None and "rate unknown (degenerate" in R.halt_rate_text(by["sn"])


def test_floor_lines_table(R):
    floors = [{"id": f"sel.floor.{s}.{c}.es{e}", "value": v, "line": ln, "pass": True}
              for c in ("fwd", "side", "back", "diag") for e in (1, 2)
              for s, v, ln in (("survival", 1.0, 0.9), ("ratio", 0.99, 0.7))]
    floors.append({"id": "sel.floor.side22_fastfast", "value": ["FAST", "FAST"], "line": "fastfast", "pass": True})
    m = "\n".join(R.md_floors(floors))
    assert "17 lines of `sel.floor.*`" in m and "| fwd | 1.000 / 1.000 | 0.900 | 0.990 / 0.990 | 0.700 | PASS |" in m
    assert "`sel.floor.side22_fastfast`: FAST/FAST vs line fastfast -- PASS" in m
    assert R.md_floors([]) == []


def test_refs_tstar_block_reads_the_calib_row_or_says_why_not(R, tmp_path):
    assert "N/A: no SN calibration record" in R.Report.refs_tstar(tmp_path / "absent.json", "SN_x")["why"]
    f = tmp_path / "sn_calib.json"
    f.write_text(json.dumps({"rows": {"HOLD": {}}}))
    assert "has no rows['REFS']" in R.Report.refs_tstar(f, "SN_x")["why"]
    cl = lambda v: {"value": v, "line": 0.5, "ok": v <= 0.5}  # noqa: E731
    f.write_text(json.dumps({"rows": {
        "REFS": {"name": "REFS_T4F19", "kind": "obs0", "gates": {"G1": {"verdict": "FAIL", "expected": "FAIL", "lands": "yes",
                 "invalid": [], "clauses": {"steps_per_s suite es1": cl(2.5), "steps_per_s suite es2": cl(2.4),
                                            "field_hard es1+2+3+4 completion": cl(0.88)}}}},
        "SN_x": {"gates": {"G1": {"verdict": "PASS", "clauses": {"steps_per_s suite es1": cl(0.0),
                                                                "steps_per_s suite es2": cl(0.1)}}}}}}))
    rt = R.Report.refs_tstar(f, "SN_x")
    assert rt["available"] and rt["gates"]["G1"]["clauses"]["steps_per_s suite es1"] == {
        "refs": 2.5, "line": 0.5, "refs_ok": False, "head": 0.0}
    m = "\n".join(R.md_refs_tstar(rt))
    assert "| G1 | FAIL (expected FAIL, lands yes) | steps_per_s suite es* | 2.500 / 2.400 | 0.000 / 0.100 | 0.500 / 0.500 |" in m
    assert "| field_hard es1+2+3+4 completion | 0.880 | N/A | 0.500 |" in m        # not mangled into es*


def test_invalid_kinds_and_the_k5_reason(R):
    k = R.invalid_kinds(["H suite@1: p2r_mode.s1_checkpoint_sha256 03b6009a6e95 != REFS 968cc5509899",
                         "[H] G1: suite@1: code_sha / eval_sha / dirty differ from REFS_S1_s18's (('a', 'b', False) vs "
                         "('c', 'd', False))"])
    assert k["n"] == 2 and k["references"] == ["REFS_S1_s18"]
    assert k["kinds"] == ["code_sha / eval_sha", "p2r_mode.s1_checkpoint_sha256 03b6009a != REFS 968cc550"]
    assert R.invalid_kinds(None)["n"] == 0


def test_clip_local_only_line_and_the_tag_name(R):
    s = R.md_clip_local(["a/x.mp4", "a/y.mp4"], {"a/x.mp4": "e/.gitignore:19:*.mp4", "a/y.mp4": "e/.gitignore:19:*.mp4"})
    assert "**local-only**" in s and "`e/.gitignore:19:*.mp4`" in s and "`a/x.mp4`" in s and "NOT ignored" not in s
    assert "NOT ignored (would be committable): `a/y.mp4`" in R.md_clip_local(["a/x.mp4", "a/y.mp4"], {"a/x.mp4": "r"})
    assert R.BASELINE_TAG == "omni-sim2real-baseline-2026-10-24"
    src = (TP / "report.py").read_text()
    assert "omni-sim2real-baseline-<date>" not in src and any(R.BASELINE_TAG in x for x in R.RECONCILIATION_MORE)


def test_real_clips_are_gitignored(R):
    mp4 = sorted((PL / "F19" / "clip").glob("*.mp4")) if (PL / "F19" / "clip").is_dir() else []
    if not mp4:
        pytest.skip("no F19 clip on this machine")
    ign = R.git_ignored(mp4)
    assert all(R.rel(p) in ign and ign[R.rel(p)].endswith("*.mp4") for p in mp4), ign


@needs_f19
def test_real_f19_st_segments_and_k5_reason(R):
    rep = R.Report(SPEC, "F19")
    rec = R.run_records(SPEC, "F19")
    rep.F = rep.run_sources("F19")
    # the tracked records first (they read on every checkout), then the gitignored queue logs' wall-clock
    sh = rep.sh_block(rep.F["ctx"].dir, rec["gates"]["sh"])
    k5 = sh["lazy_control"]["k5_invalid"]
    assert k5["verdict"] == "INVALID" and "REFS_S1_s18" in k5["references"] and "43/43 PASS" in k5["text"]
    rt = rep.refs_tstar(rep.F["ctx"].dir / "stand" / "sn_calib.json", "SN_T4F19_sn_s19")
    assert rt["available"] and rt["row"] == "REFS_T4F19" and rt["gates"]["G1"]["verdict"] == "FAIL"
    _needs_log(PL / "F19" / "pipeline_queue.log", PL / "preflight" / "stepC" / "stepC_queue.log", *_c7_failed_logs())
    pv = rep.run_provenance("F19", rec)
    info = pv["wallclock"]["stages"]["st"]["segment_info"]
    assert [x["launch"]["name"] for x in info] == ["T4F19_st", "T4F19_st_a2"]
    assert info[0]["superseded"] and info[1]["superseded"] is None and abs(info[0]["h"] - 7.65) < 0.01
    assert abs(pv["gpu_h"]["superseded_h"] - info[0]["h"]) < 1e-9
    pf = R.preflight_wallclock(PL)
    assert abs(pf["C"]["hang"]["h"] - 27.95) < 0.01 and abs(pf["C"]["h_excl_hang"] - 24.67) < 0.01
