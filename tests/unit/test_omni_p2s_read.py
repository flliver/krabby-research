"""Stage p2s (P2S_PLAN.md "Pre-registered verdicts" / CHARTER 11): the ``scripts/p2_read.py`` --p2s-* reads and the
``scenarios_p2s.yaml`` manifest. CPU only: nothing here starts Isaac/Kit or a GPU.

* the reader's own ``--p2s-self-test`` (synthetic STILL / PARTIAL / GAMED / COST / NO EFFECT, the ladder rungs above
  them, attribution, the step-7 recipe mapping, the step-5 inertness);
* ``walkstop__omni``: loaded, validated and compiled through the eval's own ``gait_eval/schedule.py`` (no Isaac), flat
  ground the suite__omni way (no env block), three "stand" holds and the two stand -> moving onset pairs
  p2s_stand.py sees; ``scenarios_p2.yaml`` byte-identical to HEAD;
* ``walkstop_read`` on a fabricated walkstop stamp (raw npz episodes in the harness's layout), and ``p2s_summary`` on
  the real REF suite stamp (= the step-1 baseline, when present);
* end to end on a fabricated ``$OUT`` tree (``P2_OUT``): REFS / C0R / the p2b points / ANCW / the S1 p2a heads / a p2s
  head, the stand raws stubbed per point, through ``load_p2s`` / ``p2s_verdicts`` and the CLI (``--p2s-screen`` /
  ``--p2s-is-still`` / ``--p2s-recipe`` / ``--p2s-md`` / ``--p2s-inertness``).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
GAIT_SCRIPTS = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts"
SC = ("c" * 64, "e" * 64, False)
SC_P2A = ("a" * 64, "e" * 64, False)
ENV4 = {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_STD_PARAM": "soft", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25",
        "KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}
FH, FL, RO, FT, FS, WS = ("field_hard__omni", "field_light__omni", "ring_obst__omni", "field_turn__omni",
                          "field_suite__omni", "walkstop__omni")
_N = [0]


def _load(out: Path | None, monkeypatch):
    if out is not None:
        monkeypatch.setenv("P2_OUT", str(out))
    _N[0] += 1
    name = f"_t_p2s_read_{_N[0]}"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "p2_read.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def read():
    spec = importlib.util.spec_from_file_location("_t_p2s_read_pure", SCRIPTS / "p2_read.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_p2s_read_pure"] = mod
    spec.loader.exec_module(mod)
    return mod


# ================================================================================================ pure
def test_the_p2s_self_test_passes(read, capsys):
    assert read._p2s_self_test() == 0
    out = capsys.readouterr().out
    assert "[FAIL]" not in out and "p2s self-test: PASS" in out
    for case in ("STILL:", "PARTIAL:", "GAMED:", "NO EFFECT:", "COST:", "OBSTACLE LOSS:", "attribution:", "recipe:",
                 "inertness:", "PARTIAL from the clearance clause alone"):
        assert case in out


def test_the_p2b_self_test_still_passes(read, capsys):
    assert read._self_test() == 0
    assert "self-test: PASS" in capsys.readouterr().out


def test_constants(read):
    assert read.P2S_VALIDITY == read.P2B_VALIDITY + ("suite@2", f"{WS}@1", f"{WS}@2")
    assert read.P2S_BAND_FLOOR == {"off_ground_share": 0.02, "steps_per_s": 0.25, "taps_per_s": 0.25, "ks_m_s": 0.10,
                                   "onset_s": 0.1}
    assert (read.P2S_FALL_FLOOR, read.P2S_ONSET_FLOOR, read.P2S_HEIGHT_TOL, read.P2S_FT_FLOOR) == (0.03, 0.3, 0.03, 0.05)
    assert read.refs_name(16) == "REFS_S1_s16" and read.C0R == "C0R_s18"
    assert read.p2s_names({}) == {"ST2_s18": "P2__ST2_s18", "ST1_s18": "P2__ST1_s18", "dose": "ST2",
                                  "rep_s16": "P2__ST2_s16", "rep_s17": "P2__ST2_s17"}
    # the STILL thresholds are p2s_stand.py's, never redefined here
    assert read.PS().STILL_THRESHOLDS == {"steps_per_s": 0.5, "clearance_m": 0.05, "off_ground_share": 0.10,
                                          "hover_episodes_per_100": 5.0, "taps_per_s": 1.0, "ks_m_s": 0.5}


# ================================================================================================ the manifest
def _schedule():
    if str(GAIT_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(GAIT_SCRIPTS))
    from gait_eval import schedule as S  # noqa: PLC0415
    return S


def test_walkstop_validates_and_compiles_through_the_eval_schedule():
    S = _schedule()
    scs, _defaults = S.load_manifest(CAMP / "scenarios_p2s.yaml")         # load_manifest runs validate_scenario
    assert [s.id for s in scs] == [WS]
    sc = scs[0]
    S.validate_scenario(sc)
    assert sc.command_frame == "world" and sc.kind == "schedule" and sc.episodes == 100
    assert sc.env_vars == {}                                               # flat ground: no env block (suite__omni)
    assert [h.label for h in sc.schedule] == ["fwd", "stand", "lat_l", "stand", "back", "stand"]
    assert [h.hold_s for h in sc.schedule] == [6.0, 12.0, 6.0, 12.0, 6.0, 12.0]
    assert [h.as_world() for h in sc.schedule] == [(0.25, 0.0, 0.0), (0.0, 0.0, 0.0), (0.20, 1.5707963, 0.0),
                                                   (0.0, 0.0, 0.0), (0.20, 3.1415927, 0.0), (0.0, 0.0, 0.0)]
    assert sc.episode_length_s >= sc.total_duration_s == 55.5
    c = S.compile_schedule(sc, dt=0.02)
    assert c.hold_labels == ["fwd", "stand", "lat_l", "stand", "back", "stand"] and c.n_steps == 2775
    assert sorted(set(c.segment_id[:, 0].tolist())) == [-1, 0, 1, 2, 3, 4, 5]


def test_walkstop_has_three_stands_and_two_onset_pairs(read):
    """p2s_stand.py keys a stand on hold_labels[i] == "stand": the three stands; stand_onset_pairs pairs each with the
    moving hold right after it -- stand -> lat_l, stand -> back (the last stand ends the episode: no onset pair)."""
    S = _schedule()
    sc = S.load_manifest(CAMP / "scenarios_p2s.yaml")[0][0]
    labels, speeds = [h.label for h in sc.schedule], [h.as_world()[0] for h in sc.schedule]
    assert [i for i, lab in enumerate(labels) if lab == "stand"] == [1, 3, 5]
    assert read.PS().stand_onset_pairs(labels, speeds) == [(1, 2, "lat_l"), (3, 4, "back")]


def test_the_header_is_scenarios_p2s_verbatim_and_p2_is_untouched():
    import yaml  # noqa: PLC0415
    a, b = (yaml.safe_load((CAMP / f).read_text()) for f in ("scenarios_p2.yaml", "scenarios_p2s.yaml"))
    assert {k: a[k] for k in ("manifest_version", "speed_ellipse", "ang_vel_max", "defaults")} == \
        {k: b[k] for k in ("manifest_version", "speed_ellipse", "ang_vel_max", "defaults")}
    rel = (CAMP / "scenarios_p2.yaml").relative_to(REPO)
    head = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"], capture_output=True)
    if head.returncode != 0:
        pytest.skip("not a git checkout")
    assert head.stdout == (CAMP / "scenarios_p2.yaml").read_bytes()


# ================================================================================================ walkstop_read (raws)
def _walkstop_stamp(root: Path, *, n_eps=4, onset_steps=(10, 20), fall_ep=None, seed=1) -> Path:
    """A walkstop__omni stamp in the harness's layout: run_meta (hold_labels / hold_speeds / dt) + raw/episode_<i>.npz.
    Every stand is a still six-down statue (500 N, feet 3 cm over the terrain); each moving hold accelerates to its
    commanded speed after ``onset_steps[k]`` steps (stand -> lat_l, stand -> back); ``fall_ep`` falls in its 2nd stand."""
    d = root / "lr" / "X" / WS / f"seed{seed:03d}" / "ts"
    (d / "raw").mkdir(parents=True)
    labels = ["fwd", "stand", "lat_l", "stand", "back", "stand"]
    speeds = [0.25, 0.0, 0.2, 0.0, 0.2, 0.0]
    dirs = [0.0, 0.0, np.pi / 2, 0.0, np.pi, 0.0]
    lens = [300, 600, 300, 600, 300, 600]
    (d / "run_meta.json").write_text(json.dumps({"hold_labels": labels, "hold_speeds": speeds, "dt": 0.02,
                                                 "env_vars": ENV4, "env_vars_manifest": {}}))
    for ep in range(n_eps):
        seg = np.concatenate([np.full(75, -1)] + [np.full(n, i) for i, n in enumerate(lens)])
        T = seg.size
        steady = np.zeros(T, bool)
        start = 75
        for n in lens:
            steady[start + 50:start + n] = True
            start += n
        force = np.full((T, 6), 500.0)
        foot = np.zeros((T, 6, 3))
        foot[..., 2] = 0.03
        cmd = np.zeros((T, 3))
        vel = np.zeros((T, 3))
        for i, (sp, di) in enumerate(zip(speeds, dirs)):
            idx = np.flatnonzero(seg == i)
            cmd[idx, 0], cmd[idx, 1] = sp * np.cos(di), sp * np.sin(di)
            k = {2: onset_steps[0], 4: onset_steps[1]}.get(i, 0)
            vel[idx[k:], 0], vel[idx[k:], 1] = sp * np.cos(di), sp * np.sin(di)
        cf = np.zeros(T, bool)
        if ep == fall_ep:
            cut = int(np.flatnonzero(seg == 3)[100]) + 1
            seg, steady, force, foot, cmd, vel, cf = (x[:cut] for x in (seg, steady, force, foot, cmd, vel, cf))
            cf[-1] = True
            T = cut
        np.savez(d / "raw" / f"episode_{ep:02d}.npz", foot_force_norm=force, foot_pos_w=foot,
                 foot_lin_vel_w=np.zeros((T, 6, 3)), terrain_z=np.zeros((T, 6)),
                 root_pos_w=np.tile([0.0, 0.0, 1.06], (T, 1)), root_lin_vel_b=vel, cmd_applied=cmd,
                 segment_id=seg, steady_mask=steady, heading_err=np.zeros(T), crab_failure=cf,
                 actions=np.zeros((T, 18)))
    return d


def test_walkstop_read_on_a_fabricated_stamp(read, tmp_path):
    st = _walkstop_stamp(tmp_path, n_eps=4, onset_steps=(10, 20), fall_ep=3)
    w = read.walkstop_read(str(st))
    assert w["stand_holds"] == [1, 3, 5] and w["onset_pairs"] == [[1, 2, "lat_l"], [3, 4, "back"]]
    assert w["n_episodes"] == 4 and w["env_seed"] == 1 and "STEADY" in w["window"] and "FULL" in w["window"]
    # 3 still episodes x 3 stands + the falling episode's first two stands (it falls inside the second)
    assert w["stand_reached"] == 3 * 3 + 2 and w["falls"] == 1
    assert w["pooled"]["steps_per_s"] == 0.0 and w["pooled"]["clearance_m"] == 0.0
    assert w["pooled"]["off_ground_share"] == 0.0 and w["pooled"]["loaded_count"] == 6.0
    # onset: full speed after 10 / 20 steps; the 5-step trailing mean first reaches 0.5 x |cmd| two steps later
    # (0.24 s to lat_l, 0.44 s to back); the faller never reaches 'back' (censored)
    assert w["onset"] == {"lat_l": {"median_s": 0.24, "n": 4}, "back": {"median_s": 0.44, "n": 3}}
    assert w["onset_censored"] == 1 and sorted(w["onset_values"]) == pytest.approx([0.24] * 4 + [0.44] * 3)
    assert w["onset_median_s"] == 0.24


def test_p2s_summary_reproduces_the_step1_ref_baseline(read):
    st = REPO / "parkour/outputs/omni_plan/phase2/lr/REF_S1_s18/suite__omni/seed001/2026-10-08_05-11-48"
    if not (st / "raw").is_dir():
        pytest.skip("REF_S1_s18's suite raws are not in this checkout")
    s = read.p2s_summary(str(st))
    assert (s["n"], s["steps_per_s"], round(s["clearance_m"], 3)) == (100, 2.83333, 0.135)   # p2s_baseline.json


# ================================================================================================ fabricated $OUT
def _suite(err=0.03) -> dict:
    by = {c: {"ratio": {"median": 1.0 + err}, "action_jitter": {"median": 0.2}, "tripod": {"median": 0.74},
              "steps_per_m": {"median": 9.6}, "swing_apex": {"median": 0.14}} for c in ("fwd", "back", "side", "diag")}
    by["turn"] = {"facing_closed": {"median": 0.8}, "yaw_progress": {"median": 0.1}, "action_jitter": {"median": 0.2},
                  "steps_per_rad_post": {"median": 5.7}}
    by["move_turn"] = {"action_jitter": {"median": 0.2}}
    by["stand"] = {"tripod": {"median": 0.76}, "steps_per_s": {"median": 2.8}, "six_down_frac": {"median": 0.01}}
    surv = {h: 1.0 for h in ("fwd", "lat_l", "lat_r", "back", "diag")}
    surv.update(move_turn=0.9, move_turn_r=0.9)
    coh = {"fwd": "fwd", "lat_l": "side", "lat_r": "side", "back": "back", "diag": "diag", "move_turn": "move_turn",
           "move_turn_r": "move_turn"}
    return {"by_category": by, "survival_by_hold": surv, "category_of_hold": coh}


def _stamp(out: Path, name: str, row: str, seed: int, *, rate=None, suite=None, ckpt="h" * 64, sc=SC,
           ts="2026-10-09_00-00-00", env=None, menv=None) -> str:
    d = out / "lr" / name / row / f"seed{seed:03d}" / ts
    d.mkdir(parents=True)
    agg = {"n_episodes": 100, "termination_reasons": {"fall": 0 if rate is None else int(round((1 - rate) * 100))}}
    if rate is not None:
        agg.update(schedule_completion_rate=rate, tracking_ratio={"median": 0.8})
    if suite is not None:
        agg["suite"] = suite
    if row == FT:
        agg.update(schedule_completion_rate=0.9, facing_by_hold={
            h: {"facing_closed": {"median": 0.8}, "yaw_progress_rate": {"median": 0.2}} for h in ("turn_l90", "turn_r90")})
    (d / "scenario_metrics.json").write_text(json.dumps({"aggregate": agg}))
    (d / "run_meta.json").write_text(json.dumps({"env_vars": env or ENV4, "env_vars_manifest": menv or {},
                                                 "checkpoint_sha256": ckpt}))
    (d / "code_state.json").write_text(json.dumps(dict(zip(("code_sha", "eval_sha", "dirty"), sc))))
    (d / ("complete.json" if suite is not None else "p2_complete.json")).write_text("{}")
    return str(d)


def _lr(out: Path, name: str, ckpt="h" * 64, sc=SC):
    (out / "arms").mkdir(parents=True, exist_ok=True)
    labels = {f"lat_{h}_{s}": {"n": 100, "excluded": {}, "feet_ss": 0.24} for h in "lr" for s in ("022", "025")}
    (out / "arms" / f"{name}_lr.json").write_text(json.dumps({name: {"class": "FAST", "class_r": "FAST",
                                                                     "labels": labels}}))
    (out / "arms" / f"{name}_lr_along.json").write_text(json.dumps({name: {"class_along": "FAST",
                                                                           "class_along_r": "FAST"}}))
    _stamp(out, name, "side22_25lr__omni", 1, ckpt=ckpt, sc=sc)


def _point(out: Path, name: str, *, fh=0.81, ckpt="h" * 64, sc=SC, driver=False, p2s=True, fh_seeds=(1, 2, 3, 4),
           suite2=True, ws_menv=None):
    for s in fh_seeds:
        _stamp(out, name, FH, s, rate=fh, ckpt=ckpt, sc=sc)
    for s in (1, 2):
        _stamp(out, name, FL, s, rate=0.98, ckpt=ckpt, sc=sc)
        _stamp(out, name, RO, s, rate=1.0, ckpt=ckpt, sc=sc)
    _stamp(out, name, FT, 1, ckpt=ckpt, sc=sc)
    _stamp(out, name, FS, 1, suite=_suite(), ckpt=ckpt, sc=sc)
    _lr(out, name, ckpt=ckpt, sc=sc)
    drv = None
    if driver:
        drv = _stamp(out / "driver", name, "suite__omni", 1, suite=_suite(), ckpt=ckpt, sc=sc)
    else:
        _stamp(out, name, "suite__omni", 1, suite=_suite(), ckpt=ckpt, sc=sc)
    if suite2:
        _stamp(out, name, "suite__omni", 2, suite=_suite(), ckpt=ckpt, sc=sc)
    if p2s:
        for s in (1, 2):
            _stamp(out, name, WS, s, ckpt=ckpt, sc=sc, menv=ws_menv)
    return drv


def _record(out: Path, ref: str, drv: str | None, **kw):
    (out / "arms").mkdir(parents=True, exist_ok=True)
    (out / "arms" / f"{ref}.record.json").write_text(json.dumps(dict({"ref": ref, "chain": ref, "driver_stamp": drv,
                                                                      "run_dir": None, "checkpoint": f"/x/{ref}.pt"},
                                                                     **kw)))


def _tree(out: Path, heads: dict):
    """ANCW + the S1 p2a heads (p2a code) + REF_S1 / P2__C0_s18 (p2b reads, same checkpoints) + REFS_S1 / C0R_s18 (p2s
    code: identical reads) + the p2s ``heads`` (driver suite stamps)."""
    for s in (16, 17, 18):
        _point(out, f"ANCW_s{s}", fh=0.45, fh_seeds=(1, 2), sc=SC_P2A, ckpt=f"{s}" * 64, p2s=False,
               suite2=(s == 18))
        drv = _point(out, f"P2__S1_s{s}", fh_seeds=(1, 2), sc=SC_P2A, ckpt=f"s{s}" * 32, driver=True, p2s=False,
                     suite2=False)
        _record(out, f"P2__S1_s{s}", drv)
        _point(out, f"REF_S1_s{s}", ckpt=f"s{s}" * 32, p2s=False, suite2=(s == 18), sc=SC_P2A)
        _point(out, f"REFS_S1_s{s}", ckpt=f"s{s}" * 32)
    drv = _point(out, "P2__C0_s18", ckpt="c0" * 32, driver=True, p2s=False, suite2=False, sc=SC_P2A)
    _record(out, "P2__C0_s18", drv)
    _point(out, "C0R_s18", ckpt="c0" * 32)
    for ref, kw in heads.items():
        _record(out, ref, _point(out, ref, driver=True, **kw))


def _raws(r, stand: dict) -> callable:
    """A p2s_raw stub: each point's stand raws by name (REF-like by default; ``stand[name]`` = a suite summary)."""
    def fake(stamps):
        st = stamps.get("suite@2") or stamps.get("suite@1") or ""
        parts = Path(st).parts
        name = parts[parts.index("lr") + 1] if "lr" in parts else ""
        s1 = stand.get(name) or r._p2s_summ(steps=2.67 if name == "C0R_s18" else 2.83)
        return {"suite": {1: s1, 2: s1}, "walkstop": {s: r._p2s_ws() for s in (1, 2)},
                "field": {k: r._p2s_summ() for k in [f"{FH}@{s}" for s in (1, 2, 3, 4)] + [f"{FL}@1", f"{FL}@2"]},
                "turn": r._p2s_summ(), "clip_share": {1: 0.3, 2: 0.3}, "field_onset": {}}
    return fake


STILL_SUMM = dict(steps=0.2, taps=0.3, clear=0.02, og=0.03, hover=0.0, ks=0.3)


def _with_dose(r, monkeypatch):
    monkeypatch.setattr(r.X(), "tail_reads", lambda rd: {"dose_ok": True, "mu_med_cam": 0.9, "mu_med_rod": 0.9})


def test_end_to_end_verdicts_and_attribution(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _tree(out, {"P2__ST2_s18": {}, "P2__ST1_s18": {"fh": 0.70}, "P2__ST2_s16": {}})
    r = _load(out, monkeypatch)
    monkeypatch.setattr(r, "p2s_raw", _raws(r, {"P2__ST2_s18": r._p2s_summ(**STILL_SUMM),
                                                "P2__ST1_s18": r._p2s_summ(**STILL_SUMM),
                                                "P2__ST2_s16": r._p2s_summ(steps=1.0, taps=1.2, clear=0.08, og=0.15,
                                                                           hover=2.0, ks=1.5)}))
    res = r.load_p2s(with_tb=False)
    vs = r.p2s_verdicts(res, need_dose=False)
    got = {k: (v["verdict"], (v.get("stand") or {}).get("outcome")) for k, v in vs.items()}
    assert got["P2__ST2_s18"] == ("STILL", "STILL")
    assert got["P2__ST1_s18"] == ("OBSTACLE LOSS", "STILL")                 # the stand outcome shows anyway
    assert got["P2__ST2_s16"] == ("PARTIAL", "PARTIAL")                      # vs REFS at s16 (no C0R there)
    assert got["C0R_s18"][1] == "NO EFFECT" and got["P2__ST2_s17"][0] == "PENDING"
    assert vs["P2__ST2_s16"]["label"] == "lever + continuation" and vs["P2__ST2_s18"]["comparator"] == "C0R_s18"
    assert vs["P2__ST2_s16"]["comparator"] == "REFS_S1_s16"
    scr = r.p2s_screen(res, vs)
    assert scr["attribution"]["P2__ST2_s18"]["status"] == "LEVER" and scr["st2_reads_still"] is True
    assert scr["c0r_clauses"][1]["steps_per_s"][2] is False                  # C0R's own clauses shown beside
    rep = scr["reported"]["P2__ST2_s18"]
    assert rep["steps_per_rad_post"] == 5.7 and rep["walking"]["fwd"]["tripod"] == 0.74
    assert rep["c0r_walking"]["fwd"]["tripod"] == 0.74 and rep["stand_clip_share"] == {1: 0.3, 2: 0.3}


def test_a_walkstop_stamp_with_obstacles_is_invalid(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _tree(out, {"P2__ST2_s18": {"ws_menv": {"KRABBY_FLAT_TERRAIN_MODE": "light"}}})
    r = _load(out, monkeypatch)
    monkeypatch.setattr(r, "p2s_raw", _raws(r, {"P2__ST2_s18": r._p2s_summ(**STILL_SUMM)}))
    vs = r.p2s_verdicts(r.load_p2s(with_tb=False, seeds=(18,)), need_dose=False)
    assert vs["P2__ST2_s18"]["verdict"] == "INVALID EVAL"
    assert any("walkstop" in w for w in vs["P2__ST2_s18"]["why"])


def test_cli_screen_is_still_md_and_recipe(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _tree(out, {"P2__ST2_s18": {}, "P2__ST2_s16": {}, "P2__ST2_s17": {}})
    r = _load(out, monkeypatch)
    monkeypatch.setattr(r, "p2s_raw", _raws(r, {n: r._p2s_summ(**STILL_SUMM)
                                                for n in ("P2__ST2_s18", "P2__ST2_s16", "P2__ST2_s17")}))
    for ref in ("P2__ST2_s18", "P2__ST2_s16", "P2__ST2_s17"):           # a run_dir so the dose read is taken
        rec = json.loads((out / "arms" / f"{ref}.record.json").read_text())
        _record(out, ref, rec["driver_stamp"], run_dir=str(tmp_path))
    _with_dose(r, monkeypatch)
    monkeypatch.setattr(r, "p2s_sigma_report", lambda *a: {"fork": None, "+1000": None, "end": None})
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2s-is-still", "P2__ST2_s18"])
    assert r.main() == 0
    assert json.loads(capsys.readouterr().out)["still"] is True
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2s-is-still", "P2__ST1_s18"])
    assert r.main() == 4                                                    # ST1 not run: PENDING, not STILL
    capsys.readouterr()
    md = tmp_path / "s.md"
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2s-screen", "--p2s-md", str(md)])
    assert r.main() == 0
    j = json.loads(capsys.readouterr().out)
    assert j["verdicts"]["P2__ST2_s18"]["verdict"] == "STILL" and j["attribution"]["P2__ST2_s18"]["lever"] is True
    txt = md.read_text()
    assert "| P2__ST2_s18 | **STILL** | STILL |" in txt and "- C0R_s18: es1" in txt
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2s-recipe"])
    assert r.main() == 0
    j = json.loads(capsys.readouterr().out)
    assert j["recipe"]["reading"] == "ADOPTABLE" and j["recipe"]["mapped"]["16"]["verdict"] == "PASS"
    assert j["recipe"]["mapping"]["STILL"] == "PASS"
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2s-recipe", "--no-tb"])
    assert r.main() == 4                                                    # --no-tb: no dose read -> PENDING


def test_cli_inertness(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _tree(out, {})
    r = _load(out, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2s-inertness"])
    assert r.main() == 0, capsys.readouterr().out
    j = json.loads(capsys.readouterr().out)
    rows = j["rows"]
    assert set(rows) == {"s18 REFS_S1_s18 vs REF_S1_s18", "s18 C0R_s18 vs P2__C0_s18"}
    assert rows["s18 REFS_S1_s18 vs REF_S1_s18"]["suite@2"] == "equal"
    assert rows["s18 C0R_s18 vs P2__C0_s18"]["suite@1"] == "p2b point lacks the row (not compared)"  # a driver stamp
    # a REFS row reading differently (a newer stamp) -> stop and ask
    _stamp(out, "REFS_S1_s18", FL, 2, rate=0.97, ckpt="s18" * 32, ts="2026-10-09_09-00-00")
    assert r.main() == 4
    assert any(f.startswith(f"s18 REFS_S1_s18 vs REF_S1_s18 {FL}@2") for f in json.loads(capsys.readouterr().out)["fails"])
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2s-inertness", "--p2s-seeds", "16,17"])
    assert r.main() == 0
    assert set(json.loads(capsys.readouterr().out)["rows"]) == {"s16 REFS_S1_s16 vs REF_S1_s16",
                                                               "s17 REFS_S1_s17 vs REF_S1_s17"}


def test_cli_inertness_fails_closed_before_the_references_exist(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _point(out, "REF_S1_s18", ckpt="s18" * 32, p2s=False)
    r = _load(out, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2s-inertness"])
    assert r.main() == 4
    j = json.loads(capsys.readouterr().out)
    assert any("p2s row unread" in f for f in j["fails"]) and any("p2b point missing" in f for f in j["fails"])
