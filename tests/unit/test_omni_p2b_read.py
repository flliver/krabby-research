"""Omni phase 2, stage p2b -- the CHARTER 10 / P2B_PLAN.md step-6 reads in ``scripts/p2_read.py``. CPU only: nothing
here starts Isaac/Kit or a GPU.

* the pure ladder vs REF (the n = 400 binomial-SE gain gate), the chained ANCW flat floor, the s18 attribution and
  the C0-only-at-s18 head list run through the reader's own ``--self-test``;
* the p2a reads are unchanged (``load_point`` without ``fh_seeds`` reads field_hard at env_seeds 1 and 2 only);
* end to end on a fabricated ``$OUT`` tree (``P2_OUT``): REF / head / ANCW / S1 p2a stamps in run_arm.sh's layout,
  read through ``load_p2b`` and the CLI (``--p2b-inertness`` / ``--p2b-screen`` / ``--p2b-recipe`` / ``--p2b-md``).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2/scripts"
SC = ("c" * 64, "e" * 64, False)
SC_P2A = ("a" * 64, "e" * 64, False)            # the p2a code_sha (ANCW / the S1 p2a heads' own stamps)
ENV4 = {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_STD_PARAM": "soft", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25",
        "KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}
FH, FL, RO, FT, FS = "field_hard__omni", "field_light__omni", "ring_obst__omni", "field_turn__omni", "field_suite__omni"
SEEDS = (16, 17, 18)
_N = [0]


def _load(out: Path | None, monkeypatch):
    """A fresh p2_read module; ``out`` -> P2_OUT (read at import, so every default path points into tmp)."""
    if out is not None:
        monkeypatch.setenv("P2_OUT", str(out))
    _N[0] += 1
    name = f"_t_p2b_read_{_N[0]}"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "p2_read.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def read():
    spec = importlib.util.spec_from_file_location("_t_p2b_read_pure", SCRIPTS / "p2_read.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_p2b_read_pure"] = mod
    spec.loader.exec_module(mod)
    return mod


# ================================================================================================ pure
def test_the_self_test_passes(read, capsys):
    assert read._self_test() == 0
    out = capsys.readouterr().out
    assert "[FAIL]" not in out and "self-test: PASS" in out
    for case in ("p2b SE gate: REF 0.73 + 0.06 -> NO GAIN", "p2b ANCW chain", "attribution:", "C0 listed only at s18",
                 "inertness:"):
        assert case in out


@pytest.mark.parametrize("p_ref,p_arm,ok", [(0.83, 0.89, True), (0.80, 0.86, True), (0.73, 0.79, False),
                                           (0.73, 0.7925, True), (0.83, 0.87, False)])
def test_the_se_gate_reachability(read, p_ref, p_arm, ok):
    """CHARTER 10: REF + max(0.05, 2 SE) at n = 400 each. +0.06 clears at REF 0.83 / 0.80; at 0.73 the bar is 0.0602."""
    g = read.se_need(p_ref, 400, p_arm, 400)
    assert g["ok"] is ok
    assert g["margin"] >= read.P2B_GAIN_FLOOR and g["margin"] < 0.061


def test_p2a_constants_and_rows_are_unchanged(read):
    assert read.NEEDED_STAMPS == ("suite@1", "lr@1", f"{FH}@1", f"{FH}@2", f"{FL}@1", f"{FL}@2", f"{RO}@1", f"{RO}@2")
    assert read.P2B_NEEDED == read.NEEDED_STAMPS + (f"{FH}@3", f"{FH}@4")
    assert read.GAIN_FH == 0.10 and read.RECIPE_ARM == "S1"
    assert read.p2b_head_names(("CL",)) == ["P2__CL_s16", "P2__CL_s17", "P2__C0_s18", "P2__CL_s18"]


# ================================================================================================ fabricated $OUT
def _suite(err=0.03, fc=0.80, yp=0.10, mt=0.90, trans=1.0) -> dict:
    by = {c: {"ratio": {"median": 1.0 + err}, "action_jitter": {"median": 0.2}} for c in ("fwd", "back", "side", "diag")}
    by["turn"] = {"facing_closed": {"median": fc}, "yaw_progress": {"median": yp}, "action_jitter": {"median": 0.2}}
    by["move_turn"] = {"action_jitter": {"median": 0.2}}
    by["stand"] = {"tripod": {"median": 0.79}, "steps_per_s": {"median": 3.2}, "six_down_frac": {"median": 0.002}}
    surv = {h: trans for h in ("fwd", "lat_l", "lat_r", "back", "diag")}
    surv.update(move_turn=mt, move_turn_r=mt)
    coh = {"fwd": "fwd", "lat_l": "side", "lat_r": "side", "back": "back", "diag": "diag", "move_turn": "move_turn",
           "move_turn_r": "move_turn"}
    return {"by_category": by, "survival_by_hold": surv, "category_of_hold": coh}


def _stamp(out: Path, name: str, row: str, seed: int, *, rate=None, track=0.8, suite=None, ckpt="h" * 64, sc=SC,
           ts="2026-10-08_00-00-00") -> str:
    d = out / "lr" / name / row / f"seed{seed:03d}" / ts
    d.mkdir(parents=True)
    agg = {"n_episodes": 100, "termination_reasons": {"fall": 0 if rate is None else int(round((1 - rate) * 100))}}
    if rate is not None:
        agg.update(schedule_completion_rate=rate, tracking_ratio={"median": track})
    if suite is not None:
        agg["suite"] = suite
    if row == FT:
        agg.update(schedule_completion_rate=0.9, facing_by_hold={
            h: {"facing_closed": {"median": 0.8}, "yaw_progress_rate": {"median": 0.2}} for h in ("turn_l90", "turn_r90")})
    (d / "scenario_metrics.json").write_text(json.dumps({"aggregate": agg}))
    (d / "run_meta.json").write_text(json.dumps({"env_vars": ENV4, "env_vars_manifest": {}, "checkpoint_sha256": ckpt}))
    (d / "code_state.json").write_text(json.dumps(dict(zip(("code_sha", "eval_sha", "dirty"), sc))))
    (d / ("complete.json" if suite is not None else "p2_complete.json")).write_text("{}")
    return str(d)


def _lr(out: Path, name: str, hands=("FAST", "FAST"), feet=0.24, ckpt="h" * 64, sc=SC):
    (out / "arms").mkdir(parents=True, exist_ok=True)
    labels = {f"lat_{h}_{s}": {"n": 100, "excluded": {}, "feet_ss": feet} for h in "lr" for s in ("022", "025")}
    (out / "arms" / f"{name}_lr.json").write_text(json.dumps({name: {"class": hands[0], "class_r": hands[1],
                                                                     "labels": labels}}))
    (out / "arms" / f"{name}_lr_along.json").write_text(json.dumps({name: {"class_along": "FAST",
                                                                           "class_along_r": "FAST"}}))
    _stamp(out, name, "side22_25lr__omni", 1, ckpt=ckpt, sc=sc)


def _point(out: Path, name: str, *, fh=0.80, fh_seeds=(1, 2, 3, 4), fl=0.95, ro=0.90, err=0.03, ckpt="h" * 64, sc=SC,
           suite_row=True, suite2=False, driver=False, lr=True, fl2=None):
    """Every run_arm.sh row of one name; ``driver``: the suite goes to a driver stamp (returned) instead of a row."""
    fh4 = fh if isinstance(fh, tuple) else (fh,) * len(fh_seeds)
    for s, r in zip(fh_seeds, fh4):
        _stamp(out, name, FH, s, rate=r, ckpt=ckpt, sc=sc)
    for s in (1, 2):
        _stamp(out, name, FL, s, rate=fl if (s == 1 or fl2 is None) else fl2, ckpt=ckpt, sc=sc)
        _stamp(out, name, RO, s, rate=ro, ckpt=ckpt, sc=sc)
    _stamp(out, name, FT, 1, ckpt=ckpt, sc=sc)
    _stamp(out, name, FS, 1, suite=_suite(), ckpt=ckpt, sc=sc)
    if lr:
        _lr(out, name, ckpt=ckpt, sc=sc)
    drv = None
    if driver:
        drv = _stamp(out / "driver", name, "suite__omni", 1, suite=_suite(err=err), ckpt=ckpt, sc=sc)
    elif suite_row:
        _stamp(out, name, "suite__omni", 1, suite=_suite(err=err), ckpt=ckpt, sc=sc)
    if suite2:
        _stamp(out, name, "suite__omni", 2, suite=_suite(err=err), ckpt=ckpt, sc=sc)
    return drv


def _record(out: Path, ref: str, drv: str):
    (out / "arms").mkdir(parents=True, exist_ok=True)
    (out / "arms" / f"{ref}.record.json").write_text(json.dumps({"ref": ref, "chain": ref, "driver_stamp": drv,
                                                                 "run_dir": None, "checkpoint": f"/x/{ref}.pt"}))


def _tree(out: Path, heads: dict, ref_fh=0.80):
    """ANCW + the S1 p2a heads (p2a code) + REF_S1 (p2b code, same checkpoint, same reads) + the p2b ``heads``."""
    for s in SEEDS:
        _point(out, f"ANCW_s{s}", fh=0.45, fh_seeds=(1, 2), sc=SC_P2A, ckpt=f"{s}" * 64, suite2=(s == 18), lr=False)
        drv = _point(out, f"P2__S1_s{s}", fh=ref_fh, fh_seeds=(1, 2), sc=SC_P2A, ckpt=f"s{s}" * 32, driver=True)
        _record(out, f"P2__S1_s{s}", drv)
        _point(out, f"REF_S1_s{s}", fh=ref_fh, ckpt=f"s{s}" * 32, suite2=(s == 18))
    for ref, kw in heads.items():
        _record(out, ref, _point(out, ref, driver=True, **kw))


def test_load_point_reads_field_hard_at_four_seeds_only_when_asked(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _point(out, "REF_S1_s18", fh=(0.80, 0.82, 0.84, 0.86))
    r = _load(out, monkeypatch)
    p2a = r.load_point("REF_S1_s18", with_tb=False)                    # the p2a path: unchanged, es1 + es2 only
    assert sorted(p2a["comp"]["field_hard"]) == [1, 2] and f"{FH}@3" not in p2a["stamps"]
    assert r.pooled(p2a, "field_hard") == (pytest.approx(0.81), 200)
    p2b = r.load_point("REF_S1_s18", with_tb=False, fh_seeds=r.P2B_FH_SEEDS)
    assert sorted(p2b["comp"]["field_hard"]) == [1, 2, 3, 4]
    assert r.fh400(p2b) == (pytest.approx(0.83), 400) and r.fh_track(p2b) == pytest.approx(0.8)
    assert r.pooled(p2b, "field_hard") == r.pooled(p2a, "field_hard")  # pooled() stays the n = 200 read


def test_end_to_end_ladder_attribution_and_recipe(tmp_path, monkeypatch):
    out = tmp_path / "out"
    heads = {"P2__C0_s18": {"fh": 0.82}, "P2__CL_s18": {"fh": 0.90}, "P2__TG_s18": {"fh": 0.84},
             "P2__CL_s16": {"fh": 0.88}, "P2__CL_s17": {"fh": 0.79},
             "P2__TG_s16": {"fh": 0.90, "err": 0.15}}                                   # +0.12 |ratio - 1| vs REF
    _tree(out, heads)
    r = _load(out, monkeypatch)
    res = r.load_p2b(with_tb=False, levers=("CL", "TG"))
    assert set(res["heads"]) == set(r.p2b_head_names(("CL", "TG")))
    assert res["heads"]["P2__TG_s17"].get("missing")                                  # not run yet: PENDING
    vs = r.p2b_verdicts(res, need_dose=False)
    got = {k: v["verdict"] for k, v in vs.items()}
    assert got == {"P2__C0_s18": "NO GAIN", "P2__CL_s18": "PASS", "P2__TG_s18": "NO GAIN", "P2__CL_s16": "PASS",
                   "P2__CL_s17": "NO GAIN", "P2__TG_s16": "FLAT FAIL", "P2__TG_s17": "PENDING"}
    # REF and the heads share the p2b sidecar; ANCW / the p2a heads carry the p2a code_sha and are NOT compared by
    # sidecar (chained from values): no INVALID EVAL anywhere
    assert vs["P2__CL_s18"]["report"]["field_hard_n"] == 400 and vs["P2__CL_s18"]["report"]["ref_field_hard_n"] == 400
    assert vs["P2__CL_s18"]["report"]["field_hard_track"] == pytest.approx(0.8)
    assert vs["P2__CL_s16"]["label"] == "lever + continuation" and vs["P2__CL_s18"]["label"] is None
    assert any("ANCW_s16 (chained)" in w for w in vs["P2__TG_s16"]["why"])
    attr = r.p2b_attributions(res, vs, ("CL", "TG"))
    assert attr["CL"]["status"] == "BEATS CONTINUATION"                                # 0.90 vs C0 0.82 (+0.08)
    assert attr["TG"]["beats_continuation"] is False                                   # its REF verdict is NO GAIN
    rec = r.p2b_recipes(vs, ("CL", "TG"))
    assert rec["CL"]["reading"] == "ADOPTABLE" and rec["CL"]["pass"] == 2                # PASS s16 / s18, NO GAIN s17
    assert rec["TG"]["reading"] == "PENDING"


def test_a_head_on_other_code_than_ref_is_invalid(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _tree(out, {"P2__CL_s18": {"fh": 0.90, "sc": ("d" * 64, "e" * 64, False)}})
    r = _load(out, monkeypatch)
    vs = r.p2b_verdicts(r.load_p2b(with_tb=False, levers=("CL",)), need_dose=False)
    assert vs["P2__CL_s18"]["verdict"] == "INVALID EVAL"


def test_cli_screen_recipe_and_md(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _tree(out, {"P2__C0_s18": {"fh": 0.82}, "P2__CL_s18": {"fh": 0.90}})
    r = _load(out, monkeypatch)
    md = tmp_path / "p2b.md"
    monkeypatch.setenv("P2B_LEVERS", "CL")
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2b-screen", "--no-tb", "--p2b-md", str(md)])
    assert r.main() == 0
    j = json.loads(capsys.readouterr().out)
    assert set(j["verdicts"]) == {"P2__C0_s18", "P2__CL_s18"} and set(j["attribution"]) == {"CL"}
    assert j["verdicts"]["P2__CL_s18"]["verdict"] == "PENDING"                         # --no-tb: no dose read
    assert any("dose read" in w for w in j["verdicts"]["P2__CL_s18"]["why"])
    rows = [ln for ln in md.read_text().splitlines() if ln.startswith("| P2__")]
    assert [ln.split(" | ")[0] for ln in rows] == ["| P2__CL_s16", "| P2__CL_s17", "| P2__C0_s18", "| P2__CL_s18"]
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2b-recipe", "--no-tb"])
    assert r.main() == 4                                                                 # CL s16 / s17 not run
    assert json.loads(capsys.readouterr().out)["recipe"]["CL"]["reading"] == "PENDING"


def test_cli_inertness(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _tree(out, {})
    r = _load(out, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2b-inertness"])
    assert r.main() == 0, capsys.readouterr().out
    j = json.loads(capsys.readouterr().out)
    assert j["ok"] and set(j["rows"]["18"].values()) == {"equal"}
    # one REF row reads differently (a newer stamp, field_light es2 0.94 vs 0.95) -> stop and ask
    _stamp(out, "REF_S1_s17", FL, 2, rate=0.94, ckpt="s17" * 32, ts="2026-10-08_09-00-00")
    assert r.main() == 4
    j = json.loads(capsys.readouterr().out)
    assert any(f.startswith(f"s17 {FL}@2") and "rate" in f for f in j["fails"])


def test_cli_inertness_fails_closed_without_ref(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    for s in SEEDS:
        _record(out, f"P2__S1_s{s}", _point(out, f"P2__S1_s{s}", fh_seeds=(1, 2), sc=SC_P2A, driver=True))
    r = _load(out, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2b-inertness"])
    assert r.main() == 4
    assert "REF row unread" in capsys.readouterr().out
