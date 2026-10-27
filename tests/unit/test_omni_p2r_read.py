# SPDX-License-Identifier: BSD-3-Clause
"""Stage p2r (CHARTER 12 / P2R_PLAN.md step 1 -- a controller-level stand mode). CPU only: nothing here starts
Isaac/Kit or a GPU.

* ``scripts/p2_read.py``'s new ``--p2r-*`` reads: ``p2r_mode_fails`` (INVALID without ``run_meta.p2r_mode`` / a
  wrong variant / a checkpoint sha other than REFS's), tier-1 rung skipping (HAND LOSS / DOSE / ring_obst are never
  PENDING -- they were never going to be read), ``p2r_tier1_verdict``'s own ladder (FLAT FAIL, OBSTACLE LOSS on
  field_hard and on field_light at matched n, STILL / COST on pooled tier-1 stand-hold falls), ``p2r_survives_tier1``
  (field_hard-only, per the plan's literal wording), the raw reads (stand-mode share, handover falls by entry
  phase, settle-mode share, turn final heading, the drift premise and its STOP rule) and the two CLI entries
  (``--p2r-probe`` / ``--p2r-step1``) end to end on a fabricated ``$OUT`` tree;
* ``scripts/p2r_drift.py``: the hand-replicated actor forward pass on a tiny synthetic checkpoint (D == 0 for two
  copies of the same checkpoint, D > 0 for a perturbed one), ``actor_widths`` off a real-shaped ``agent.yaml``, and
  the CLI end to end on a tiny on-disk bank + checkpoints.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
SC = ("c" * 64, "e" * 64, False)
ENV4 = {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_STD_PARAM": "soft", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25",
        "KRABBY_OMNI_FACING_OBS_CLIP": "0.6"}
FH, FL, FT, WS, SUITE = "field_hard__omni", "field_light__omni", "field_turn__omni", "walkstop__omni", "suite__omni"
REFS_SHA = "s18" * 32
_N = [0]


def _load(out: Path | None, monkeypatch):
    if out is not None:
        monkeypatch.setenv("P2_OUT", str(out))
    _N[0] += 1
    name = f"_t_p2r_read_{_N[0]}"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "p2_read.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def read():
    spec = importlib.util.spec_from_file_location("_t_p2r_read_pure", SCRIPTS / "p2_read.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_p2r_read_pure"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def drift():
    spec = importlib.util.spec_from_file_location("_t_p2r_drift", SCRIPTS / "p2r_drift.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_p2r_drift"] = mod
    spec.loader.exec_module(mod)
    return mod


# ================================================================================================ names / mode
def test_p2r_names_and_tag(read):
    assert read.p2r_names({}) == {"probe": "P2R_PROBE_s18", "hold": "P2R_HOLD_s18", "st2m": "P2R_ST2M_s18"}
    assert read.p2r_names({"P2R_HOLD_S18": "P2R_HOLDb_s18"})["hold"] == "P2R_HOLDb_s18"
    assert read.p2r_name_tag("P2R_HOLD_s18") == "HOLD"
    assert read.p2r_name_tag("P2R_ST2M_s18") == "ST2M"
    assert read.p2r_name_tag("P2R_PROBE_s18") == "PROBE"
    assert read.p2r_name_tag("P2__A1_s18") is None
    assert read.P2R_VARIANT_OF_TAG == {"HOLD": "hold", "ST2M": "st2", "PROBE": "probe"}


def _run_meta(stamp: Path, **kw):
    base = {"env_vars": ENV4, "env_vars_manifest": {}, "checkpoint_sha256": REFS_SHA, "dt": 0.02}
    base.update(kw)
    (stamp / "run_meta.json").write_text(json.dumps(base))


def test_p2r_mode_of_and_fails(read, tmp_path):
    s = tmp_path / "s"
    s.mkdir()
    assert read.p2r_mode_of(str(s)) is None             # no run_meta at all
    _run_meta(s)                                        # run_meta but no p2r_mode key
    pt = {"name": "P2R_HOLD_s18", "stamps": {"suite@1": str(s)}, "valid": {"suite@1": {}}}
    fails = read.p2r_mode_fails(pt, "HOLD", REFS_SHA, stamps=("suite@1",))
    assert fails and "no run_meta.p2r_mode" in fails[0]

    _run_meta(s, p2r_mode={"variant": "st2", "s1_checkpoint_sha256": REFS_SHA})
    fails = read.p2r_mode_fails(pt, "HOLD", REFS_SHA, stamps=("suite@1",))
    assert fails and "variant 'st2' != 'hold'" in fails[0]

    _run_meta(s, p2r_mode={"variant": "hold", "s1_checkpoint_sha256": REFS_SHA})
    assert read.p2r_mode_fails(pt, "HOLD", REFS_SHA, stamps=("suite@1",)) == []

    # the stamp's OWN top-level checkpoint is irrelevant (ST2M's acting policy is legitimately ST2, not S1) --
    # only p2r_mode.s1_checkpoint_sha256 is compared
    _run_meta(s, checkpoint_sha256="not_s1_at_all", p2r_mode={"variant": "hold", "s1_checkpoint_sha256": REFS_SHA})
    assert read.p2r_mode_fails(pt, "HOLD", REFS_SHA, stamps=("suite@1",)) == []

    _run_meta(s, p2r_mode={"variant": "hold", "s1_checkpoint_sha256": "other" * 16})
    fails = read.p2r_mode_fails(pt, "HOLD", REFS_SHA, stamps=("suite@1",))
    assert fails and "s1_checkpoint_sha256" in fails[0]

    # a stamp this point does not have is never flagged (PENDING's business, not INVALID's)
    assert read.p2r_mode_fails({"name": "x", "stamps": {}, "valid": {}}, "HOLD", REFS_SHA, stamps=("suite@1",)) == []


# ================================================================================================ tier-1 pending
def test_p2r_tier1_pending_skips_lr_ring_and_dose(read):
    """A point that never ran lr / ring_obst / field_suite and has no run_dir (no dose read) is not PENDING for
    any of them -- tier 1 never reads them (:data:`P2R_TIER1_SKIPPED_RUNGS`)."""
    pt = {"name": "P2R_HOLD_s18", "stamps": {k: f"/{k}" for k in read.P2R_NEEDED_TIER1}, "p2s": {
        "suite": {1: {"n": 1}, 2: {"n": 1}}, "walkstop": {1: {"n": 1}, 2: {"n": 1}}}, "suite": {
        m: 0.0 for m in read.LISTED}, "field_turn": {"completion": 0.9}}
    pt["suite"]["trans_surv"] = {h: 1.0 for h in read.TRANSLATE_HOLDS}
    refs = dict(pt, name="REFS_S1_s18")
    sbd = {"off_ground_share": 0.02, "steps_per_s": 0.25, "taps_per_s": 0.25, "ks_m_s": 0.1, "onset_s": 0.1}
    miss = read.p2r_tier1_pending(pt, refs, sbd)
    assert not any("lr" in m or "ring_obst" in m or "field_suite" in m or "dose" in m for m in miss)
    assert miss == []                                    # every tier-1 need is present here


# ================================================================================================ fabricated $OUT
def _suite(err=0.03) -> dict:
    by = {c: {"ratio": {"median": 1.0 + err}, "action_jitter": {"median": 0.2}} for c in ("fwd", "back", "side", "diag")}
    by["turn"] = {"facing_closed": {"median": 0.8}, "yaw_progress": {"median": 0.1}}
    by["move_turn"] = {}
    surv = {h: 1.0 for h in ("fwd", "lat_l", "lat_r", "back", "diag")}
    surv.update(move_turn=0.9, move_turn_r=0.9)
    coh = {"fwd": "fwd", "lat_l": "side", "lat_r": "side", "back": "back", "diag": "diag", "move_turn": "move_turn",
           "move_turn_r": "move_turn"}
    return {"by_category": by, "survival_by_hold": surv, "category_of_hold": coh}


def _stamp(out: Path, name: str, row: str, seed: int, *, rate=None, suite=None, ckpt=REFS_SHA, sc=SC,
          ts="2026-10-09_00-00-00", env=None, menv=None, p2r_mode=None) -> str:
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
    rm = {"env_vars": env or ENV4, "env_vars_manifest": menv or {}, "checkpoint_sha256": ckpt, "dt": 0.02}
    if p2r_mode is not None:
        rm["p2r_mode"] = p2r_mode
    (d / "run_meta.json").write_text(json.dumps(rm))
    (d / "code_state.json").write_text(json.dumps(dict(zip(("code_sha", "eval_sha", "dirty"), sc))))
    (d / ("complete.json" if suite is not None else "p2_complete.json")).write_text("{}")
    return str(d)


def _point(out: Path, name: str, *, fh=0.81, fl=0.98, ckpt=REFS_SHA, sc=SC, driver=False, mode="hold",
          full_fl=False) -> str | None:
    """A tier-1 point: suite es1+es2, walkstop es1+es2, field_hard es1-4, field_turn es1, field_light es1 ONLY
    (no es2 -- tier 1 never reads it) unless ``full_fl`` (REFS_S1_s18 itself: it is the existing p2s/p2b
    reference, already read at both env_seeds everywhere -- only the ARM's tier-1 runs skip field_light es2).
    Every stamp carries ``run_meta.p2r_mode``."""
    p2r_mode = {"variant": mode, "s1_checkpoint_sha256": ckpt}
    for s in (1, 2, 3, 4):
        _stamp(out, name, FH, s, rate=fh, ckpt=ckpt, sc=sc, p2r_mode=p2r_mode)
    _stamp(out, name, FL, 1, rate=fl, ckpt=ckpt, sc=sc, p2r_mode=p2r_mode)
    if full_fl:
        _stamp(out, name, FL, 2, rate=fl, ckpt=ckpt, sc=sc, p2r_mode=p2r_mode)
    _stamp(out, name, FT, 1, ckpt=ckpt, sc=sc, p2r_mode=p2r_mode)
    for s in (1, 2):
        _stamp(out, name, WS, s, ckpt=ckpt, sc=sc, p2r_mode=p2r_mode)
    drv = None
    if driver:
        drv = _stamp(out / "driver", name, SUITE, 1, suite=_suite(), ckpt=ckpt, sc=sc, p2r_mode=p2r_mode)
    else:
        _stamp(out, name, SUITE, 1, suite=_suite(), ckpt=ckpt, sc=sc, p2r_mode=p2r_mode)
    _stamp(out, name, SUITE, 2, suite=_suite(), ckpt=ckpt, sc=sc, p2r_mode=p2r_mode)
    return drv


def _record(out: Path, ref: str, drv: str | None, **kw):
    (out / "arms").mkdir(parents=True, exist_ok=True)
    (out / "arms" / f"{ref}.record.json").write_text(json.dumps(dict(
        {"ref": ref, "chain": ref, "driver_stamp": drv, "run_dir": None, "checkpoint": f"/x/{ref}.pt"}, **kw)))


STILL_SUMM = dict(steps=0.2, taps=0.3, clear=0.02, og=0.03, hover=0.0, ks=0.3)


def _raws(r, overrides: dict) -> callable:
    """A p2s_raw stub: ``overrides[name]`` = the suite stand summary (defaults to a STILL stance)."""
    def fake(stamps):
        st = stamps.get("suite@2") or stamps.get("suite@1") or ""
        parts = Path(st).parts
        name = parts[parts.index("lr") + 1] if "lr" in parts else ""
        s1 = overrides.get(name) or r._p2s_summ(**STILL_SUMM)
        fh = r._p2s_summ(**STILL_SUMM)
        return {"suite": {1: s1, 2: s1}, "walkstop": {s: r._p2s_ws() for s in (1, 2)},
               "field": {k: fh for k in [f"{FH}@{s}" for s in (1, 2, 3, 4)] + [f"{FL}@1"]} | {f"{FL}@2": None},
               "turn": fh, "clip_share": {1: 0.3, 2: 0.3}, "field_onset": {}}
    return fake


def _tree(out: Path, heads: dict):
    _point(out, "REFS_S1_s18", mode="hold", full_fl=True)              # REFS carries any p2r_mode; it is never checked against it
    for ref, kw in heads.items():
        _record(out, ref, _point(out, ref, driver=True, **kw))


def test_end_to_end_tier1_still_survives(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _tree(out, {"P2R_HOLD_s18": {}})
    r = _load(out, monkeypatch)
    monkeypatch.setattr(r, "p2s_raw", _raws(r, {}))
    res = r.load_p2r_step1()
    refs = res["refs"]
    bd = r.bands(refs, refs)
    sbd = r.p2s_stand_bands(refs)
    pt = res["heads"]["P2R_HOLD_s18"]
    v = r.p2r_tier1_verdict(pt, refs, refs, bd, sbd)
    assert v["verdict"] == "STILL" and v["stand"]["outcome"] == "STILL"
    assert r.p2r_survives_tier1(v) is True
    assert v["obstacle"]["ring_obst"] == "not read in tier 1"
    assert set(v["skipped_rungs"]) == set(r.P2R_TIER1_SKIPPED_RUNGS)


def test_end_to_end_tier1_flat_fail(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _tree(out, {})
    r = _load(out, monkeypatch)
    monkeypatch.setattr(r, "p2s_raw", _raws(r, {}))
    pt = r.load_p2s_point("P2R_HOLD_s18", suite_stamp=_point(out, "P2R_HOLD_s18", driver=True), with_tb=False,
                          lrout=out / "lr")
    _record(out, "P2R_HOLD_s18", pt["stamps"]["suite@1"])
    refs = r.load_p2s_point("REFS_S1_s18", with_tb=False, lrout=out / "lr")
    bd, sbd = r.bands(refs, refs), r.p2s_stand_bands(refs)
    # re-stamp the arm's suite with a much worse tracking error -> FLAT FAIL
    _stamp(out, "P2R_HOLD_s18", SUITE, 1, suite=_suite(err=0.20), p2r_mode={"variant": "hold",
          "s1_checkpoint_sha256": REFS_SHA}, ts="2026-10-09_01-00-00")
    newest = r.newest_stamp("P2R_HOLD_s18", SUITE, 1, out / "lr")
    pt = r.load_p2s_point("P2R_HOLD_s18", suite_stamp=newest, with_tb=False, lrout=out / "lr")
    v = r.p2r_tier1_verdict(pt, refs, refs, bd, sbd)
    assert v["verdict"] == "FLAT FAIL"
    assert r.p2r_survives_tier1(v) is False


def test_end_to_end_tier1_obstacle_loss_field_hard(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _tree(out, {"P2R_ST2M_s18": {"mode": "st2", "fh": 0.60}})            # well below REFS 0.81 - margin
    r = _load(out, monkeypatch)
    monkeypatch.setattr(r, "p2s_raw", _raws(r, {}))
    res = r.load_p2r_step1()
    refs = res["refs"]
    bd, sbd = r.bands(refs, refs), r.p2s_stand_bands(refs)
    pt = res["heads"]["P2R_ST2M_s18"]
    v = r.p2r_tier1_verdict(pt, refs, refs, bd, sbd)
    assert v["verdict"] == "OBSTACLE LOSS"
    assert v["obstacle"]["field_hard"][3] is False
    assert r.p2r_survives_tier1(v) is False               # field_hard itself failed


def test_end_to_end_tier1_obstacle_loss_field_light_only_still_survives(tmp_path, monkeypatch):
    """A field_light-only OBSTACLE LOSS (field_hard itself fine) reads as surviving tier 1 -- the plan's literal
    "no OBSTACLE LOSS on field_hard (n = 400)" wording, per :func:`p2r_survives_tier1`'s docstring."""
    out = tmp_path / "out"
    _tree(out, {"P2R_HOLD_s18": {"fh": 0.82, "fl": 0.50}})               # field_hard fine, field_light tanked
    r = _load(out, monkeypatch)
    monkeypatch.setattr(r, "p2s_raw", _raws(r, {}))
    res = r.load_p2r_step1()
    refs = res["refs"]
    bd, sbd = r.bands(refs, refs), r.p2s_stand_bands(refs)
    pt = res["heads"]["P2R_HOLD_s18"]
    v = r.p2r_tier1_verdict(pt, refs, refs, bd, sbd)
    assert v["verdict"] == "OBSTACLE LOSS"
    assert v["obstacle"]["field_hard"][3] is True and v["obstacle"]["field_light"][3] is False
    assert r.p2r_survives_tier1(v) is True


def test_end_to_end_tier1_cost_on_stand_falls(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _tree(out, {"P2R_HOLD_s18": {}})
    r = _load(out, monkeypatch)
    # the arm's suite stamp (and only the arm's) reads a pile of stand-hold falls; REFS stays clean
    arm_still = r._p2s_summ(**dict(STILL_SUMM, falls=60, n=100))  # noqa: this n overrides the default
    monkeypatch.setattr(r, "p2s_raw", _raws(r, {"P2R_HOLD_s18": arm_still}))
    res = r.load_p2r_step1()
    refs = res["refs"]
    bd, sbd = r.bands(refs, refs), r.p2s_stand_bands(refs)
    pt = res["heads"]["P2R_HOLD_s18"]
    v = r.p2r_tier1_verdict(pt, refs, refs, bd, sbd)
    assert v["verdict"] == "COST" and v["cost"] == ["stand_falls"]
    assert r.p2r_survives_tier1(v) is False


def test_end_to_end_tier1_invalid_without_p2r_mode(tmp_path, monkeypatch):
    out = tmp_path / "out"
    _point(out, "REFS_S1_s18", mode="hold", full_fl=True)
    drv = _point(out, "P2R_HOLD_s18", driver=True, mode="hold")
    _record(out, "P2R_HOLD_s18", drv)
    # knock out p2r_mode on one tier-1 stamp of the arm
    (Path(r_fh := drv).parent)  # noqa: F841  (keep flake8 quiet about the walrus-adjacent name)
    r = _load(out, monkeypatch)
    monkeypatch.setattr(r, "p2s_raw", _raws(r, {}))
    res = r.load_p2r_step1()
    stamp = Path(res["heads"]["P2R_HOLD_s18"]["stamps"][f"{FH}@1"])
    meta = json.loads((stamp / "run_meta.json").read_text())
    del meta["p2r_mode"]
    (stamp / "run_meta.json").write_text(json.dumps(meta))
    res = r.load_p2r_step1()
    refs = res["refs"]
    bd, sbd = r.bands(refs, refs), r.p2s_stand_bands(refs)
    v = r.p2r_tier1_verdict(res["heads"]["P2R_HOLD_s18"], refs, refs, bd, sbd)
    assert v["verdict"] == "INVALID EVAL"
    assert any("no run_meta.p2r_mode" in w for w in v["why"])
    assert r.p2r_survives_tier1(v) is None


# ================================================================================================ raw reads
def _ft_stamp(root: Path, *, mode_during=("stand",), fall_at=None) -> Path:
    """A field_turn-shaped schedule stamp (walk / stand / turn_l90 / turn_r90), p2r_stand_mode True during every
    label in ``mode_during``. ``fall_at`` (hold_idx, step-within-hold) marks ``crab_failure`` there."""
    d = root / "lr" / "P2R_HOLD_s18" / FT / "seed001" / "ts"
    (d / "raw").mkdir(parents=True)
    labels = ["walk", "stand", "turn_l90", "turn_r90"]
    lens = [100, 100, 100, 100]
    (d / "run_meta.json").write_text(json.dumps({"hold_labels": labels, "hold_speeds": [0.25, 0.0, 0.0, 0.0],
                                                 "dt": 0.02, "env_vars": ENV4, "env_vars_manifest": {}}))
    seg = np.concatenate([np.full(n, i) for i, n in enumerate(lens)])
    T = seg.size
    mode = np.isin(seg, [labels.index(lab) for lab in mode_during])
    force = np.where(mode[:, None], np.hstack([np.zeros((T, 3)), np.full((T, 3), 600.0)]), np.full((T, 6), 600.0))
    heading = np.zeros(T)
    t_idx = np.flatnonzero(seg == labels.index("turn_l90"))
    heading[t_idx] = np.linspace(1.5, 0.1, t_idx.size)
    t_idx2 = np.flatnonzero(seg == labels.index("turn_r90"))
    heading[t_idx2] = np.linspace(-1.5, -0.2, t_idx2.size)
    cf = np.zeros(T, bool)
    if fall_at is not None:
        hold_idx, k = fall_at
        cf[np.flatnonzero(seg == hold_idx)[k]] = True
    actions = np.zeros((T, 18))
    actions[mode] = 0.0
    np.savez(d / "raw" / "episode_00.npz", foot_force_norm=force, foot_pos_w=np.zeros((T, 6, 3)),
             foot_lin_vel_w=np.zeros((T, 6, 3)), terrain_z=np.zeros((T, 6)), root_pos_w=np.tile([0, 0, 1.06], (T, 1)),
             root_lin_vel_b=np.zeros((T, 3)), cmd_applied=np.zeros((T, 3)), segment_id=seg,
             steady_mask=np.ones(T, bool), heading_err=heading, crab_failure=cf, actions=actions,
             p2r_stand_mode=mode)
    return d


def test_p2r_hold_reads_stand_share_and_falls_in_stand(read, tmp_path):
    st = _ft_stamp(tmp_path, mode_during=("stand",), fall_at=(1, 50))     # falls INSIDE the stand hold
    r = read.p2r_hold_reads(str(st))
    assert r["n_episodes"] == 1
    assert r["stand_mode_share_by_label"] == {"walk": 0.0, "stand": 1.0, "turn_l90": 0.0, "turn_r90": 0.0}
    assert r["falls_in_stand_mode"] == 1
    assert r["handover_falls_by_entry_phase"] == {}          # the mode never exits mid-episode (stand is the last)


def test_p2r_hold_reads_handover_fall_by_entry_phase(read, tmp_path):
    # mode covers "stand" AND the first part of "turn_l90" (a settled-turn latch carry-over), then exits with a
    # fall 10 steps after the exit (within the 2s / 100-step window) -- 3 feet unloaded at the exit step (force's
    # construction: unloaded during mode, loaded otherwise, so the exit step itself is the FIRST loaded step)
    st = _ft_stamp(tmp_path, mode_during=("stand",), fall_at=(2, 10))
    r = read.p2r_hold_reads(str(st))
    assert r["falls_in_stand_mode"] == 0
    assert r["handover_falls_by_entry_phase"] == {0: 1}       # exit step is loaded (0 feet < 50N): phase 0


def test_p2r_turn_final_heading(read, tmp_path):
    st = _ft_stamp(tmp_path)
    out = read.p2r_turn_final_heading(str(st))
    assert out == {"turn_l90": 0.1, "turn_r90": 0.2}
    assert read.p2r_turn_final_heading(None) is None
    assert read.p2r_turn_final_heading(str(tmp_path / "nope")) is None


def test_p2r_settle_share_reads_run_meta_verbatim(read, tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "run_meta.json").write_text(json.dumps({"p2r_mode": {"settle_mode_share_by_wave": [0.1, 0.2, 0.0]}}))
    assert read.p2r_settle_share(str(d)) == [0.1, 0.2, 0.0]
    assert read.p2r_settle_share(None) is None
    d2 = tmp_path / "s2"
    d2.mkdir()
    (d2 / "run_meta.json").write_text(json.dumps({"p2r_mode": {}}))
    assert read.p2r_settle_share(str(d2)) is None


# ================================================================================================ bitwise / swap-live
def _episode_pair(root: Path, name: str, row: str, *, seed=1, mode=None, actions_a=None, actions_b=None) -> tuple:
    da = root / "lr" / "REFS_S1_s18" / row / f"seed{seed:03d}" / "a"
    db = root / "lr" / name / row / f"seed{seed:03d}" / "b"
    (da / "raw").mkdir(parents=True)
    (db / "raw").mkdir(parents=True)
    n = 20
    a = np.zeros((n, 3)) if actions_a is None else actions_a
    b = a.copy() if actions_b is None else actions_b
    extra = {} if mode is None else {"p2r_stand_mode": mode}
    np.savez(da / "raw" / "episode_00.npz", actions=a, segment_id=np.zeros(n, int))
    np.savez(db / "raw" / "episode_00.npz", actions=b, segment_id=np.zeros(n, int), **extra)
    return str(da), str(db)


def test_p2r_bitwise_diff(read, tmp_path):
    ref, same = _episode_pair(tmp_path, "P2R_PROBE_s18", FH)
    assert read.p2r_bitwise_diff(ref, same) == []
    ref2, diff = _episode_pair(tmp_path / "x", "P2R_PROBE_s18", FH, actions_b=np.ones((20, 3)))
    d = read.p2r_bitwise_diff(ref2, diff)
    assert d and "not bitwise equal" in d[0]


def test_p2r_launcher_inertness(read, tmp_path):
    rows = ("suite@1",)
    ref, same = _episode_pair(tmp_path, "P2R_PROBE_s18", SUITE)
    ok = read.p2r_launcher_inertness({"suite@1": ref}, {"suite@1": same}, rows=rows)
    assert ok["ok"] is True
    ref2, diff = _episode_pair(tmp_path / "x", "P2R_PROBE_s18", SUITE, actions_b=np.ones((20, 3)))
    bad = read.p2r_launcher_inertness({"suite@1": ref2}, {"suite@1": diff}, rows=rows)
    assert bad["ok"] is False and "suite@1" in bad["fails"]
    missing = read.p2r_launcher_inertness({"suite@1": ref}, {}, rows=rows)
    assert missing["ok"] is False and "missing" in missing["fails"]["suite@1"]


def test_p2r_hold_swap_live(read, tmp_path):
    n = 20
    mode = np.zeros(n, bool)
    mode[10:] = True                                      # stand mode from step 10 onward
    actions_ref = np.random.default_rng(0).normal(size=(n, 3))
    actions_hold = actions_ref.copy()
    actions_hold[mode] = 0.0                               # the controller holds the gait: zero on every stand step
    ref, hold = _episode_pair(tmp_path, "P2R_HOLD_s18", SUITE, mode=mode, actions_a=actions_ref, actions_b=actions_hold)
    r = read.p2r_hold_swap_live(ref, hold)
    assert r["zero_actions_on_stand"] is True
    assert r["stand_mode_share"] == pytest.approx(0.5)
    assert r["pre_entry_matches_refs"] is True
    assert r["fails"] == []

    actions_bad = actions_hold.copy()
    actions_bad[0] = 99.0                                  # pre-entry action now differs from REFS
    ref3, hold_bad = _episode_pair(tmp_path / "y", "P2R_HOLD_s18", SUITE, mode=mode, actions_a=actions_ref,
                                   actions_b=actions_bad)
    r2 = read.p2r_hold_swap_live(ref3, hold_bad)
    assert r2["pre_entry_matches_refs"] is False
    assert any("pre-entry" in f for f in r2["fails"])

    assert read.p2r_hold_swap_live(None, None)["fails"] == ["no HOLD suite stamp"]


# ================================================================================================ drift premise
def _dmu_stamp(root: Path, *, name_values: dict, mode: np.ndarray) -> Path:
    d = root / "lr" / "P2R_PROBE_s18" / SUITE / "seed001" / "ts"
    (d / "raw").mkdir(parents=True)
    n = mode.size
    kw = {"p2r_stand_mode": mode, "segment_id": np.zeros(n, int)}
    for name, vals in name_values.items():
        kw[f"p2r_dmu_{name}"] = vals
    np.savez(d / "raw" / "episode_00.npz", **kw)
    return d


def test_p2r_dmu_series_splits_stand_and_non_stand(read, tmp_path):
    mode = np.array([False] * 6 + [True] * 4)
    vals = np.array([1.0] * 6 + [9.0] * 4)
    st = _dmu_stamp(tmp_path, name_values={"ST2_m20500": vals}, mode=mode)
    r = read.p2r_dmu_series(str(st), "ST2_m20500")
    assert r == {"non_stand_mean": 1.0, "non_stand_n": 6, "stand_mean": 9.0, "stand_n": 4}
    assert read.p2r_dmu_series(str(st), "NOPE") is None
    assert read.p2r_dmu_series(None, "ST2_m20500") is None


def test_p2r_drift_premise_stop_rule(read, tmp_path):
    mode = np.zeros(10, bool)
    st2_506 = np.full(10, 0.40)
    st = _dmu_stamp(tmp_path, name_values={"ST2_m20500": st2_506, "ST2_m21000": st2_506, "ST2_model_24993": st2_506,
                                           "C0R_m24993": st2_506}, mode=mode)
    stamps = [str(st)]
    # C0 at +506 well under half of D_ref (0.40) -> no stop
    dj = {"ckpts": {"C0_m20500": {"D": 0.1}, "C0_m21000": {"D": 0.1}}}
    r = read.p2r_drift_premise(stamps, drift_json=dj)
    assert r["d_ref_506"] == pytest.approx(0.40)
    assert r["d_c0_506"] == pytest.approx(0.1)
    assert r["stop"] is False
    # C0 at +506 above half of D_ref -> stop
    dj2 = {"ckpts": {"C0_m20500": {"D": 0.25}, "C0_m21000": {"D": 0.1}}}
    r2 = read.p2r_drift_premise(stamps, drift_json=dj2)
    assert r2["stop"] is True
    # no drift_json at all -> never stops (nothing to compare)
    r3 = read.p2r_drift_premise(stamps, drift_json=None)
    assert r3["stop"] is False and r3["c0_bank"] == {}
    # a flat {name: D} drift_json (no "ckpts" wrapper) is accepted too
    r4 = read.p2r_drift_premise(stamps, drift_json={"C0_m20500": 0.25})
    assert r4["d_c0_506"] == pytest.approx(0.25) and r4["stop"] is True


# ================================================================================================ CLI
def test_cli_p2r_probe(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    ref_suite, probe_suite = _episode_pair(out, "P2R_PROBE_s18", SUITE)
    ref_fh, probe_fh = _episode_pair(out, "P2R_PROBE_s18", FH)
    ref_ws, probe_ws = _episode_pair(out, "P2R_PROBE_s18", WS)
    for row, p in ((SUITE, probe_suite), (FH, probe_fh), (WS, probe_ws)):
        (Path(p) / ("complete.json" if row == SUITE else "p2_complete.json")).write_text("{}")
    for row, p in ((SUITE, ref_suite), (FH, ref_fh), (WS, ref_ws)):
        (Path(p) / ("complete.json" if row == SUITE else "p2_complete.json")).write_text("{}")
    _record(out, "P2R_PROBE_s18", None)
    r = _load(out, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2r-probe"])
    rc = r.main()
    j = json.loads(capsys.readouterr().out)
    assert j["inertness"]["ok"] is True
    assert rc == 0


def test_cli_p2r_step1_pending_when_nothing_run(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _point(out, "REFS_S1_s18", mode="hold", full_fl=True)
    r = _load(out, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2r-step1"])
    assert r.main() == 0                                   # always exits 0 (CHARTER 12: every verdict stops and asks)
    j = json.loads(capsys.readouterr().out)
    assert j["heads"]["P2R_HOLD_s18"]["verdict"] == "PENDING"
    assert j["heads"]["P2R_ST2M_s18"]["verdict"] == "PENDING"
    assert j["heads"]["P2R_HOLD_s18"]["survives_tier1"] is None
    assert "no P2R_PROBE_s18 record" in j["drift_premise"]["missing"]


def test_cli_p2r_step1_still(tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _tree(out, {"P2R_HOLD_s18": {}})
    r = _load(out, monkeypatch)
    monkeypatch.setattr(r, "p2s_raw", _raws(r, {}))
    monkeypatch.setattr(sys, "argv", ["p2_read.py", "--p2r-step1"])
    assert r.main() == 0
    j = json.loads(capsys.readouterr().out)
    assert j["heads"]["P2R_HOLD_s18"]["verdict"] == "STILL"
    assert j["heads"]["P2R_HOLD_s18"]["survives_tier1"] is True
    assert j["heads"]["P2R_ST2M_s18"]["verdict"] == "PENDING"      # not run in this tree


# ================================================================================================ p2r_drift.py
def _tiny_agent_yaml(path: Path, **actor_kw):
    import yaml                                            # noqa: PLC0415
    actor = {"num_prop": 4, "num_scan": 3, "num_priv_explicit": 2, "num_priv_latent": 2}
    actor.update(actor_kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"policy": {"actor": actor, "tanh_encoder_output": False, "noise_std_type": "soft"}}, open(path, "w"))


def test_actor_widths_reads_agent_yaml(drift, tmp_path):
    p = tmp_path / "params" / "agent.yaml"
    _tiny_agent_yaml(p)
    w = drift.actor_widths(p)
    assert w == {"num_prop": 4, "num_scan": 3, "num_priv_explicit": 2, "num_priv_latent": 2,
                "tanh_encoder_output": False, "noise_std_type": "soft"}


def test_drift_zero_for_identical_checkpoints_and_positive_for_a_perturbed_one(drift):
    import torch                                           # noqa: PLC0415
    w = {"num_prop": 4, "num_scan": 3, "num_priv_explicit": 2, "num_priv_latent": 2, "tanh_encoder_output": False}
    n = 16
    obs = torch.randn(n, 4 + 3 + 2 + 2 + 20)
    bank = {"obs": obs, "s1_sigma": torch.full((2,), 0.4),
           "hold_label": np.array((["fwd"] * (n // 2)) + (["diag"] * (n - n // 2)))}
    sd_s1 = drift._tiny_sd(seed=7, scale=1.0)
    sd_same = drift._tiny_sd(seed=7, scale=1.0)
    sd_pert = drift._tiny_sd(seed=7, scale=1.5)
    r_same = drift.drift(bank, sd_same, sd_s1, w)
    assert r_same["D"] == 0.0 and r_same["n"] == n
    r_pert = drift.drift(bank, sd_pert, sd_s1, w)
    assert r_pert["D"] > 0.0
    assert set(r_pert["by_label"]) == {"fwd", "diag"}


def test_drift_self_test_passes(drift):
    assert drift._self_test() == 0


def test_cli_end_to_end_json_and_errors(drift, tmp_path, capsys):
    agent = tmp_path / "params" / "agent.yaml"
    _tiny_agent_yaml(agent)
    sd_ref = drift._tiny_sd(seed=3, scale=1.0)
    sd_other = drift._tiny_sd(seed=3, scale=1.2)
    import torch                                           # noqa: PLC0415
    torch.save({"model_state_dict": sd_ref}, tmp_path / "ref.pt")
    torch.save({"model_state_dict": sd_other}, tmp_path / "other.pt")
    n = 8
    obs = np.random.default_rng(0).normal(size=(n, 4 + 3 + 2 + 2 + 20)).astype(np.float32)
    np.savez(tmp_path / "bank.npz", obs=obs, s1_sigma=np.full(2, 0.3, dtype=np.float32),
            hold_label=np.array(["fwd"] * n), category=np.array(["fwd"] * n))
    out_json = tmp_path / "out.json"
    rc = drift.main(["--bank", str(tmp_path / "bank.npz"), "--ref", str(tmp_path / "ref.pt"), "--ckpt",
                     f"OTHER={tmp_path / 'other.pt'}", "--json", str(out_json)])
    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["ckpts"]["OTHER"]["D"] > 0.0
    capsys.readouterr()
    with pytest.raises(SystemExit):
        drift.main(["--bank", str(tmp_path / "bank.npz"), "--ref", str(tmp_path / "ref.pt"), "--ckpt", "BAD"])


def test_parse_ckpt_args(drift):
    assert drift.parse_ckpt_args(["A=/x/a.pt", "B=/x/b.pt"]) == {"A": Path("/x/a.pt"), "B": Path("/x/b.pt")}
    with pytest.raises(ValueError):
        drift.parse_ckpt_args(["noequals"])
    with pytest.raises(ValueError):
        drift.parse_ckpt_args(["=novalue"])
