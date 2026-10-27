"""Stage p2r Z2d seed replicates (P2R_Z2D_REP_PLAN.md step 1; CHARTER 12): the read / live tooling for s16 / s17.
CPU only: nothing here starts Isaac/Kit or a GPU; the real-record checks are read-only. NOTE(p2r Z2d rep, 2026-10-11).

* ``scripts/p2_read.py``: the seed parsed from the head name; ``p2r_z2_report`` at s16 loads every per-seed point at
  s16 (no ``KeyError: 16``), labels Z1 / HOLD / probe / C0R ``cross-seed (s18)`` and adds the SAME-seed HOLD; the
  pre-registered ``replicate_grade`` (pass / each failing clause / PENDING), calibrated on the real s18 records (Z2d
  passes, Z1 fails the statue clauses against P2R_HOLD_s18); ``--p2r-graft-identity NAME SEED`` PASS / FAIL /
  PENDING on fabricated stamp trees with real raw npz;
* ``scripts/p2_live.py``: the per-seed Z2D dose file (``p2r_z2d_dose_s<seed>.json`` off s18), the dose-checkpoint
  refusal (``bc.checkpoint`` must be the chain's fork) in the preflight and the +200 judge, and the real
  ``P2__Z2dR1_s18`` / ``p2r_z2d_dose.json`` pair still accepted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
Z2_READS_TESTS = REPO / "tests/unit/test_omni_p2r_z2_reads.py"
Z2D_TOOLS_TESTS = REPO / "tests/unit/test_omni_p2r_z2d_tools.py"
Z2C_TOOLS_TESTS = REPO / "tests/unit/test_omni_p2r_z2c_tools.py"
REAL_READ = CAMP / "p2r_z2d_read.json"
REAL_DOSE = CAMP / "p2r_z2d_dose.json"
SUITE, FT, WS, FH = "suite__omni", "field_turn__omni", "walkstop__omni", "field_hard__omni"
STILL_V = {"verdict": "STILL", "stand": {"outcome": "STILL"}, "why": []}
_N = [0]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def read():
    return _load(SCRIPTS / "p2_read.py", "_t_z2drep_read")


@pytest.fixture(scope="module")
def live():
    return _load(SCRIPTS / "p2_live.py", "_t_z2drep_live")


@pytest.fixture(scope="module")
def T():
    """The Z2 read tests' helpers (synthetic walkstop episodes, raw stamps, light points, Z2 summaries)."""
    return _load(Z2_READS_TESTS, "_t_z2drep_from_z2_reads")


@pytest.fixture(scope="module")
def Z():
    return _load(Z2C_TOOLS_TESTS, "_t_z2drep_from_z2c_tools")


@pytest.fixture(scope="module")
def D():
    return _load(Z2D_TOOLS_TESTS, "_t_z2drep_from_z2d_tools")


def _read_at(out: Path, monkeypatch):
    monkeypatch.setenv("P2_OUT", str(out))
    _N[0] += 1
    return _load(SCRIPTS / "p2_read.py", f"_t_z2drep_read_{_N[0]}")


# ===================================================================================== seed parsing
def test_seed_of_name(read, live):
    assert read.seed_of_name("P2__Z2d_s16") == 16 and read.seed_of_name("P2__Z2dR1_s18") == 18
    assert read.seed_of_name("GRAFT_S1_s17") == 17 and read.seed_of_name("P2R_HOLD_s16") == 16
    assert read.seed_of_name("P2__Z2d") is None and read.seed_of_name(None) is None
    assert read.seed_of_name("P2__Z2d_s16x") is None
    assert live.seed_of("P2__Z2d_s16") == 16 and live.seed_of("P2__Z2dR1_s18") == 18
    assert read.z2d_hold_name(16, env={}) == "P2R_HOLD_s16"
    assert read.z2d_hold_name(17, env={"P2R_HOLD_S17": "P2R_HOLDR1_s17"}) == "P2R_HOLDR1_s17"
    assert read.z2d_reference_label("P2__Z1_s18", 16) == {"name": "P2__Z1_s18", "seed": 18,
                                                          "label": "cross-seed (s18)"}
    assert read.z2d_reference_label("REFS_S1_s16", 16)["label"] == "same seed"


# ===================================================================================== the replicate rule (pure)
def _ws(yaw=(1.98, 1.89), six=(0.58, 0.52), falls=(0.0, 0.0)):
    return {"walkstop": {s: {"hip_yaw_mean_deg": yaw[i], "six_down_stand_share": six[i], "falls_per_300": falls[i]}
                         for i, s in enumerate((1, 2))}}


HOLD18 = _ws(yaw=(5.96, 6.14), six=(0.35, 0.34))


def test_replicate_grade_pass(read):
    r = read.z2d_replicate_grade(STILL_V, _ws(), HOLD18, hold_name="P2R_HOLD_s18", seed=18)
    assert r["replicate_grade"] is True and r["replicate_pending"] == [], r
    cl = r["replicate_clauses"]
    assert len(cl) == 5
    yaw = next(v for k, v in cl.items() if "hip-yaw" in k)
    assert yaw[0] == pytest.approx(1.935) and yaw[1] == pytest.approx(0.5 * 6.05) and yaw[2] is True
    six = next(v for k, v in cl.items() if "six-down" in k)
    assert six[0] == pytest.approx(0.55) and six[1] == pytest.approx(0.345) and six[2] is True
    assert any("REFS_S1_s18" in k for k in cl)                      # ladder_ok vs the seed's own REFS


@pytest.mark.parametrize("case,frag", [
    ("cost", "ladder_ok"), ("partial", "ladder_ok"), ("invalid", "ladder_ok"),
    ("falls_es2", "falls per 300 es2"), ("yaw", "hip-yaw"), ("six_equal", "six-down"),
    ("yaw_none", "hip-yaw")])
def test_replicate_grade_each_failure(read, case, frag):
    v, z2 = STILL_V, _ws()
    if case == "cost":
        v = {"verdict": "COST", "stand": {"outcome": "STILL"}}
    elif case == "partial":
        v = {"verdict": "STILL", "stand": {"outcome": "PARTIAL"}}
    elif case == "invalid":
        v = {"verdict": "INVALID EVAL", "stand": {}}
    elif case == "falls_es2":
        z2 = _ws(falls=(0.0, 1.01))
    elif case == "yaw":
        z2 = _ws(yaw=(3.1, 3.0))                                     # 3.05 > 0.5 x 6.05
    elif case == "six_equal":
        z2 = _ws(six=(0.35, 0.34))                                   # equal to HOLD's: not strictly greater
    elif case == "yaw_none":
        z2 = _ws()
        z2["walkstop"][2]["hip_yaw_mean_deg"] = None                 # a present read with no value FAILS
    r = read.z2d_replicate_grade(v, z2, HOLD18, hold_name="P2R_HOLD_s16", seed=16)
    assert r["replicate_grade"] is False, (case, r)
    bad = [k for k, (_v, _t, ok) in r["replicate_clauses"].items() if not ok]
    assert len(bad) == 1 and frag in bad[0], (case, bad)


def test_replicate_grade_pending(read):
    def g(**kw):
        a = dict(v=STILL_V, z2=_ws(), hold_z2=HOLD18, hold_name="P2R_HOLD_s16", seed=16)
        a.update(kw)
        return read.z2d_replicate_grade(a.pop("v"), a.pop("z2"), a.pop("hold_z2"), **a)
    r = g(hold_z2=None)                                             # the same-seed HOLD never ran
    assert r["replicate_grade"] is None and any("P2R_HOLD_s16" in p for p in r["replicate_pending"])
    hz = _ws(yaw=(5.96, 6.14), six=(0.35, 0.34))
    del hz["walkstop"][2]
    assert g(hold_z2=hz)["replicate_grade"] is None                 # HOLD es2 missing
    r = g(hold_fails=["walkstop__omni@1: p2r_mode.variant 'st2' != 'hold'"])
    assert r["replicate_grade"] is None and any("variant" in p for p in r["replicate_pending"])
    assert g(v={"verdict": "PENDING", "why": ["REFS_S1_s16 not run"]})["replicate_grade"] is None
    assert g(v=None)["replicate_grade"] is None
    z = _ws()
    del z["walkstop"][1]
    r = g(z2=z)
    assert r["replicate_grade"] is None and any("walkstop es1" in p for p in r["replicate_pending"])
    assert g(z2=None)["replicate_grade"] is None
    # str seed keys (a report read back from JSON) read the same
    js = json.loads(json.dumps(_ws()))
    assert g(z2=js, hold_z2=json.loads(json.dumps(HOLD18)))["replicate_grade"] is True


def test_replicate_grade_s18_calibration_on_the_recorded_baselines(read):
    """P2R_Z2D_REP_PLAN.md "Calibrated at s18": Z2d (P2__Z2dR1_s18) passes, Z1 (P2__Z1_s18) fails the statue clauses
    against P2R_HOLD_s18 -- on p2r_z2d_read.json's baselines (the s18 read of record)."""
    if not REAL_READ.is_file():
        pytest.skip("no p2r_z2d_read.json")
    d = json.loads(REAL_READ.read_text())
    b = d["baselines"]
    assert (b["Z2"]["name"], b["Z1"]["name"], b["HOLD"]["name"]) == ("P2__Z2dR1_s18", "P2__Z1_s18", "P2R_HOLD_s18")
    z2d = read.z2d_replicate_grade(d["verdict"], b["Z2"], b["HOLD"], hold_name="P2R_HOLD_s18", seed=18)
    assert z2d["replicate_grade"] is True, z2d
    cl = z2d["replicate_clauses"]
    yaw = next(v for k, v in cl.items() if "hip-yaw" in k)
    six = next(v for k, v in cl.items() if "six-down" in k)
    assert yaw[0] == pytest.approx(1.93, abs=0.01) and yaw[1] == pytest.approx(3.02, abs=0.01)
    assert six[0] == pytest.approx(0.55, abs=0.01) and six[1] == pytest.approx(0.34, abs=0.01)
    z1 = read.z2d_replicate_grade(STILL_V, b["Z1"], b["HOLD"], hold_name="P2R_HOLD_s18", seed=18)   # Z1 read STILL
    assert z1["replicate_grade"] is False
    bad = sorted(k for k, (_v, _t, ok) in z1["replicate_clauses"].items() if not ok)
    assert len(bad) == 2 and any("hip-yaw" in k for k in bad) and any("six-down" in k for k in bad), bad
    v = {k: v for k, v in z1["replicate_clauses"].items()}
    assert next(x for k, x in v.items() if "hip-yaw" in k)[0] == pytest.approx(5.91, abs=0.01)
    assert next(x for k, x in v.items() if "six-down" in k)[0] == pytest.approx(0.23, abs=0.01)


def test_replicate_grade_s18_calibration_on_the_real_stamps(read):
    """The same calibration straight off the real walkstop raws (newest complete stamps; read-only), when reachable."""
    names = ("P2__Z2dR1_s18", "P2__Z1_s18", "P2R_HOLD_s18")
    stamps = {nm: {s: read.newest_stamp(nm, WS, s) for s in (1, 2)} for nm in names}
    if not all(all(v.values()) for v in stamps.values()):
        pytest.skip("the s18 walkstop stamps are not reachable")
    z2 = {nm: {"walkstop": {s: read.z2_summary(st[s]) for s in (1, 2)}} for nm, st in stamps.items()}
    hold = z2["P2R_HOLD_s18"]
    assert read.z2d_replicate_grade(STILL_V, z2["P2__Z2dR1_s18"], hold, seed=18)["replicate_grade"] is True
    assert read.z2d_replicate_grade(STILL_V, z2["P2__Z1_s18"], hold, seed=18)["replicate_grade"] is False


# ===================================================================================== p2r_z2_report at seed 16
def _setup_report(read, T, monkeypatch, *, name="P2__Z2d_s16", z2=None, hold16=None, ladder=None,
                  seed18_attr=False):
    """``p2r_z2_report`` on synthetic inputs: the REAL load_p2r_z / load_p2s / p2s_verdicts path with the IO stubbed
    (records, points, light points), so a seed that is not threaded through fails the way the s16 crash did."""
    seen = {}
    real_load_p2s = read.load_p2s

    def load_p2s(with_tb=True, seeds=read.SEEDS, names=None):
        seen["seeds"] = list(seeds)
        return real_load_p2s(with_tb=with_tb, seeds=seeds, names=names)
    monkeypatch.setattr(read, "load_p2s", load_p2s)
    monkeypatch.setattr(read, "_anchors_yaml", lambda: {16: {"name": "ANCW_s16"}, 17: {"name": "ANCW_s17"},
                                                        18: {"name": "ANCW_s18"}})
    monkeypatch.setattr(read, "load_point", lambda nm, **k: {"name": nm, "stamps": {}})
    monkeypatch.setattr(read, "load_p2s_point", lambda nm, **k: {"name": nm, "stamps": {f"{WS}@1": "/w1"}})
    # the seed's p2a head is not recorded: p2s_verdicts reads it at THAT seed (PENDING), never p2a[18]
    monkeypatch.setattr(read, "_rec", lambda nm: None if nm.startswith("P2__S1_") else {"driver_stamp": "/d",
                                                                                       "run_dir": "/r"})
    monkeypatch.setattr(read, "bands", lambda *a: {})
    if seed18_attr:
        monkeypatch.setattr(read, "p2s_attribution", lambda *a: {"status": "LEVER", "control": "C0R_s18"})
    monkeypatch.setattr(read, "z2_suite_stamp", lambda nm, lrout=None: "/s")
    monkeypatch.setattr(read, "z_raw_bit_reasons", lambda pt, lag=1: ([], 0))
    monkeypatch.setattr(read, "z2_reads", lambda stamps: z2 or T._z2(hip_yaw_mean_deg=1.9, six_down_stand_share=0.55))
    if ladder is not None:
        monkeypatch.setattr(read, "p2r_z2_verdicts", lambda res: {name: ladder})
    monkeypatch.setattr(read, "p2r_mode_fails", lambda pt, tag, ref_sha, stamps=(): [])
    lights = {nm: T._light(read, nm) for nm in ("P2R_RST_s18", "P2__Z1_s18", "P2R_HOLD_s18", "REFS_S1_s16",
                                                "REFS_S1_s18", "P2__Z2dR1_s18")}
    lights["P2R_HOLD_s16"] = T._light(read, "P2R_HOLD_s16", z2=hold16 or T._z2(hip_yaw_mean_deg=6.0,
                                                                               six_down_stand_share=0.34))
    for lp in lights.values():
        for s in (1, 2, 3, 4):
            lp["field"][f"{FH}@{s}"] = {"stand": None, "comp": {"rate": 0.82, "n": 100}}
    asked = []

    def light(nm, lrout=None):
        asked.append(nm)
        return lights[nm]
    monkeypatch.setattr(read, "z2_light_point", light)
    return seen, asked


def test_p2r_z2_report_at_seed16_loads_seed16_and_labels_cross_seed_refs(read, T, monkeypatch):
    seen, asked = _setup_report(read, T, monkeypatch)
    out = read.p2r_z2_report(with_tb=False, env={}, name="P2__Z2d_s16")
    assert out["seed"] == 16 and seen["seeds"] == [16]               # the seed came from the name
    assert "REFS_S1_s16" in asked and "P2R_HOLD_s16" in asked and "REFS_S1_s18" not in asked
    v = out["verdict"]
    assert v["verdict"] == "PENDING" and any("P2__S1_s16" in w for w in v["why"]), v   # p2a at 16, no KeyError
    refs = out["references"]
    for lab in ("probe", "Z1", "HOLD", "C0R"):
        assert refs[lab]["seed"] == 18 and refs[lab]["label"] == "cross-seed (s18)", (lab, refs[lab])
    assert refs["REFS"] == {"name": "REFS_S1_s16", "seed": 16, "label": "same seed"}
    assert refs["HOLD_same_seed"]["name"] == "P2R_HOLD_s16" and refs["HOLD_same_seed"]["label"] == "same seed"
    assert set(out["baselines"]) == {"Z2", "probe", "Z1", "HOLD", "REFS", "HOLD_s16"}
    assert out["baselines"]["REFS"]["name"] == "REFS_S1_s16"
    assert out["attribution"]["status"] == "PENDING" and "s18 only" in out["attribution"]["note"]
    assert out["adoption_grade"] is False
    assert any("P2__Z1_s18" in n and "cross-seed (s18)" in n for n in out["z2_gates"]["notes"])
    assert out["walls_vs_z1"]["z2d_s18"] == (0.82, 400)
    # the ladder is PENDING (no p2a record at s16 here): the replicate grade waits on it
    assert out["replicate_grade"] is None and any("ladder" in p for p in out["replicate_pending"])
    md = read.p2r_z2_markdown(out)
    assert "cross-seed (s18)" in md and "Replicate grade (s16, vs P2R_HOLD_s16): None" in md


def test_p2r_z2_report_at_seed16_replicate_grade_uses_the_same_seed_hold(read, T, monkeypatch):
    _setup_report(read, T, monkeypatch, ladder=STILL_V)
    out = read.p2r_z2_report(with_tb=False, env={}, name="P2__Z2d_s16")
    assert out["replicate_grade"] is True and out["replicate_hold"] == "P2R_HOLD_s16", out["replicate_clauses"]
    assert out["adoption_grade"] is False                            # attribution PENDING off s18, by construction
    # the same head against a same-seed HOLD that is already as still: the statue clauses fail
    _setup_report(read, T, monkeypatch, ladder=STILL_V,
                  hold16=T._z2(hip_yaw_mean_deg=3.0, six_down_stand_share=0.60))
    out = read.p2r_z2_report(with_tb=False, env={}, name="P2__Z2d_s16")
    assert out["replicate_grade"] is False
    assert "**FAIL**" in read.p2r_z2_markdown(out)


def test_p2r_z2_report_at_s18_is_unchanged_but_for_the_replicate_fields(read, T, monkeypatch):
    seen, asked = _setup_report(read, T, monkeypatch, name="P2__Z2dR1_s18", ladder=STILL_V, seed18_attr=True)
    out = read.p2r_z2_report(with_tb=False, env={}, name="P2__Z2dR1_s18")
    assert out["seed"] == 18 and seen["seeds"] == [18]
    assert set(out["baselines"]) == {"Z2", "probe", "Z1", "HOLD", "REFS"}          # HOLD is the same-seed HOLD
    assert all(r["label"] == "same seed" for r in out["references"].values())
    assert "z2d_s18" not in out["walls_vs_z1"] and "P2__Z2dR1_s18" not in asked
    assert out["replicate_hold"] == "P2R_HOLD_s18" and out["replicate_grade"] is not None


# ===================================================================================== --p2r-graft-identity
def _id_tree(T, lr: Path, seed: int, *, skip=(), graft_bit=True, perturb=None):
    """REFS_S1_s<seed> and GRAFT_S1_s<seed> stamps on suite / field_turn / walkstop es1 with real raw npz: identical
    raws (REFS's clock_obs zero, the graft's the stand bit), unless ``graft_bit`` is False (the graft's bit zero)
    or ``perturb`` names a row whose graft actions differ."""
    seg = T._walkstop_seg(walk=30, stand=60)
    eps = [T._stand_episode(seg, (1, 3, 5), settle_at=20) for _ in range(2)]
    for nm in (f"REFS_S1_s{seed}", f"GRAFT_S1_s{seed}"):
        for row in (SUITE, FT, WS):
            if nm.startswith("GRAFT") and row in skip:
                continue
            e = [dict(d) for d in eps]
            if nm.startswith("GRAFT") and row == perturb:
                e[1]["actions"] = e[1]["actions"] + 1e-6
            root = T._write_raw_stamp(lr / nm / row / "seed001" / "2026-10-12_00-00-00", e, T.WS_LABELS,
                                      T.WS_SPEEDS)
            if nm.startswith("REFS") or not graft_bit:
                for f in sorted((root / "raw").glob("episode_*.npz")):
                    z = dict(np.load(f))
                    z["clock_obs"] = np.zeros_like(z["clock_obs"])
                    np.savez(f, **z)


def _cli(r, monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["p2_read.py", *argv])
    rc = r.main()
    return rc, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_graft_identity_pass(T, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _id_tree(T, out / "lr", 16)
    r = _read_at(out, monkeypatch)
    rc, j = _cli(r, monkeypatch, capsys, "--p2r-graft-identity", "GRAFT_S1_s16", "16")
    assert rc == 0 and j["ok"] is True and j["verdict"] == "PASS", j
    assert set(j["rows"]) == {SUITE, FT, WS}
    for row in (SUITE, FT, WS):
        assert j["rows"][row]["diff"] == [] and j["rows"][row]["stand_bit_ok"] is True
        assert "/REFS_S1_s16/" in j["rows"][row]["refs"] and "/GRAFT_S1_s16/" in j["rows"][row]["graft"]
    assert len(j["stamps_to_write"]) == 3                            # outside the repo root: absolute paths
    # under the repo root: repo-relative, graft.json's identity_gpu_stamps form
    monkeypatch.setattr(r, "R", tmp_path)
    j2 = r.p2r_graft_identity("GRAFT_S1_s16", 16)
    assert j2["stamps_to_write"][0].startswith("out/lr/GRAFT_S1_s16/suite__omni/seed001/")
    assert not any(p.name.endswith(".json") and "graft" in p.name for p in tmp_path.rglob("*"))   # nothing written


@pytest.mark.parametrize("case", ["actions_differ", "bit_off", "wrong_seed"])
def test_graft_identity_fail(T, tmp_path, monkeypatch, capsys, case):
    out = tmp_path / "out"
    _id_tree(T, out / "lr", 16, perturb=WS if case == "actions_differ" else None, graft_bit=case != "bit_off")
    r = _read_at(out, monkeypatch)
    nm = "GRAFT_S1_s17" if case == "wrong_seed" else "GRAFT_S1_s16"
    rc, j = _cli(r, monkeypatch, capsys, "--p2r-graft-identity", nm, "16")
    assert rc == 4 and j["ok"] is False and j["verdict"] == "FAIL" and j["stamps_to_write"] == [], j
    if case == "actions_differ":
        assert j["rows"][WS]["diff"] and "actions" in j["rows"][WS]["diff"][0]
        assert j["rows"][SUITE]["diff"] == []
    elif case == "bit_off":
        assert all(j["rows"][row]["stand_bit_ok"] is False for row in (SUITE, FT, WS))
        assert all(j["rows"][row]["diff"] == [] for row in (SUITE, FT, WS))   # clock_obs left out of the diff
    else:
        assert any("not a seed-16 name" in f for f in j["fails"])


def test_graft_identity_pending(T, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    _id_tree(T, out / "lr", 17, skip=(FT,))
    r = _read_at(out, monkeypatch)
    rc, j = _cli(r, monkeypatch, capsys, "--p2r-graft-identity", "GRAFT_S1_s17", "17")
    assert rc == 2 and j["verdict"] == "PENDING" and j["ok"] is False and j["stamps_to_write"] == [], j
    assert j["rows"][FT]["graft"] is None and j["rows"][SUITE]["diff"] == []
    rc, j = _cli(r, monkeypatch, capsys, "--p2r-graft-identity", "GRAFT_S1_s17", "x")
    assert rc == 4 and j["verdict"] == "FAIL"


def test_bitwise_diff_skip_is_keyword_only_in_effect(read, T, tmp_path):
    _id_tree(T, tmp_path, 16)
    a = read.newest_stamp("REFS_S1_s16", WS, 1, tmp_path)
    b = read.newest_stamp("GRAFT_S1_s16", WS, 1, tmp_path)
    assert read.p2r_bitwise_diff(a, b) and all("clock_obs" in x for x in read.p2r_bitwise_diff(a, b))
    assert read.p2r_bitwise_diff(a, b, skip=("clock_obs",)) == []


# ===================================================================================== p2_live: the per-seed dose
def test_z2_dose_default_per_seed(live):
    assert live.z2_dose_default("Z2D", 16) == CAMP / "p2r_z2d_dose_s16.json"
    assert live.z2_dose_default("Z2D", 17) == CAMP / "p2r_z2d_dose_s17.json"
    assert live.z2_dose_default("Z2D", 18) == live.z2_dose_default("Z2D") == live.P2R_Z2D_DOSE_JSON
    assert live.z2_dose_default("Z2D", None) == CAMP / "p2r_z2d_dose.json"
    assert live.z2_dose_default("Z2C", 16) == live.z2_dose_default("Z2", 17) == live.P2R_Z2_DOSE_JSON
    assert live.z2_dose_default(live.z_arm("P2__Z2dR1_s18"), live.seed_of("P2__Z2dR1_s18")).name == \
        "p2r_z2d_dose.json"
    assert live.z2_dose_default(live.z_arm("P2__Z2d_s16"), live.seed_of("P2__Z2d_s16")).name == \
        "p2r_z2d_dose_s16.json"


def test_judge_z2d_dose_fork(live):
    fork = "parkour/outputs/omni_plan/phase2/graft/S1_s16_c15/model_19994.pt"
    absf = str(live.R / fork)
    ok = {"checkpoint": absf, "bc": {"checkpoint": absf}}
    assert live.judge_z2d_dose_fork(ok, fork) == []
    assert live.judge_z2d_dose_fork({"bc": {"checkpoint": fork}}, absf) == []          # relative vs absolute
    s18 = str(live.R / "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt")
    f = live.judge_z2d_dose_fork({"checkpoint": s18, "bc": {"checkpoint": s18}}, fork, "p2r_z2d_dose_s16.json")
    assert len(f) == 2 and all("not the chain's fork" in x and "p2r_z2d_dose_s16.json" in x for x in f)
    f = live.judge_z2d_dose_fork({"checkpoint": s18, "bc": {"checkpoint": absf}}, fork)   # top-level disagrees
    assert len(f) == 1 and "'checkpoint'" not in f[0] and "checkpoint" in f[0]
    assert any("records no bc.checkpoint" in x for x in live.judge_z2d_dose_fork({"bc": {}}, fork))
    assert live.judge_z2d_dose_fork(None, fork) == []                                   # the band check refuses it
    assert any("fork checkpoint is unknown" in x for x in live.judge_z2d_dose_fork(ok, None))


def test_real_s18_dose_file_passes_for_its_chain(live):
    cands = yaml.safe_load(open(CAMP / "candidates.yaml"))
    if "P2__Z2dR1_s18" not in (cands.get("chains") or {}) or not REAL_DOSE.is_file():
        pytest.skip("no P2__Z2dR1_s18 chain / p2r_z2d_dose.json")
    arm, seed = live.z_arm("P2__Z2dR1_s18"), live.seed_of("P2__Z2dR1_s18")
    assert live.z2_dose_default(arm, seed) == REAL_DOSE
    fork = live.chain_source_ckpt(cands, "P2__Z2dR1_s18")
    doc = json.loads(REAL_DOSE.read_text())
    # the dose file is a RECORD: its checkpoint paths name the checkout it was computed in (absolute), while the
    # chain's fork is repo-relative and resolves against THIS checkout -- map the recording root here (the identity
    # on the recording checkout), so the judge is checked on the same graft wherever the repo lives
    rec_root = "/home/nickmagus/krabby/krabby-research"
    for blk in (doc, doc.get("bc") or {}):
        ck = blk.get("checkpoint")
        if isinstance(ck, str) and ck.startswith(rec_root + "/"):
            blk["checkpoint"] = str(live.R) + ck[len(rec_root):]
    assert live.judge_z2d_dose_fork(doc, fork) == []
    ov = live.chain_overlay(cands, "P2__Z2dR1_s18")
    assert live.judge_z2d_dose_fork(doc, ov[live.P2R_ANCHOR_CKPT]) == []                # the +200 judge's input


def _pf_setup(D, Z, live, tmp_path, monkeypatch, dose_ckpt="fork", write=("s16",)):
    """A fabricated campaign: candidates.yaml with P2__Z2d_s16 forking a tmp graft (sidecar with identity), the
    chain overlay = the Z2d fixture's (anchor = that graft), and the dose files ``write`` in the tmp CAMPAIGN."""
    banks = D._banks(tmp_path)
    fork = tmp_path / "graft" / "S1_s16_c15" / "model_19994.pt"
    fork.parent.mkdir(parents=True)
    fork.write_bytes(b"pt")
    fork.with_suffix(".json").write_text(json.dumps({"identity_cpu": True, "identity_gpu_stamps": ["x"]}))
    ov = D._ov_d(Z, live, banks, KRABBY_ANCHOR_CKPT=str(fork))
    cands = {"chains": {"P2__Z2d_s16": {"stages": [{"resume_from": {"checkpoint": str(fork)}, "overlay": ov}]}}}
    (tmp_path / "candidates.yaml").write_text(yaml.safe_dump(cands))
    ck = {"fork": str(fork), "s18": str(D.GRAFT_S18), "none": None}[dose_ckpt]
    doc = D._dose_d(Z, live, banks)
    doc["bc"]["checkpoint"] = ck
    if ck is None:
        del doc["bc"]["checkpoint"]
    for w in write:
        (tmp_path / ("p2r_z2d_dose.json" if w == "s18" else f"p2r_z2d_dose_{w}.json")).write_text(json.dumps(doc))
    monkeypatch.setattr(live, "CAMPAIGN", tmp_path)
    monkeypatch.setattr(live, "emitted_metric_keys", lambda root, keys: {k: ["x.py"] for k in keys})
    return argparse.Namespace(candidates=tmp_path / "candidates.yaml", chain="P2__Z2d_s16", code_root=str(tmp_path),
                              replay=None, arms=tmp_path, z2_dose=None)


def test_preflight_reads_the_seeds_own_dose_and_passes(D, Z, live, tmp_path, monkeypatch):
    rc, res = live.preflight(_pf_setup(D, Z, live, tmp_path, monkeypatch))
    assert rc == live.RC_PASS, res["fails"]


@pytest.mark.parametrize("case", ["s18_ckpt", "no_ckpt", "only_s18_file"])
def test_preflight_refuses_a_dose_not_read_on_the_fork(D, Z, live, tmp_path, monkeypatch, case):
    if case == "s18_ckpt":
        a = _pf_setup(D, Z, live, tmp_path, monkeypatch, dose_ckpt="s18")
        frag = "is not the chain's fork checkpoint"
    elif case == "no_ckpt":
        a = _pf_setup(D, Z, live, tmp_path, monkeypatch, dose_ckpt="none")
        frag = "records no bc.checkpoint"
    else:                                            # the s18 file present, the seed's own missing: never reused
        a = _pf_setup(D, Z, live, tmp_path, monkeypatch, write=("s18",))
        frag = "p2r_z2d_dose_s16.json"
    rc, res = live.preflight(a)
    assert rc == live.RC_FAIL and any(frag in f for f in res["fails"]), res["fails"]


def test_wiring_200_refuses_a_dose_not_read_on_the_fork(D, Z, live, tmp_path):
    banks = D._banks(tmp_path)
    ov, ser = D._ov_d(Z, live, banks), D._ser_d(Z, live)
    assert D._wire(live, "Z2D", ov, ser, D._dose_d(Z, live, banks))["rc"] == live.RC_PASS
    bad = D._dose_d(Z, live, banks)
    bad["bc"]["checkpoint"] = str(tmp_path / "graft" / "S1_s16_c15" / "model_19994.pt")
    w = dict(live._facts()["weights"], reward_action_rate=-0.05)
    audit = [("overlay KRABBY_TERRAIN_PROMOTE=0.75:0.4 in params/env.yaml", True)]
    r = live.judge_wiring("Z2D", ov, live._facts(weights=w), ser, D.F0, audit, None, z2_dose=bad,
                          z2_dose_name="p2r_z2d_dose_s16.json")
    assert r["rc"] == live.RC_FAIL
    assert any("p2r_z2d_dose_s16.json's bc.checkpoint" in f and "not the chain's fork" in f for f in r["fails"])


def test_check_and_preflight_thread_the_chain_seed(live):
    src = (SCRIPTS / "p2_live.py").read_text()
    assert src.count('getattr(a, "z2_dose", None) or z2_dose_default(arm, seed_of(a.chain))') == 2
    assert "seed=seed_of(a.chain))" in src.split("def preflight(a)")[1].split("def anchor_reference_sidecar")[0]
    assert "z2_dose_default(arm, seed).name" in src.split("def judge_preflight")[1].split("def ")[0]
