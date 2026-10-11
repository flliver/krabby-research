"""Stage p2r Z2d (P2R_Z2D_PLAN.md step 1, the CAMPAIGN-SCRIPT half; CHARTER 12): the tools of the stand-row input
mask run. CPU only: nothing here starts Isaac/Kit or a GPU. NOTE(p2r Z2d, 2026-10-11).

* ``scripts/p2r_z2c_gate.py`` (+ ``p2r_z2_fidelity.py``): the mask read off ``<run dir>/params/env.yaml`` (an omni obs
  term's ``stand_mask: true``) or forced (``--stand-mask``); when masked the bank reaches BOTH scored paths through
  ``stand_mask_obs`` (after R); the JSON records ``masked`` / ``mask_source``; masking a bank whose columns 65..88
  already hold the constants changes nothing; the LIFT / HOME lines and ``--report-bank`` are REPORTED, never gated
  (an unreadable report bank never turns a verdict into an error);
* ``scripts/p2r_z2_dose.py --bc-stand-mask``: the bank rows reach the fork's forward masked, ``bc.stand_mask``
  recorded (absent when off), refused without banks or onto the Z2 / Z2c file, a new output dir created;
* ``scripts/p2_live.py``: ``z_arm`` reads P2__Z2d_s18 as Z2D and the +200 wiring dispatches it to the Z2D judge (Z2c's
  checks + Metrics/stand_mask_frac in the prior-rows band + the mask knob + the dose file's ``bc.stand_mask``), never
  to the generic Z branch; an unknown lettered Z arm (Z2E) is refused at +200 and in the preflight; the Z2d
  preflight; Z2D's own default dose file;
* ``scripts/run_arm.sh``: ``is_z2d`` (and ``is_z2c`` answering for it), the launch refusal and the model_20500 gate's
  rc-3 kill path for a Z2d chain;
* ``scripts/p2_read.py``: ``head_masked`` from the chain overlay, the ``want_mask`` INVALID rules (the probe path's
  default adds none), the widened Z2 INVALID filter, ``z2_suite_stamp`` skipping the driver stamp for a masked head;
* ``scripts/p2r_z2c_observe_read.py``: an unmasked observe bank REFUSED under ``--stand-mask`` (exit 3), a masked one
  scored through the mask with ``masked`` in each row.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
TASKS = REPO / "parkour/parkour_tasks/parkour_tasks"
CAMP = TASKS / "crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
RESTANCE = TASKS / "crab_hex_omni_task/mdp/crab_hex_omni_restance.py"
RUN_ARM = SCRIPTS / "run_arm.sh"
Z2C_TESTS = REPO / "tests/unit/test_omni_p2r_z2c_tools.py"
Z2_PROBE_TESTS = REPO / "tests/unit/test_omni_p2r_z2_probe.py"
ENV_Z2 = {"KRABBY_OMNI_CLOCK_HIDE": "1", "KRABBY_STD_PARAM": "soft", "KRABBY_OMNI_SPEED_ELLIPSE": "0.35:0.35:0.25",
          "KRABBY_OMNI_FACING_OBS_CLIP": "0.6", "KRABBY_OMNI_STAND_OBS": "clock"}
SUITE = "suite__omni"
WS = "walkstop__omni"
_N = [0]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def RST():
    return _load(RESTANCE, "_t_z2d_restance")


@pytest.fixture(scope="module")
def FID():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return _load(SCRIPTS / "p2r_z2_fidelity.py", "p2r_z2_fidelity")


@pytest.fixture(scope="module")
def GATE(FID):
    return _load(SCRIPTS / "p2r_z2c_gate.py", "_t_z2d_gate")


@pytest.fixture(scope="module")
def D(FID):
    return _load(SCRIPTS / "p2r_z2_dose.py", "_t_z2d_dose")


@pytest.fixture(scope="module")
def live():
    return _load(SCRIPTS / "p2_live.py", "_t_z2d_live")


@pytest.fixture(scope="module")
def read():
    return _load(SCRIPTS / "p2_read.py", "_t_z2d_read")


@pytest.fixture(scope="module")
def Z():
    """The Z2c tool tests' helpers (the tiny REAL actor, run dirs, banks, the Z2c overlay / dose / series, the
    run_arm.sh watch harness)."""
    return _load(Z2C_TESTS, "_t_z2d_from_z2c_tools")


def _masked_cols(RST, width):
    cols, vals = RST.stand_mask_columns(width)
    return np.asarray(cols), np.asarray(vals, np.float32)


# ===================================================================================== the gate (+ the scorer)
def _capture_score_paths(monkeypatch, GATE):
    """Wrap the gate's ``fid.score_paths``: record the obs every call scores, then score as before."""
    seen = []
    real = GATE.fid.score_paths

    def spy(bank, *a, **k):
        seen.append(np.asarray(bank["obs"]).copy())
        return real(bank, *a, **k)
    monkeypatch.setattr(GATE.fid, "score_paths", spy)
    return seen


def test_gate_reads_the_mask_off_the_runs_env_yaml_and_masks_both_paths(GATE, RST, Z, tmp_path, monkeypatch):
    actor, est = Z._tiny_actor()
    bank_p = tmp_path / "es2" / "p2r_stand_bank.npz"
    bank = Z._bank(RST, bank_p)
    raw = np.asarray(bank["obs"])
    cols, vals = _masked_cols(RST, raw.shape[1])
    assert not np.array_equal(raw[:, cols], np.broadcast_to(vals, (raw.shape[0], cols.size)))   # random columns
    run = tmp_path / "run"
    Z._save_run(run, actor, est)
    seen = _capture_score_paths(monkeypatch, GATE)
    # no env.yaml: unmasked, the bank scored as stored
    j = tmp_path / "g0.json"
    rc0 = GATE.main(["--run-dir", str(run), "--bank", str(bank_p), "--json", str(j)])
    out0 = json.loads(j.read_text())
    assert out0["masked"] is False and out0["mask_source"] == "no params/env.yaml"
    assert np.array_equal(seen[-1], raw)
    # an env.yaml WITHOUT the mask: unmasked, named
    env = {"observations": {"policy": {"extreme_parkour_observations": {
        "func": "x:CrabHexOmniParkourObservations", "params": {"stand_bit_clock": True}}}}}
    (run / "params" / "env.yaml").write_text(yaml.safe_dump(env))
    GATE.main(["--run-dir", str(run), "--bank", str(bank_p), "--json", str(j)])
    out1 = json.loads(j.read_text())
    assert out1["masked"] is False and out1["mask_source"].endswith("no stand_mask")
    # the masked run: stand_mask true on the omni obs term
    env["observations"]["policy"]["extreme_parkour_observations"]["params"]["stand_mask"] = True
    (run / "params" / "env.yaml").write_text(yaml.safe_dump(env))
    rc2 = GATE.main(["--run-dir", str(run), "--bank", str(bank_p), "--json", str(j)])
    out2 = json.loads(j.read_text())
    assert out2["masked"] is True and "extreme_parkour_observations.params.stand_mask" in out2["mask_source"]
    got = seen[-1]
    assert np.array_equal(got[:, cols], np.broadcast_to(vals, got[:, cols].shape))
    keep = np.setdiff1d(np.arange(raw.shape[1]), cols)
    assert np.array_equal(got[:, keep], raw[:, keep])                    # nothing else touched
    # R is the unmasked bank's (R reads none of the masked columns) and the kill clause is the same statistic
    assert rc0 == rc2 == out2["exit"] and out2["eval"]["allhold_rows"] == out0["eval"]["allhold_rows"] == 50
    assert out2["lines"] == {"hip": 0.05, "cam": 0.05}


def test_gate_masking_a_bank_already_holding_the_constants_changes_nothing(GATE, RST, Z, tmp_path):
    actor, est = Z._tiny_actor(seed=3)
    bank_p = tmp_path / "es2" / "p2r_stand_bank.npz"
    b = Z._bank(RST, bank_p)
    obs = RST.stand_mask_obs(torch.as_tensor(b["obs"])).numpy()          # the closed-loop bank of a masked head
    np.savez(bank_p, **dict(b, obs=obs), restance=np.array("default"), has_residual=np.array(False))
    run = tmp_path / "run"
    Z._save_run(run, actor, est)
    ju, jm = tmp_path / "u.json", tmp_path / "m.json"
    GATE.main(["--run-dir", str(run), "--bank", str(bank_p), "--json", str(ju)])
    GATE.main(["--run-dir", str(run), "--bank", str(bank_p), "--json", str(jm), "--stand-mask"])
    u, m = json.loads(ju.read_text()), json.loads(jm.read_text())
    assert u["masked"] is False and m["masked"] is True and m["mask_source"] == "--stand-mask"
    for path in ("eval", "train_reported"):
        assert m[path] == u[path], path
    assert m["reported"] == u["reported"] and m["exit"] == u["exit"]


def test_gate_reports_lift_home_lines_and_report_banks_never_gated(GATE, RST, Z, tmp_path):
    actor, est = Z._tiny_actor(seed=5)
    bank_p = tmp_path / "es2" / "p2r_stand_bank.npz"
    Z._bank(RST, bank_p, n=120, n_hold=50, seed=1)
    crouch = tmp_path / "crouch" / "p2r_stand_bank.npz"
    Z._bank(RST, crouch, n=60, n_hold=30, seed=2)
    run = tmp_path / "run"
    mu = np.zeros(18, np.float32)
    mu[6:12] = -0.2                                                    # a crouch: the gate FAILS on hip
    Z._save_run(run, actor, est, const_mu=mu)
    j = tmp_path / "g.json"
    rc = GATE.main(["--run-dir", str(run), "--bank", str(bank_p), "--json", str(j), "--report-bank", str(crouch),
                    "--report-bank", str(tmp_path / "nope.npz")])
    out = json.loads(j.read_text())
    assert rc == GATE.RC_FAIL == out["exit"]                           # the verdict is the all-HOLD clause alone
    rep = out["reported"]
    assert rep["lift"]["line"] == 0.30 and rep["home"]["line"] == 0.25
    lift = out["eval"]["by_branch"]["lift"]
    assert rep["lift"]["leg_rows"] == lift["leg_rows"] > 0
    assert rep["lift"]["hip_mean_abs"] == pytest.approx(lift["hip_abs"]["mean"])
    assert rep["lift"]["ok"] == (lift["hip_abs"]["mean"] <= 0.30)
    if not rep["home"]["leg_rows"]:
        assert rep["home"]["hip_mean_abs"] is None and rep["home"]["ok"] is None
    rb = out["report_banks"]
    assert len(rb) == 2 and rb[0]["bank"] == str(crouch) and rb[0]["masked"] is False
    assert rb[0]["in_run_bc_banks"] is False and rb[0]["eval"]["allhold_rows"] == 30
    assert rb[0]["eval"]["hip_max_abs"] == pytest.approx(0.2, abs=1e-6) and set(rb[0]["reported"]) == {"lift", "home"}
    assert "error" in rb[1] and "eval" not in rb[1]
    # a PASSING run stays PASS with an unreadable report bank
    run2 = tmp_path / "run2"
    Z._save_run(run2, actor, est, const_mu=np.zeros(18, np.float32))
    assert GATE.main(["--run-dir", str(run2), "--bank", str(bank_p), "--report-bank",
                      str(tmp_path / "nope.npz")]) == GATE.RC_PASS
    # the scorer's own CLI: off, its JSON is unchanged (no masked / by_branch keys); on, masked recorded
    fj = tmp_path / "f.json"
    Z._save_run(tmp_path / "run3", actor, est)
    GATE.fid.main(["--bank", str(bank_p), "--ckpt", str(tmp_path / "run3" / "model_20500.pt"), "--json", str(fj)])
    f0 = json.loads(fj.read_text())
    assert "masked" not in f0 and "by_branch" not in f0["paths"]["eval"]
    GATE.fid.main(["--bank", str(bank_p), "--ckpt", str(tmp_path / "run3" / "model_20500.pt"), "--json", str(fj),
                   "--stand-mask"])
    assert json.loads(fj.read_text())["masked"] is True


# ===================================================================================== the masked BC dose
def test_bc_dose_masks_the_rows_before_the_forward(D, FID, RST, Z, tmp_path, monkeypatch):
    actor, est = Z._tiny_actor(seed=7)
    ck = Z._save_run(tmp_path / "run", actor, est, name="graft.pt", std_raw=0.3)
    b1 = tmp_path / "b1.npz"
    raw = Z._bank(RST, b1, n=30, n_hold=10, seed=1)["obs"]
    cols, vals = _masked_cols(RST, raw.shape[1])
    seen = []
    real = FID.eval_mu

    def spy(obs, *a, **k):
        seen.append(obs.numpy().copy())
        return real(obs, *a, **k)
    monkeypatch.setattr(FID, "eval_mu", spy)
    p = RST.parse_restance("trim=0.1,k_h=1")
    off = D.bc_dose([b1], ck, p, rst=RST)
    assert "stand_mask" not in off and np.array_equal(seen[-1], raw)
    on = D.bc_dose([b1], ck, p, rst=RST, stand_mask=True)
    assert on["stand_mask"] is True
    assert np.array_equal(seen[-1][:, cols], np.broadcast_to(vals, (30, cols.size)))
    keep = np.setdiff1d(np.arange(raw.shape[1]), cols)
    assert np.array_equal(seen[-1][:, keep], raw[:, keep])
    # R reads none of the masked columns: the targets (and so the stored-target diff) are the same
    assert on["banks"][0]["target_vs_stored_max_abs"] == off["banks"][0]["target_vs_stored_max_abs"]


def test_dose_cli_bc_stand_mask(D, RST, Z, tmp_path):
    zp = _load(Z2_PROBE_TESTS, "_t_z2d_from_z2_probe")
    stamp = zp._dose_stamp(tmp_path, D)
    actor, est = Z._tiny_actor(seed=2)
    ck = Z._save_run(tmp_path / "run", actor, est, name="graft.pt")
    b1 = tmp_path / "b1.npz"
    Z._bank(RST, b1)
    base = ["--raws", str(stamp), "--ckpt", str(ck), "--restance", "default", "--xcheck-stamp", "none"]
    j = tmp_path / "new" / "dir" / "p2r_z2d_dose.json"                 # a NEW file in a new directory
    md = tmp_path / "new" / "md" / "p2r_z2d_dose.md"
    rc = D.main(base + ["--bc-bank", str(b1), "--bc-stand-mask", "--json", str(j), "--md", str(md)])
    out = json.loads(j.read_text())
    assert rc == out["rc"] and out["bc"]["stand_mask"] is True and "bc_mb0_expected" in out
    assert "stand-row input mask" in md.read_text()
    # refused: no banks to mask; the Z2 / Z2c file (never written)
    assert D.main(base + ["--bc-stand-mask"]) == D.RC_ERROR
    z2 = D.Z2_DOSE_JSON
    assert z2 == CAMP / "p2r_z2_dose.json"
    before = z2.stat().st_mtime_ns if z2.exists() else None
    assert D.main(base + ["--bc-bank", str(b1), "--bc-stand-mask", "--json", str(z2)]) == D.RC_ERROR
    assert (z2.stat().st_mtime_ns if z2.exists() else None) == before


# ===================================================================================== p2_live: the Z2D arm
F0 = 19995


def _banks(tmp_path):
    out = []
    for n in ("ws_es1.npz", "suite_es1.npz", "z2c_suite_observe.npz"):
        p = tmp_path / n
        p.write_bytes(b"npz")
        out.append(p)
    return out


def _ov_d(Z, live, banks, **over):
    return Z._ov(live, banks, **dict({"KRABBY_OMNI_STAND_MASK": "1"}, **over))


def _dose_d(Z, live, banks, stand_mask=True, **over):
    bc = {"banks": [{"path": str(Path(b).resolve())} for b in banks]}
    if stand_mask is not None:
        bc["stand_mask"] = stand_mask
    return Z._dose(live, banks, **dict({"bc": bc}, **over))


def _ser_d(Z, live, frac=0.12, drop_frac=False, **kw):
    ser = Z._series(live, **kw)
    if not drop_frac:
        ser[live.P2R_STAND_MASK_FRAC_TAG] = [(F0 + i, frac) for i in range(215)]
    return ser


def _wire(live, arm, ov, ser, dose):
    w = dict(live._facts()["weights"], reward_action_rate=-0.05)
    audit = [("overlay KRABBY_TERRAIN_PROMOTE=0.75:0.4 in params/env.yaml", True)]
    return live.judge_wiring(arm, ov, live._facts(weights=w), ser, F0, audit, None, z2_dose=dose)


def test_z_arm_reads_z2d_and_the_constants(live):
    assert live.z_arm("P2__Z2d_s18") == "Z2D" == live.P2R_Z2D_ARM and live.z_arm("P2__Z2dR1_s16") == "Z2D"
    assert live.P2R_Z2D_DOSE_JSON == CAMP / "p2r_z2d_dose.json" != live.P2R_Z2_DOSE_JSON
    assert (live.P2R_STAND_MASK, live.P2R_STAND_MASK_FRAC_TAG) == ("KRABBY_OMNI_STAND_MASK", "Metrics/stand_mask_frac")
    assert live.z2_dose_default("Z2D") == live.P2R_Z2D_DOSE_JSON
    assert live.z2_dose_default("Z2C") == live.z2_dose_default("Z2") == live.P2R_Z2_DOSE_JSON
    assert live.z_variant_unknown("Z2E") and not live.z_variant_unknown("Z2D") and not live.z_variant_unknown("Z1")
    assert not live.z_variant_unknown("Z2C") and not live.z_variant_unknown("C0")
    src = (SCRIPTS / "p2_live.py").read_text()
    assert "{P2R_STAND_MASK_FRAC_TAG}" in src.split("def read_tb")[1].split("def load_rc")[0]
    # check() / preflight() read the arm's own dose file when --z2-dose is not given
    assert src.count('getattr(a, "z2_dose", None) or z2_dose_default(arm)') == 2
    assert 'ap.add_argument("--z2-dose", type=Path, default=None' in src


def test_z2d_wiring_clean_passes_and_dispatches_to_the_z2d_judge(Z, live, tmp_path):
    banks = _banks(tmp_path)
    r = _wire(live, "Z2D", _ov_d(Z, live, banks), _ser_d(Z, live), _dose_d(Z, live, banks))
    assert r["rc"] == live.RC_PASS, r["fails"]
    assert r["liveness"]["stand_mask_frac"] == pytest.approx(0.12)          # the Z2D judge ran
    assert r["liveness"]["bc_mb0_first_update"] == 8.9                      # ... over Z2c's
    # the same inputs under a generic Z arm would have read only Z1's checks: Z2D never falls through
    gen = _wire(live, "Z3", _ov_d(Z, live, banks), _ser_d(Z, live), _dose_d(Z, live, banks))
    assert "stand_mask_frac" not in gen["liveness"]


@pytest.mark.parametrize("case", ["frac_high", "frac_low", "frac_missing", "frac_nan", "frac_edge_lo_ok",
                                  "frac_edge_hi_ok", "no_mask_knob", "mask_knob_0", "dose_unmasked",
                                  "dose_no_stand_mask", "no_dose_file", "dose_other_banks", "z2c_bc_mb0_low",
                                  "z2c_no_loss_bc", "z2_band_still_read", "prior_below_band_live_ok",
                                  "prior_dead", "prior_none"])
def test_z2d_wiring_cases(Z, live, tmp_path, case):
    banks = _banks(tmp_path)
    ov, ser, dose = _ov_d(Z, live, banks), _ser_d(Z, live), _dose_d(Z, live, banks)
    want = live.RC_FAIL
    if case == "frac_high":
        ser = _ser_d(Z, live, frac=0.3)
    elif case == "frac_low":
        ser = _ser_d(Z, live, frac=0.01)
    elif case == "frac_missing":
        ser = _ser_d(Z, live, drop_frac=True)
    elif case == "frac_nan":
        ser = _ser_d(Z, live, frac=float("nan"))
    elif case == "frac_edge_lo_ok":
        ser, want = _ser_d(Z, live, frac=0.05), live.RC_PASS
    elif case == "frac_edge_hi_ok":
        ser, want = _ser_d(Z, live, frac=0.2), live.RC_PASS
    elif case == "no_mask_knob":
        ov = _ov_d(Z, live, banks, KRABBY_OMNI_STAND_MASK=None)
    elif case == "mask_knob_0":
        ov = _ov_d(Z, live, banks, KRABBY_OMNI_STAND_MASK="0")
    elif case == "dose_unmasked":
        dose = _dose_d(Z, live, banks, stand_mask=False)
    elif case == "dose_no_stand_mask":
        dose = _dose_d(Z, live, banks, stand_mask=None)
    elif case == "no_dose_file":
        dose = None
    elif case == "dose_other_banks":
        dose = _dose_d(Z, live, banks[:2])
    elif case == "z2c_bc_mb0_low":
        ser = _ser_d(Z, live, bc_mb0=0.51 * 8.9)
    elif case == "z2c_no_loss_bc":
        ser = _ser_d(Z, live, drop=(live.P2R_LOSS_BC,))
    elif case == "z2_band_still_read":
        ser = _ser_d(Z, live, prior_first=9.0)                              # above the band's upper edge: FAIL
    elif case == "prior_below_band_live_ok":
        # NOTE(p2r Z2d, 2026-10-11; user decision after the +200 kill): below the UNMASKED band but live -> PASS
        ser, want = _ser_d(Z, live, prior_first=0.5), live.RC_PASS
    elif case == "prior_dead":
        ser = _ser_d(Z, live, prior_first=0.05)                             # <= P2R_Z2D_PRIOR_LIVE_FLOOR: FAIL
    elif case == "prior_none":
        ser = _ser_d(Z, live, prior_first=float("nan"))                     # not finite: FAIL
    r = _wire(live, "Z2D", ov, ser, dose)
    assert r["rc"] == want, (case, r.get("fails"))
    if case == "prior_below_band_live_ok":
        assert "Z2D liveness" in r["liveness"]["prior_dose_rule"]
        assert any("below the UNMASKED band" in n for n in r.get("notes", [])), r.get("notes")
    if case == "no_dose_file":
        assert any("p2r_z2d_dose.json" in f for f in r["fails"]), r["fails"]


def test_unknown_lettered_z_arm_is_refused(Z, live, tmp_path):
    banks = _banks(tmp_path)
    r = _wire(live, "Z2E", _ov_d(Z, live, banks), _ser_d(Z, live), _dose_d(Z, live, banks))
    assert r["rc"] == live.RC_FAIL and any("variant letter" in f for f in r["fails"]), r["fails"]
    em = {"terrain_levels_obst": ["x.py"], "promote_count": ["x.py"]}
    graft = {"identity_cpu": True, "identity_gpu_stamps": ["GRAFT_S1_s18/suite__omni/seed001/x"]}
    ov = _ov_d(Z, live, banks)
    pf = live.judge_preflight("Z2E", em, None, None, ov, p2r_graft=graft, fork_ckpt=ov["KRABBY_ANCHOR_CKPT"])
    assert pf["rc"] == live.RC_FAIL and any("variant letter" in f for f in pf["fails"])
    # Z1 (no letter) is untouched by the refusal
    assert not any("variant letter" in f for f in live.judge_preflight(
        "Z1", em, None, None, ov, p2r_graft=graft, fork_ckpt=ov["KRABBY_ANCHOR_CKPT"])["fails"])


def test_z2d_preflight(Z, live, tmp_path):
    banks = _banks(tmp_path)
    em = {"terrain_levels_obst": ["x.py"], "promote_count": ["x.py"]}
    graft = {"identity_cpu": True, "identity_gpu_stamps": ["GRAFT_S1_s18/suite__omni/seed001/x"]}
    ov = _ov_d(Z, live, banks)
    fk = ov["KRABBY_ANCHOR_CKPT"]

    def pf(arm, ov_, dose_):
        return live.judge_preflight(arm, em, None, None, ov_, p2r_graft=graft, fork_ckpt=fk, z2_dose=dose_)
    assert pf("Z2D", ov, _dose_d(Z, live, banks))["rc"] == live.RC_PASS
    r = pf("Z2D", _ov_d(Z, live, banks, KRABBY_OMNI_STAND_MASK=None), _dose_d(Z, live, banks))
    assert r["rc"] == live.RC_FAIL and any("KRABBY_OMNI_STAND_MASK" in f for f in r["fails"])
    r = pf("Z2D", ov, None)                                                  # the Z2d dose file is missing
    assert r["rc"] == live.RC_FAIL and any("p2r_z2d_dose.json" in f for f in r["fails"])
    r = pf("Z2D", ov, _dose_d(Z, live, banks, stand_mask=None))              # a dose without stand_mask
    assert r["rc"] == live.RC_FAIL and any("bc.stand_mask" in f for f in r["fails"])
    assert pf("Z2D", ov, _dose_d(Z, live, banks, stand_mask=False))["rc"] == live.RC_FAIL
    assert pf("Z2D", ov, _dose_d(Z, live, banks[:2]))["rc"] == live.RC_FAIL  # banks differ from the chain's
    assert pf("Z2D", _ov_d(Z, live, banks, KRABBY_BC_W=None), _dose_d(Z, live, banks))["rc"] == live.RC_FAIL
    assert pf("Z2D", _ov_d(Z, live, banks, KRABBY_STAND_PRIOR_TARGET="zero"),
              _dose_d(Z, live, banks))["rc"] == live.RC_FAIL                 # Z2's knobs still apply
    # Z2c's preflight is unchanged: no mask, no stand_mask asked
    zc = Z._ov(live, banks)
    assert pf("Z2C", zc, Z._dose(live, banks))["rc"] == live.RC_PASS


def test_z2d_inherits_the_ep_length_notify_only_guard(live):
    assert "Z2D" in live.P2R_Z2_FAMILY
    src = (SCRIPTS / "p2_live.py").read_text()
    assert "ep_len_notify=(arm in P2R_Z2_FAMILY)" in src


# ===================================================================================== run_arm.sh: the Z2d gate
def _bash(script, env=None):
    e = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2_"))}
    e.update(P2_RUN_ARM_LIB="1", P2_PY=sys.executable, P2_CAP="", **(env or {}))
    return subprocess.run(["bash", "-c", f"source {RUN_ARM}\n{script}"], env=e, capture_output=True, text=True,
                          timeout=60)


def test_run_arm_is_z2d_and_is_z2c_answers_for_it(tmp_path):
    camp = tmp_path / "camp"
    (camp / "logs").mkdir(parents=True)
    env = {"P2_OUT": str(tmp_path / "out"), "CAMPAIGN_DIR": str(camp)}
    script = "\n".join(f'is_z2d {c} && echo "d:{c}"; is_z2c {c} && echo "c:{c}"' for c in (
        "P2__Z2d_s18", "P2__Z2dR1_s16", "P2__Z2c_s18", "P2__Z2_s18", "P2__Z1_s18", "P2__ST2_s18", "P2__Z2e_s18"))
    out = _bash(script, env).stdout.split()
    assert out == ["d:P2__Z2d_s18", "c:P2__Z2d_s18", "d:P2__Z2dR1_s16", "c:P2__Z2dR1_s16", "c:P2__Z2c_s18"]
    src = RUN_ARM.read_text()
    assert "is_z2d()" in src and "NOTE(p2r Z2d, 2026-10-11)" in src


def test_run_arm_refuses_a_z2d_chain_without_a_held_out_gate_bank(tmp_path):
    camp = tmp_path / "camp"
    (camp / "logs").mkdir(parents=True)
    bank = tmp_path / "es2.npz"
    bank.write_bytes(b"x")
    es1 = tmp_path / "es1.npz"
    es1.write_bytes(b"x")
    (camp / "candidates.yaml").write_text(json.dumps({"chains": {"P2__Z2d_s18": {"stages": [
        {"overlay": {"KRABBY_BC_BANK": f"{es1},{bank}", "KRABBY_OMNI_STAND_MASK": "1"}}]}}}))
    env = {"P2_OUT": str(tmp_path / "out"), "CAMPAIGN_DIR": str(camp)}
    assert "is not set" in _bash("z2c_gate_refusal P2__Z2d_s18", env).stdout
    assert "one of the chain's KRABBY_BC_BANK" in _bash("z2c_gate_refusal P2__Z2d_s18",
                                                         dict(env, P2_Z2C_GATE_BANK=str(bank))).stdout
    other = tmp_path / "es2_other.npz"
    other.write_bytes(b"x")
    assert _bash("z2c_gate_refusal P2__Z2d_s18", dict(env, P2_Z2C_GATE_BANK=str(other))).stdout.strip() == ""
    # run_ref reaches the refusal through is_z2c (which answers for a Z2d chain)
    run_ref = RUN_ARM.read_text().split("run_ref() {")[1].split("\n}\n")[0]
    assert 'if is_z2c "$CHAIN"; then' in run_ref


def test_run_arm_z2d_gate_rc3_kills_through_the_live_read_path(Z, tmp_path):
    log, calls = Z._gate_run(tmp_path, 3, chain="P2__Z2d_s18", age=120)
    assert len(calls) == 1 and f"--bank {tmp_path / 'es2.npz'}" in calls[0] and "CUDA=''" in calls[0]
    assert "Z2CGATE P2__Z2d_s18 rc 3 FAIL" in log and "KILL P2__Z2d_s18 (z2c-fidelity-gate at model_20500.pt" in log
    assert (tmp_path / "out" / "STOP_QUEUE").exists()
    assert (tmp_path / "out" / "arms" / "P2__Z2d_s18.killed").read_text().startswith("z2c-fidelity-gate\t")


def test_run_arm_z2d_gate_pass_runs_once(Z, tmp_path):
    log, calls = Z._gate_run(tmp_path, 0, chain="P2__Z2d_s18", age=120)
    assert len(calls) == 1 and "Z2CGATE P2__Z2d_s18 PASS model_20500.pt" in log and "KILL" not in log


# ===================================================================================== p2_read: masked heads
def _meta_stamp(root: Path, env=None) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "run_meta.json").write_text(json.dumps({"env_vars": ENV_Z2 if env is None else env,
                                                    "env_vars_manifest": {}, "checkpoint_sha256": "a" * 64}))
    (root / "code_state.json").write_text(json.dumps({"code_sha": "c" * 64, "eval_sha": "e" * 64, "dirty": False}))
    return str(root)


def test_head_masked_reads_the_chain_overlay(read):
    cands = {"chains": {"P2__Z2d_s18": {"stages": [{"overlay": {"KRABBY_OMNI_STAND_MASK": "1"}}]},
                        "P2__Z2c_s18": {"stages": [{"overlay": {"KRABBY_BC_W": "1.0"}}]},
                        "P2__Z2x_s18": {"stages": [{"overlay": {"KRABBY_OMNI_STAND_MASK": "1"}},
                                                   {"overlay": {"KRABBY_OMNI_STAND_MASK": "0"}}]}},
             "references": {"Z2D_REF": {"chain": "P2__Z2d_s18"}}}
    assert read.head_masked("P2__Z2d_s18", cands) is True
    assert read.head_masked("Z2D_REF", cands) is True                     # a reference resolves to its chain
    assert read.head_masked("P2__Z2c_s18", cands) is False
    assert read.head_masked("P2__Z2x_s18", cands) is False                # the last stage that sets it wins
    assert read.head_masked("nope", cands) is None and read.head_masked(None, cands) is None   # fails closed
    assert any("cannot resolve" in r for r in read.z2_invalid_reasons({"name": "nope", "stamps": {}}))
    # the real campaign: no chain masked before candidates v17
    assert read.head_masked("P2__Z2c_s18") is False and read.head_masked("P2__Z1_s18") is False


def test_want_mask_invalid_rules(read, tmp_path):
    on = _meta_stamp(tmp_path / "on", env=dict(ENV_Z2, KRABBY_OMNI_STAND_MASK="1"))
    off = _meta_stamp(tmp_path / "off")
    zero = _meta_stamp(tmp_path / "zero", env=dict(ENV_Z2, KRABBY_OMNI_STAND_MASK="0"))

    def pt(**st):
        return {"name": "P2__Z2d_s18", "stamps": st}
    # a masked head: every stamp must carry the mask
    assert read.z2_env_reasons(pt(**{"suite@1": on, f"{WS}@1": on}), want_mask=True) == []
    r = read.z2_env_reasons(pt(**{"suite@1": on, f"{WS}@1": off}), want_mask=True)
    assert len(r) == 1 and "KRABBY_OMNI_STAND_MASK" in r[0] and WS in r[0]
    assert len(read.z2_env_reasons(pt(**{"suite@1": zero}), want_mask=True)) == 1
    # an unmasked head: a stamp WITH the mask is INVALID; absent / 0 are fine
    assert read.z2_env_reasons(pt(**{"suite@1": off, f"{WS}@1": zero}), want_mask=False) == []
    r = read.z2_env_reasons(pt(**{"suite@1": on}), want_mask=False)
    assert len(r) == 1 and "want it off" in r[0]
    # the default (the controller probe's path): no mask rule at all
    assert read.z2_env_reasons(pt(**{"suite@1": on, f"{WS}@1": off})) == []


def test_z2_invalid_reasons_keeps_the_mask_reasons(read, tmp_path, monkeypatch):
    monkeypatch.setattr(read, "z_raw_bit_reasons", lambda pt, lag=1: ([], 1))
    on = _meta_stamp(tmp_path / "on", env=dict(ENV_Z2, KRABBY_OMNI_STAND_MASK="1"))
    off = _meta_stamp(tmp_path / "off")
    assert read.z2_invalid_reasons({"name": "X", "stamps": {"suite@1": on}}, want_mask=True) == []
    r = read.z2_invalid_reasons({"name": "X", "stamps": {"suite@1": off}}, want_mask=True)
    assert len(r) == 1 and "KRABBY_OMNI_STAND_MASK" in r[0]
    r = read.z2_invalid_reasons({"name": "X", "stamps": {"suite@1": on}}, want_mask=False)
    assert len(r) == 1 and "KRABBY_OMNI_STAND_MASK" in r[0]
    # the default reads the head's chain overlay
    monkeypatch.setattr(read, "head_masked", lambda name, cands=None: name == "P2__Z2d_s18")
    assert read.z2_invalid_reasons({"name": "P2__Z2d_s18", "stamps": {"suite@1": off}})
    assert read.z2_invalid_reasons({"name": "P2__Z2d_s18", "stamps": {"suite@1": on}}) == []
    assert read.z2_invalid_reasons({"name": "P2__Z2c_s18", "stamps": {"suite@1": on}})
    # ... and so does the verdict override (INVALID EVAL for a masked head's unmasked stamp)
    pt = {"name": "P2__Z2d_s18", "stamps": {"suite@1": off}}
    monkeypatch.setattr(read, "p2r_z_verdicts", lambda res: {"P2__Z2d_s18": {"verdict": "STILL", "why": []}})
    assert read.p2r_z2_verdicts({"heads": {"P2__Z2d_s18": pt}})["P2__Z2d_s18"]["verdict"] == "INVALID EVAL"


def test_rst_probe_path_is_unchanged_by_a_mask_carrying_stamp(read, tmp_path):
    s = _meta_stamp(tmp_path / "p", env=dict(ENV_Z2, KRABBY_OMNI_STAND_MASK="1"))
    (Path(s) / "run_meta.json").write_text(json.dumps({
        "env_vars": dict(ENV_Z2, KRABBY_OMNI_STAND_MASK="1"), "env_vars_manifest": {}, "checkpoint_sha256": "a" * 64,
        "p2r_mode": {"variant": "restance", "s1_checkpoint_sha256": "a" * 64}}))
    refs_s = _meta_stamp(tmp_path / "refs", env={k: v for k, v in ENV_Z2.items() if k != "KRABBY_OMNI_STAND_OBS"})
    refs = {"name": "REFS", "stamps": {"suite@1": refs_s}, "valid": {"suite@1": read.stamp_validity(refs_s)}}
    probe = {"name": "RST", "stamps": {"suite@1": s}, "valid": {"suite@1": read.stamp_validity(s)}, "ws_env": {}}
    assert not any("STAND_MASK" in r for r in read.rst_invalid_reasons(probe, refs, raw_bit=False))


def test_z2_suite_stamp_skips_the_driver_stamp_for_a_masked_head(tmp_path, monkeypatch):
    out = tmp_path / "out"
    drv = Path(_meta_stamp(out / "driver" / "suite"))                         # the driver suite: no mask
    (out / "arms").mkdir(parents=True)
    (out / "arms" / "P2__Z2d_s18.record.json").write_text(json.dumps({"driver_stamp": str(drv)}))
    monkeypatch.setenv("P2_OUT", str(out))
    _N[0] += 1
    r = _load(SCRIPTS / "p2_read.py", f"_t_z2d_read_at_{_N[0]}")
    assert r.z2_suite_stamp("P2__Z2d_s18", masked=False) == str(drv)
    assert r.z2_suite_stamp("P2__Z2d_s18", masked=True) is None               # no run_arm re-read yet
    lr = out / "lr" / "P2__Z2d_s18" / SUITE / "seed001"
    for ts in ("2026-10-12_00-00-00", "2026-10-12_01-00-00"):
        _meta_stamp(lr / ts, env=dict(ENV_Z2, KRABBY_OMNI_STAND_MASK="1"))
        (lr / ts / "complete.json").write_text("{}")
    assert r.z2_suite_stamp("P2__Z2d_s18", masked=True) == str(lr / "2026-10-12_01-00-00")   # the newest
    assert r.z2_suite_stamp("P2__Z2d_s18", masked=False) == str(drv)
    # the default reads the chain overlay (head_masked)
    monkeypatch.setattr(r, "head_masked", lambda name, cands=None: True)
    assert r.z2_suite_stamp("P2__Z2d_s18") == str(lr / "2026-10-12_01-00-00")
    monkeypatch.setattr(r, "head_masked", lambda name, cands=None: False)
    assert r.z2_suite_stamp("P2__Z2d_s18") == str(drv)


# ===================================================================================== the observe read
def _observe_tree(lr: Path, name: str, obs: np.ndarray, mu: np.ndarray):
    d = lr / name / SUITE / "seed001" / "2026-10-12_00-00-00"
    d.mkdir(parents=True)
    n = obs.shape[0]
    np.savez(d / "p2r_stand_bank.npz", obs=obs, mu=mu, t=np.arange(n), observe=np.array(True))
    return d / "p2r_stand_bank.npz"


@pytest.fixture(scope="module")
def OBS(FID):
    return _load(SCRIPTS / "p2r_z2c_observe_read.py", "_t_z2d_observe_read")


def test_observe_read_refuses_an_unmasked_bank_under_stand_mask(OBS, RST, Z, tmp_path, capsys):
    b = Z._bank(RST, None, n=40, n_hold=20)
    obs = np.asarray(b["obs"], np.float32)
    lr = tmp_path / "lr"
    _observe_tree(lr, "P2R_OBSZ2d_s18", obs, np.zeros((40, 18), np.float32))
    nockpt = tmp_path / "nope.pt"                                      # refused before any checkpoint is read
    assert OBS.main(["--name", "P2R_OBSZ2d_s18", "--lr", str(lr), "--ckpt", str(nockpt),
                     "--stand-mask"]) == OBS.RC_REFUSED
    assert "REFUSED" in capsys.readouterr().err
    # one stand row left unmasked is enough to refuse
    m = RST.stand_mask_obs(torch.as_tensor(obs)).numpy()
    m[7, 70] = 0.25
    assert OBS.mask_proof(m) == {"stand_rows": 40, "unmasked_stand_rows": 1, "ok": False}
    assert OBS.mask_proof(RST.stand_mask_obs(torch.as_tensor(obs)).numpy())["ok"] is True
    # walking rows (stand bit 0) are not asked to carry the mask; no stand row at all proves nothing
    w = obs.copy()
    w[:, 15] = 0.0
    assert OBS.mask_proof(w) == {"stand_rows": 0, "unmasked_stand_rows": 0, "ok": False}


def test_observe_read_scores_a_masked_bank_through_the_mask(OBS, FID, RST, Z, tmp_path, capsys):
    actor, est = Z._tiny_actor(seed=9)
    ck = Z._save_run(tmp_path / "run", actor, est, name="model_1.pt")
    b = Z._bank(RST, None, n=40, n_hold=20)
    obs = RST.stand_mask_obs(torch.as_tensor(b["obs"])).numpy()
    w = FID.widths(tmp_path / "run" / "params" / "agent.yaml")
    sd, esd = FID.load_ckpt(ck)
    with torch.no_grad():
        mu = FID.eval_mu(torch.as_tensor(obs), sd, esd, w).numpy()      # what the masked env applied
    lr = tmp_path / "lr"
    _observe_tree(lr, "P2R_OBSZ2d_s18", obs, mu)
    j = tmp_path / "obs.json"
    rc = OBS.main([str(j), "--name", "P2R_OBSZ2d_s18", "--lr", str(lr), "--ckpt", str(ck), "--stand-mask",
                   "--restance", "default", "--agent-yaml", str(tmp_path / "run" / "params" / "agent.yaml")])
    assert rc == OBS.RC_OK
    res = json.loads(j.read_text())
    assert set(res) == {"suite"} and res["suite"]["masked"] is True
    assert res["suite"]["cpu_vs_applied_mu_max_abs"] < 1e-5 and res["suite"]["rows"] == 40
    assert res["suite"]["column_scale_vs_train_banks"]["proxy(83:89)"]["train_mean_abs"] is None   # no train bank
    # without --stand-mask the same bank reads unmasked (no refusal, masked false)
    assert OBS.main([str(j), "--name", "P2R_OBSZ2d_s18", "--lr", str(lr), "--ckpt", str(ck), "--restance",
                     "default", "--agent-yaml", str(tmp_path / "run" / "params" / "agent.yaml")]) == OBS.RC_OK
    assert json.loads(j.read_text())["suite"]["masked"] is False
    assert OBS.DEFAULT_NAME == "P2R_OBSZ2c_s18"
    assert OBS.rst.parse_restance(OBS.Z2C_RESTANCE) == OBS.PARAMS
