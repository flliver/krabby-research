"""Stage p2r Z2c (P2R_Z2C_PLAN.md steps 1 / 4 / 5; CHARTER 12): the TOOLS of the clean-state cloning run. CPU only:
nothing here starts Isaac/Kit or a GPU. NOTE(p2r Z2c, 2026-10-10).

* ``scripts/p2r_z2_fidelity.py``: R RECOMPUTED with the run's constants (``--restance`` / the run's
  ``params/agent.yaml`` ``algorithm.restance``; refused without either; ``--stored-target`` = the old read); the
  all-HOLD statistic (per-leg signed hip / knee / cam bias, max over legs) on BOTH the eval and the training path;
* ``scripts/p2r_z2c_gate.py``: the model_20500 fidelity gate -- exit 0 PASS / 3 FAIL (hip or cam over 0.05) /
  2 WAIT (no checkpoint) / 1 error (no constants, no bank, a BC-trained bank, no all-HOLD row); ``--ckpt-state``
  scores an offline actor through the same path;
* ``scripts/p2r_z2_dose.py --bc-bank``: ``bc_mb0_expected`` = the fork's training-path BC loss per row on the pooled
  banks (an independent recomputation through the REAL ``Actor``), the stand-bit refusal, the CLI JSON;
* ``scripts/p2_live.py``: ``z_arm`` reads P2__Z2c_s18 as Z2C; the Z2c +200 wiring (Z2's + Loss/bc / Loss/bc_mb0
  logged, finite, the first-update Loss/bc_mb0 inside [0.52, 2] x bc_mb0_expected, the BC knobs, the dose file's
  banks = the chain's; Loss/stand_prior_mb0 read for Z2's dose clause when logged); the preflight; Z2's ep-length
  notify-only guard inherited;
* ``scripts/run_arm.sh``: the Z2c gate branch in watch_arm (rc 3 -> the kill path; rc 0 -> PASS, the run
  continues; rc 1 -> one NOTIFY line; not yet -> retried), the pre-launch refusal without a held-out gate bank.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import time
import types
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
MODULES = REPO / "parkour/scripts/rsl_rl/modules"
RUN_ARM = SCRIPTS / "run_arm.sh"
Z2_PROBE_TESTS = REPO / "tests/unit/test_omni_p2r_z2_probe.py"
DIMS = dict(num_prop=20, num_scan=5, num_priv_explicit=3, num_priv_latent=4, num_hist=10)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def RST():
    return _load(RESTANCE, "_t_z2c_restance")


@pytest.fixture(scope="module")
def FID():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return _load(SCRIPTS / "p2r_z2_fidelity.py", "p2r_z2_fidelity")


@pytest.fixture(scope="module")
def GATE(FID):
    return _load(SCRIPTS / "p2r_z2c_gate.py", "_t_z2c_gate")


@pytest.fixture(scope="module")
def D():
    return _load(SCRIPTS / "p2r_z2_dose.py", "_t_z2c_dose")


@pytest.fixture(scope="module")
def live():
    return _load(SCRIPTS / "p2_live.py", "_t_z2c_live")


# ===================================================================================== a tiny REAL actor
def _real_modules():
    pkg, sub = "_t_z2c_rslmods", "_t_z2c_rslmods.feature_extractors"
    if pkg not in sys.modules:
        p = types.ModuleType(pkg)
        p.__path__ = [str(MODULES)]
        sys.modules[pkg] = p
        s = types.ModuleType(sub)
        s.__path__ = [str(MODULES / "feature_extractors")]
        sys.modules[sub] = s
    import importlib  # noqa: PLC0415

    return importlib.import_module(pkg + ".actor_critic_with_encoder"), importlib.import_module(sub + ".estimator")


def _tiny_actor(seed=0):
    acwe, est = _real_modules()
    torch.manual_seed(seed)
    actor = acwe.Actor(num_actions=18, scan_encoder_dims=[9, 6], actor_hidden_dims=[11, 7],
                       priv_encoder_dims=[7, 5], activation=torch.nn.ELU(), tanh_encoder_output=False,
                       state_history_encoder={"class_name": "StateHistoryEncoder", "channel_size": 4}, **DIMS)
    estimator = est.DefaultEstimator(num_prop=DIMS["num_prop"], num_priv_explicit=DIMS["num_priv_explicit"],
                                     hidden_dims=[10, 8])
    for prm in list(actor.parameters()) + list(estimator.parameters()):
        torch.nn.init.normal_(prm, 0.0, 0.3)
    return actor, estimator


def _last_layer(sd: dict) -> str:
    """``actor.actor_backbone.<N>`` of the backbone's LAST Linear (its output = mu before the optional tanh)."""
    n = max(int(k.split(".")[2]) for k in sd if k.startswith("actor.actor_backbone.") and k.endswith(".weight"))
    return f"actor.actor_backbone.{n}"


def _save_run(tmp: Path, actor, estimator, *, name="model_20500.pt", restance="default", bc_bank=None,
              const_mu=None, std_raw=0.0) -> Path:
    """A run dir with ``name`` + ``params/agent.yaml`` (widths, ``algorithm.restance`` as a DICT, like the trainer's
    dump). ``const_mu`` (18,) zeroes the backbone's last layer and sets its bias: mu == const_mu on every row."""
    msd = {f"actor.{k}": v.detach().clone() for k, v in actor.state_dict().items()}
    msd["std_raw"] = torch.full((18,), float(std_raw))
    if const_mu is not None:
        last = _last_layer(msd)
        msd[last + ".weight"].zero_()
        msd[last + ".bias"] = torch.as_tensor(const_mu, dtype=torch.float32).clone()
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "params").mkdir(exist_ok=True)
    torch.save({"model_state_dict": msd, "estimator_state_dict": estimator.state_dict()}, tmp / name)
    alg = {}
    if restance is not None:
        rst = _load(RESTANCE, "_t_z2c_restance_save")
        p = rst.parse_restance(restance)
        alg["restance"] = p.to_dict()
    if bc_bank is not None:
        alg["bc_bank"] = [str(b) for b in bc_bank]
    agent = {"algorithm": alg, "policy": {"activation": "elu", "tanh_encoder_output": False,
                                          "actor": dict(DIMS, state_history_encoder={"channel_size": 4})}}
    (tmp / "params" / "agent.yaml").write_text(yaml.safe_dump(agent))
    return tmp / name


def _obs_width():
    return max(89, sum(DIMS[k] for k in ("num_prop", "num_scan", "num_priv_explicit", "num_priv_latent"))) + \
        DIMS["num_hist"] * DIMS["num_prop"]


def _bank(RST, path: Path | None = None, n=80, n_hold=50, seed=0, stand_bit=1.0, stored_restance="default"):
    """``n`` stand rows, the first ``n_hold`` ALL-HOLD (every cam 0, every hip at default), the rest with cams out of
    band (WAIT / LIFT). Every other column random; col 15 = the stand bit."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, _obs_width(), generator=g)
    obs[:, 15] = stand_bit
    obs[:, list(RST.HIP_POS_OBS_COLS)] = 0.0
    obs[:, list(RST.CAM_OBS_COLS)] = 0.0
    obs[n_hold:, list(RST.CAM_OBS_COLS)] = 0.5 + torch.rand(n - n_hold, 6, generator=g)
    obs[:, [RST.ROLL_OBS_COL, RST.PITCH_OBS_COL]] = 0.0
    tgt, br = RST.restance_target(obs, RST.parse_restance(stored_restance))
    bank = {"obs": obs.numpy(), "target": tgt.numpy(), "branch": br.numpy().astype(np.int8),
            "s1_sigma": np.full(18, 0.5, np.float32)}
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **bank, restance=np.array(stored_restance), has_residual=np.array(False))
    return bank


# ===================================================================================== fidelity
def test_allhold_bias_is_the_per_leg_signed_mean_on_all_hold_rows(FID, RST):
    bank = _bank(RST, n=40, n_hold=25)
    tgt = torch.as_tensor(bank["target"])
    br = torch.as_tensor(bank["branch"]).long()
    assert bool((br[:25] == RST.BRANCH_HOLD).all()) and not bool((br[25:] == RST.BRANCH_HOLD).all(1).any())
    mu = tgt.clone()
    mu[:25, 6 + 2] -= 0.15                         # ML hip: a partial LIFT on the clean rows (Z2's signature)
    mu[:25, 4] += 0.07                             # RL cam
    mu[25:, :] += 3.0                              # anything on the non-all-HOLD rows is never read
    st = FID.allhold_bias(mu, tgt, br, RST)
    assert st["allhold_rows"] == 25 and st["allhold_share"] == pytest.approx(25 / 40)
    assert st["hip_by_leg"]["ML"] == pytest.approx(-0.15) and st["hip_max_abs"] == pytest.approx(0.15)
    assert st["cam_by_leg"]["RL"] == pytest.approx(0.07) and st["cam_max_abs"] == pytest.approx(0.07)
    assert st["knee_max_abs"] == pytest.approx(0.0)
    none = FID.allhold_bias(mu[25:], tgt[25:], br[25:], RST)
    assert none["allhold_rows"] == 0 and none["hip_max_abs"] is None


def test_fidelity_recomputes_R_and_refuses_without_constants(FID, RST, tmp_path):
    actor, est = _tiny_actor()
    bank_p = tmp_path / "bank" / "p2r_stand_bank.npz"
    # the bank's stored target was written under OTHER constants (a large trim): recompute must not read it
    _bank(RST, bank_p, stored_restance="trim=0.5,k_h=1")
    ck = _save_run(tmp_path / "run", actor, est, restance="default")
    out = FID.run(bank_p, ck)                                        # agent.yaml algorithm.restance (a dict)
    assert out["target_source"].endswith("algorithm.restance")
    assert out["restance_scored"] == RST.format_restance(RST.RestanceParams())
    assert set(out["paths"]) == {"eval", "train"}
    assert out["paths"]["eval"]["hist_encoding"] is True and out["paths"]["train"]["hist_encoding"] is False
    assert out["allhold"]["allhold_rows"] == 50 and out["paths"]["eval"]["allhold_rows"] == 50
    # the paths' eval block equals the primary (default flags = the eval path)
    assert out["paths"]["eval"]["cam_by_leg"] == pytest.approx(out["allhold"]["cam_by_leg"])
    stored = FID.run(bank_p, ck, stored_target=True)
    assert stored["target_source"].startswith("the bank's stored") and stored["restance_scored"] is None
    # an explicit --restance with a wide re-step dead band (eps_step 2 > every cam): every row is all-HOLD now
    wide = FID.run(bank_p, ck, restance="eps_step=2")
    assert wide["allhold"]["allhold_rows"] == 80 and wide["branch_vs_stored_share"] > 0
    ck2 = _save_run(tmp_path / "run_noR", actor, est, restance=None)
    with pytest.raises(ValueError, match="no R constants"):
        FID.run(bank_p, ck2)
    assert FID.main(["--bank", str(bank_p), "--ckpt", str(ck2)]) == FID.RC_ERROR
    assert FID.main(["--bank", str(bank_p), "--ckpt", str(ck2), "--stored-target"]) in (FID.RC_OK, FID.RC_NOTIFY)
    assert FID.main(["--bank", str(bank_p), "--ckpt", str(ck2), "--stored-target", "--restance", "default"]) == \
        FID.RC_ERROR


def test_fidelity_train_path_is_the_true_priv_no_history_forward(FID, RST, tmp_path):
    actor, est = _tiny_actor(seed=4)
    bank_p = tmp_path / "b.npz"
    bank = _bank(RST, bank_p, n=30, n_hold=30)
    ck = _save_run(tmp_path / "run", actor, est)
    out = FID.run(bank_p, ck)
    with torch.no_grad():
        mu = actor(torch.as_tensor(bank["obs"]), False)              # the REAL Actor, training path
    want = (mu - torch.as_tensor(bank["target"]))[:, 6:12].double().mean(0)
    got = [out["paths"]["train"]["hip_by_leg"][leg] for leg in RST.LEGS]
    assert got == pytest.approx(want.tolist(), abs=1e-5)


# ===================================================================================== the gate
def test_gate_judge_lines(GATE):
    base = {"allhold_rows": 10, "hip_max_abs": 0.01, "cam_max_abs": 0.01}
    assert GATE.judge(base)["rc"] == GATE.RC_PASS
    assert GATE.judge(dict(base, hip_max_abs=0.05))["rc"] == GATE.RC_PASS               # the line itself passes
    assert GATE.judge(dict(base, hip_max_abs=0.0501))["rc"] == GATE.RC_FAIL
    assert GATE.judge(dict(base, cam_max_abs=0.06))["rc"] == GATE.RC_FAIL
    assert GATE.judge(dict(base, allhold_rows=0))["rc"] == GATE.RC_ERROR
    assert GATE.judge(dict(base, cam_max_abs=float("nan")))["rc"] == GATE.RC_ERROR


@pytest.mark.parametrize("bias,rc", [({}, 0), ({6 + 1: -0.13}, 3), ({3: 0.06}, 3), ({12 + 0: 0.3}, 0),
                                     ({6 + 1: -0.04, 3: 0.049}, 0)])
def test_gate_exit_codes_on_a_constant_actor(GATE, RST, tmp_path, bias, rc):
    """mu == const on every row; R == 0 on the all-HOLD rows (cams 0, hips at default): the statistic IS the bias.
    The knee (col 12+) is reported, never gated."""
    actor, est = _tiny_actor()
    mu = np.zeros(18, np.float32)
    for k, v in bias.items():
        mu[k] = v
    bank_p = tmp_path / "es2" / "p2r_stand_bank.npz"
    _bank(RST, bank_p)
    run = tmp_path / "run"
    _save_run(run, actor, est, const_mu=mu)
    j = tmp_path / "gate.json"
    got = GATE.main(["--run-dir", str(run), "--bank", str(bank_p), "--json", str(j)])
    out = json.loads(j.read_text())
    assert got == rc == out["exit"], out
    assert out["verdict"] == GATE.VERDICT[rc] and out["eval"]["allhold_rows"] == 50
    assert out["train_reported"]["allhold_rows"] == 50 and out["stand_bit_share"] == 1.0
    assert out["eval"]["hip_max_abs"] == pytest.approx(max(abs(mu[6:12])), abs=1e-6)
    assert out["eval"]["cam_max_abs"] == pytest.approx(max(abs(mu[0:6])), abs=1e-6)


def test_gate_wait_error_heldout_and_ckpt_state(GATE, RST, tmp_path, monkeypatch, capsys):
    actor, est = _tiny_actor()
    bank_p = tmp_path / "es2" / "p2r_stand_bank.npz"
    _bank(RST, bank_p)
    run = tmp_path / "run"
    run.mkdir()
    # WAIT: model_20500.pt not there yet (bank from the env, as run_arm passes it)
    monkeypatch.setenv(GATE.BANK_ENV, str(bank_p))
    assert GATE.main(["--run-dir", str(run)]) == GATE.RC_WAIT
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["verdict"] == "WAIT"
    monkeypatch.delenv(GATE.BANK_ENV)
    assert GATE.main(["--run-dir", str(run)]) == GATE.RC_ERROR                         # no bank at all
    # a FAILING run, then the same base scored with an offline state that fixes the hip -> PASS
    bad = np.zeros(18, np.float32)
    bad[6:12] = -0.15
    ck = _save_run(run, actor, est, const_mu=bad)
    assert GATE.main(["--run-dir", str(run), "--bank", str(bank_p)]) == GATE.RC_FAIL
    sd = torch.load(ck, weights_only=False)["model_state_dict"]
    last = _last_layer(sd)
    st = tmp_path / "offline.pt"
    torch.save({"model_state_dict": {last + ".bias": torch.zeros(18)}}, st)
    assert GATE.main(["--run-dir", str(run), "--bank", str(bank_p), "--ckpt-state", str(st)]) == GATE.RC_PASS
    torch.save({"actor.nope.weight": torch.zeros(3)}, st)                               # an unknown key: error
    assert GATE.main(["--run-dir", str(run), "--bank", str(bank_p), "--ckpt-state", str(st)]) == GATE.RC_ERROR
    # a base checkpoint without a run dir (the GRAFT): --restance + --agent-yaml carry the constants and widths
    graft = tmp_path / "graft" / "model_19994.pt"
    graft.parent.mkdir()
    graft.write_bytes(ck.read_bytes())
    assert GATE.main(["--ckpt", str(graft), "--bank", str(bank_p)]) == GATE.RC_ERROR   # no constants anywhere
    assert GATE.main(["--ckpt", str(graft), "--bank", str(bank_p), "--restance", "default",
                      "--agent-yaml", str(run / "params" / "agent.yaml")]) == GATE.RC_FAIL
    # the gate bank among the run's BC banks: refused (not held out)
    run2 = tmp_path / "run2"
    _save_run(run2, actor, est, bc_bank=[bank_p])
    j = tmp_path / "g2.json"
    assert GATE.main(["--run-dir", str(run2), "--bank", str(bank_p), "--json", str(j)]) == GATE.RC_ERROR
    assert "not held out" in json.loads(j.read_text())["why"]
    # a bank with no all-HOLD row cannot be judged
    nohold = tmp_path / "nohold.npz"
    _bank(RST, nohold, n=20, n_hold=0)
    assert GATE.main(["--run-dir", str(run), "--bank", str(nohold)]) == GATE.RC_ERROR


# ===================================================================================== the BC dose
def test_bc_dose_is_the_forks_training_path_loss_per_pooled_row(D, RST, tmp_path):
    actor, est = _tiny_actor(seed=7)
    ck = _save_run(tmp_path / "run", actor, est, name="graft.pt", std_raw=0.3)
    b1, b2 = tmp_path / "b1.npz", tmp_path / "b2.npz"
    o1 = _bank(RST, b1, n=30, n_hold=10, seed=1)["obs"]
    o2 = _bank(RST, b2, n=50, n_hold=40, seed=2)["obs"]
    p = RST.parse_restance("trim=0.1,k_h=1")
    out = D.bc_dose([b1, b2], ck, p, rst=RST)
    sigma = D.anchor_sigma({"std_raw": torch.full((18,), 0.3)})
    rows = []
    for o in (o1, o2):
        o = torch.as_tensor(o)
        with torch.no_grad():
            mu = actor(o, False)                                       # the REAL Actor, training path
        tgt, _ = RST.restance_target(o, p)
        rows.append(0.5 * (((mu - tgt) / sigma) ** 2).sum(1))
    want = float(torch.cat(rows).double().mean())
    assert out["bc_mb0_expected"] == pytest.approx(want, rel=1e-5)
    assert out["rows"] == 80 and [b["rows"] for b in out["banks"]] == [30, 50]
    assert out["banks"][0]["per_row"]["mean"] == pytest.approx(float(rows[0].double().mean()), rel=1e-5)
    assert out["bc_mb0_band"] == pytest.approx([0.52 * want, 2.0 * want], rel=1e-6)
    assert out["restance_knob"] == RST.format_restance(p)
    assert out["banks"][0]["target_vs_stored_max_abs"] == 0.0                   # cams 0 / out of band: same R
    bad = tmp_path / "walking.npz"
    _bank(RST, bad, stand_bit=0.0)
    with pytest.raises(ValueError, match="stand bit"):
        D.bc_dose([b1, bad], ck, p, rst=RST)
    with pytest.raises(ValueError):
        D.bc_dose([], ck, p, rst=RST)


def test_dose_cli_writes_bc_mb0_expected(D, RST, tmp_path):
    zp = _load(Z2_PROBE_TESTS, "_t_z2c_from_z2_probe")
    stamp = zp._dose_stamp(tmp_path, D)
    actor, est = _tiny_actor(seed=2)
    ck = _save_run(tmp_path / "run", actor, est, name="graft.pt")
    b1 = tmp_path / "b1.npz"
    _bank(RST, b1)
    j = tmp_path / "dose.json"
    rc = D.main(["--raws", str(stamp), "--ckpt", str(ck), "--restance", "default", "--xcheck-stamp", "none",
                 "--bc-bank", str(b1), "--json", str(j), "--md", str(tmp_path / "dose.md")])
    out = json.loads(j.read_text())
    assert rc == out["rc"]
    assert out["bc_mb0_expected"] == pytest.approx(D.bc_dose([b1], ck, RST.RestanceParams(), rst=RST)["bc_mb0_expected"])
    assert out["bc"]["banks"][0]["path"] == str(b1.resolve()) and "band_lo" in out
    assert "BC dose (Z2c)" in (tmp_path / "dose.md").read_text()
    # no --bc-bank: no BC keys (the Z2 dose file is unchanged)
    j2 = tmp_path / "dose2.json"
    D.main(["--raws", str(stamp), "--ckpt", str(ck), "--restance", "default", "--xcheck-stamp", "none",
            "--json", str(j2)])
    assert "bc_mb0_expected" not in json.loads(j2.read_text())
    # an unreadable BC bank is an error, not a silent omission
    assert D.main(["--raws", str(stamp), "--ckpt", str(ck), "--restance", "default", "--xcheck-stamp", "none",
                   "--bc-bank", str(tmp_path / "nope.npz")]) == D.RC_ERROR


# ===================================================================================== p2_live: Z2c wiring
def test_z_arm_reads_z2c(live):
    assert live.z_arm("P2__Z2c_s18") == "Z2C" and live.z_arm("P2__Z2cR1_s16") == "Z2C"
    assert live.z_arm("P2__Z2_s18") == "Z2" and live.z_arm("P2__Z1R2_s16") == "Z1"
    assert live.z_arm("P2__Z2C_s18") == "" and live.arm_of("P2__Z2c_s18") == ""
    assert live.Z_ARM_RE.match("Z2C") and live.Z_ARM_RE.match("Z1")


F0 = 19995


def _ov(live, banks, **over):
    rst = live.restance_mod()
    ov = {"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_ACTION_RATE_W": "-0.05", "KRABBY_OMNI_STAND_OBS": "clock",
          "KRABBY_ANCHOR_W": "1.0", "KRABBY_STAND_PRIOR_W": "1.0",
          "KRABBY_ANCHOR_CKPT": "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt",
          "KRABBY_STAND_PRIOR_TARGET": "restance",
          "KRABBY_RESTANCE": rst.format_restance(rst.parse_restance("trim=0.1,k_h=1")),
          "KRABBY_BC_W": "1.0", "KRABBY_BC_BANK": ",".join(str(b) for b in banks), "KRABBY_BC_BATCH": "2048"}
    ov.update(over)
    return {k: v for k, v in ov.items() if v is not None}


def _dose(live, banks, exp=8.9, **over):
    rst = live.restance_mod()
    d = {"band_lo": 1.0, "band_hi": 5.4, "rc": 0,
         "restance_knob": rst.format_restance(rst.parse_restance("trim=0.1,k_h=1")),
         "bc_mb0_expected": exp, "bc": {"banks": [{"path": str(Path(b).resolve())} for b in banks]}}
    d.update(over)
    return {k: v for k, v in d.items() if v is not None}


def _series(live, *, bc_mb0=8.9, loss_bc=3.0, drop=(), prior_first=2.0, prior_mb0=None, n=215):
    s = {live.TAGS["reach_obst"]: 0.3, live.TAGS["clock_hidden"]: 1.0, "Episode_Reward/reward_action_rate": -0.004,
         live.P2R_ANCHOR_ROWS_TAG: 0.88, live.P2R_PRIOR_ROWS_TAG: 0.12, live.P2R_LOSS_ANCHOR: 0.01,
         live.P2R_RESTANCE_NONZERO_TAG: 0.9, live.P2R_LOSS_BC: loss_bc}
    s.update(dict(zip(live.P2R_RESTANCE_BRANCH_TAGS, (0.6, 0.1, 0.1, 0.05, 0.15))))
    ser = {t: [(F0 + i, v) for i in range(n)] for t, v in s.items() if t not in drop}
    ser[live.P2R_ANCHOR_MB0_TAG] = [(F0 + i, 0.0 if i == 0 else 0.05) for i in range(n)]
    ser[live.P2R_LOSS_PRIOR] = [(F0 + i, prior_first if i == 0 else 0.4) for i in range(n)]
    if bc_mb0 is not None and live.P2R_BC_MB0_TAG not in drop:
        ser[live.P2R_BC_MB0_TAG] = [(F0 + i, bc_mb0 if i == 0 else 1.0) for i in range(n)]
    if prior_mb0 is not None:
        ser[live.P2R_PRIOR_MB0_TAG] = [(F0 + i, prior_mb0 if i == 0 else 0.5) for i in range(n)]
    return ser


def _wire(live, ov, ser, dose):
    w = dict(live._facts()["weights"], reward_action_rate=-0.05)
    audit = [("overlay KRABBY_TERRAIN_PROMOTE=0.75:0.4 in params/env.yaml", True)]
    return live.judge_wiring("Z2C", ov, live._facts(weights=w), ser, F0, audit, None, z2_dose=dose)


@pytest.fixture()
def banks(tmp_path):
    out = []
    for n in ("ws_es1.npz", "suite_es1.npz"):
        p = tmp_path / n
        p.write_bytes(b"npz")
        out.append(p)
    return out


def test_z2c_wiring_clean_passes(live, banks):
    r = _wire(live, _ov(live, banks), _series(live), _dose(live, banks))
    assert r["rc"] == live.RC_PASS, r["fails"]
    assert r["liveness"]["bc_mb0_first_update"] == 8.9 and r["liveness"]["bc_dose_band"] == pytest.approx(
        [0.52 * 8.9, 2 * 8.9])
    assert any("stand_prior_mb0 not logged" in n for n in r["notes"])


@pytest.mark.parametrize("case", ["bc_mb0_low", "bc_mb0_high", "bc_mb0_band_edge_ok", "no_loss_bc", "no_bc_mb0",
                                  "loss_bc_zero", "loss_bc_nan", "no_expected", "no_dose_file", "dose_other_banks",
                                  "bc_w_zero", "bank_missing", "no_batch", "batch_bad", "z2_band_still_read"])
def test_z2c_wiring_cases(live, banks, tmp_path, case):
    ov, ser, dose = _ov(live, banks), _series(live), _dose(live, banks)
    want = live.RC_FAIL
    if case == "bc_mb0_low":
        ser = _series(live, bc_mb0=0.51 * 8.9)
    elif case == "bc_mb0_high":
        ser = _series(live, bc_mb0=2.01 * 8.9)
    elif case == "bc_mb0_band_edge_ok":
        ser, want = _series(live, bc_mb0=0.52 * 8.9), live.RC_PASS
    elif case == "no_loss_bc":
        ser = _series(live, drop=(live.P2R_LOSS_BC,))
    elif case == "no_bc_mb0":
        ser = _series(live, drop=(live.P2R_BC_MB0_TAG,))
    elif case == "loss_bc_zero":
        ser = _series(live, loss_bc=0.0)
    elif case == "loss_bc_nan":
        ser = _series(live, loss_bc=float("nan"))
    elif case == "no_expected":
        dose = _dose(live, banks, exp=None)
    elif case == "no_dose_file":
        dose = None
    elif case == "dose_other_banks":
        dose = _dose(live, banks[:1])
    elif case == "bc_w_zero":
        ov = _ov(live, banks, KRABBY_BC_W="0")
    elif case == "bank_missing":
        ov = _ov(live, banks + [tmp_path / "nope.npz"])
        dose = _dose(live, banks + [tmp_path / "nope.npz"])
    elif case == "no_batch":
        ov = _ov(live, banks, KRABBY_BC_BATCH=None)
    elif case == "batch_bad":
        ov = _ov(live, banks, KRABBY_BC_BATCH="-5")
    elif case == "z2_band_still_read":                                # Z2's stand-prior band still applies
        ser = _series(live, prior_first=9.0)
    r = _wire(live, ov, ser, dose)
    assert r["rc"] == want, (case, r.get("fails"))


def test_z2c_stand_prior_dose_reads_its_own_mb0_when_logged(live, banks):
    # BC's in-update pre-fit drags the update's MEAN prior below the band; the mb0 value (before any step) is in it
    ser = _series(live, prior_first=0.5, prior_mb0=2.0)
    r = _wire(live, _ov(live, banks), ser, _dose(live, banks))
    assert r["rc"] == live.RC_PASS, r["fails"]
    assert r["liveness"]["stand_prior_dose_tag"] == live.P2R_PRIOR_MB0_TAG
    assert _wire(live, _ov(live, banks), _series(live, prior_first=0.5), _dose(live, banks))["rc"] == live.RC_FAIL
    # Z2 itself never reads the mb0 tag
    w = dict(live._facts()["weights"], reward_action_rate=-0.05)
    audit = [("overlay KRABBY_TERRAIN_PROMOTE=0.75:0.4 in params/env.yaml", True)]
    z2 = live.judge_wiring("Z2", _ov(live, banks), live._facts(weights=w), ser, F0, audit, None,
                           z2_dose=_dose(live, banks))
    assert z2["rc"] == live.RC_FAIL


def test_z2c_preflight(live, banks):
    em = {"terrain_levels_obst": ["x.py"], "promote_count": ["x.py"]}
    graft = {"identity_cpu": True, "identity_gpu_stamps": ["GRAFT_S1_s18/suite__omni/seed001/x"]}
    ov = _ov(live, banks)
    fk = ov["KRABBY_ANCHOR_CKPT"]

    def pf(ov_, dose_):
        return live.judge_preflight("Z2C", em, None, None, ov_, p2r_graft=graft, fork_ckpt=fk, z2_dose=dose_)
    assert pf(ov, _dose(live, banks))["rc"] == live.RC_PASS
    assert pf(ov, _dose(live, banks, exp=None))["rc"] == live.RC_FAIL
    assert pf(ov, None)["rc"] == live.RC_FAIL
    assert pf(_ov(live, banks, KRABBY_BC_W=None), _dose(live, banks))["rc"] == live.RC_FAIL
    assert pf(_ov(live, banks, KRABBY_RESTANCE="default"), _dose(live, banks))["rc"] == live.RC_FAIL   # stale dose
    assert pf(_ov(live, banks, KRABBY_STAND_PRIOR_TARGET="zero"), _dose(live, banks))["rc"] == live.RC_FAIL
    # Z2's preflight does not ask for the BC block
    assert live.judge_preflight("Z2", em, None, None, _ov(live, banks, KRABBY_BC_W=None), p2r_graft=graft,
                                fork_ckpt=fk, z2_dose=_dose(live, banks, exp=None))["rc"] == live.RC_PASS


def test_z2c_inherits_z2s_ep_length_notify_only_guard(live, tmp_path):
    """check(--at p2s) passes ep_len_notify for Z2C (source-level: the arm set the guard reads)."""
    src = (SCRIPTS / "p2_live.py").read_text()
    # NOTE(p2r Z2d, 2026-10-11): the arm set is the Z2 family (Z2 / Z2C / Z2D)
    assert "ep_len_notify=(arm in P2R_Z2_FAMILY)" in src
    assert live.P2R_Z2_FAMILY == ("Z2", "Z2C", "Z2D")
    EL, CF, SD = live.TAGS["ep_len"], live.TAGS["crab_failure"], live.TAGS["std"]
    ref = {"kind": "run", "name": "P2__C0_s18", "first": 19994,
           "ser": {t: [(19994 + i, v) for i in range(5000)] for t, v in ((EL, 1800.0), (CF, 0.2), (SD, 0.82))}}
    arm = {t: [(19994 + i, v) for i in range(400)] for t, v in ((EL, 1500.0), (CF, 0.2), (SD, 0.82))}
    assert live.judge_p2s_guards(arm, 19994, ref, ep_len_notify=True)["rc"] == live.RC_NOTIFY


def test_read_tb_requests_the_bc_tags(live):
    src = (SCRIPTS / "p2_live.py").read_text()
    assert "{P2R_LOSS_BC, P2R_BC_MB0_TAG, P2R_GRAD_NORM_BC_TAG, P2R_PRIOR_MB0_TAG}" in src
    assert (live.P2R_LOSS_BC, live.P2R_BC_MB0_TAG, live.P2R_GRAD_NORM_BC_TAG) == (
        "Loss/bc", "Loss/bc_mb0", "Loss/grad_norm_bc")
    assert (live.P2R_BC_W, live.P2R_BC_BANK, live.P2R_BC_BATCH) == ("KRABBY_BC_W", "KRABBY_BC_BANK",
                                                                     "KRABBY_BC_BATCH")


def test_p2_live_self_test_still_passes():
    r = subprocess.run([sys.executable, str(SCRIPTS / "p2_live.py"), "--self-test"], capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0 and "self-test: PASS" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


# ===================================================================================== run_arm.sh: the Z2c gate
def test_run_arm_parses_and_keeps_the_other_arms_blocks():
    assert subprocess.run(["bash", "-n", str(RUN_ARM)], capture_output=True).returncode == 0
    src = RUN_ARM.read_text()
    assert "P2__ST[0-9]*_s*|P2__Z[0-9]*_s*)" in src and src.count("--preflight --chain") == 1
    assert "p2r_z2c_gate.py" in src and "P2_Z2C_GATE_BANK" in src and "SIGKILL" not in src.split("kill_arm()")[1][:600]


def _watch_env(tmp: Path, gate_rc: int, chain="P2__Z2c_s18", make_ckpt=True, age=0):
    """run_arm.sh sourced with stubs: p2_live (--iters -> '19995 20500', every read 0), a gate stub exiting
    ``gate_rc`` and logging argv, systemctl active for 2 polls then inactive."""
    bin_ = tmp / "bin"
    bin_.mkdir()
    (bin_ / "systemctl").write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        n=$(cat {tmp}/sc.n 2>/dev/null || echo 0); n=$((n+1)); echo $n > {tmp}/sc.n
        case "$*" in *is-active*) [ "$n" -le 2 ] && exit 0 || exit 3 ;; esac
        echo "$@" >> {tmp}/systemctl.calls; exit 0
        """))
    (bin_ / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")
    for b in ("systemctl", "sleep"):
        (bin_ / b).chmod(0o755)
    scripts = tmp / "scripts"
    scripts.mkdir()
    (scripts / "p2_live.py").write_text(textwrap.dedent("""\
        import sys
        a = sys.argv[1:]
        if '--iters' in a: print('19995 20500'); sys.exit(0)
        print('{}'); sys.exit(0)
        """))
    (scripts / "p2r_z2c_gate.py").write_text(textwrap.dedent(f"""\
        import os, sys
        open({str(tmp / 'gate.calls')!r}, 'a').write(' '.join(sys.argv[1:]) + ' CUDA=' +
                                                       repr(os.environ.get('CUDA_VISIBLE_DEVICES')) + chr(10))
        print('{{"verdict": "stub", "exit": {gate_rc}}}'); sys.exit({gate_rc})
        """))
    rd = tmp / "rd"
    rd.mkdir()
    if make_ckpt:
        ck = rd / "model_20500.pt"
        ck.write_bytes(b"x")
        t = time.time() - age
        os.utime(ck, (t, t))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2_"))}
    camp = tmp / "camp"
    (camp / "logs").mkdir(parents=True)
    env.update(P2_OUT=str(tmp / "out"), CAMPAIGN_DIR=str(camp), P2_CAP="", P2_PY=sys.executable,
               P2_RUN_ARM_LIB="1", PATH=f"{bin_}:{env['PATH']}", P2_Z2C_GATE_BANK=str(tmp / "es2.npz"))
    script = f"""
        source {RUN_ARM}
        S={scripts}
        PREC={tmp}
        fork_source_rd() {{ echo {tmp}/src; }}
        printf 'import sys\\nprint("{{}}")\\nsys.exit(2)\\n' > {tmp}/resume_read.py
        watch_arm {chain} {rd} unit-x {chain} {tmp}/train.log
        echo "watch rc=$?"
    """
    return env, script


def _gate_run(tmp, gate_rc, **kw):
    env, script = _watch_env(tmp, gate_rc, **kw)
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert "watch rc=0" in r.stdout, r.stdout + r.stderr
    log = (tmp / "out" / "queue.log").read_text() if (tmp / "out" / "queue.log").exists() else ""
    calls = (tmp / "gate.calls").read_text().splitlines() if (tmp / "gate.calls").exists() else []
    return log, calls


def test_run_arm_z2c_gate_rc3_kills_through_the_live_read_path(tmp_path):
    log, calls = _gate_run(tmp_path, 3, age=120)
    assert len(calls) == 1 and f"--bank {tmp_path / 'es2.npz'}" in calls[0] and "CUDA=''" in calls[0]
    assert f"--run-dir {tmp_path / 'rd'}" in calls[0] and "--json" in calls[0]
    assert "Z2CGATE P2__Z2c_s18 rc 3 FAIL" in log and "KILL P2__Z2c_s18 (z2c-fidelity-gate at model_20500.pt" in log
    assert "NOTIFY P2__Z2c_s18 killed for z2c-fidelity-gate" in log
    assert (tmp_path / "out" / "STOP_QUEUE").exists()
    assert "stop unit-x" in (tmp_path / "systemctl.calls").read_text()
    assert (tmp_path / "out" / "arms" / "P2__Z2c_s18.killed").read_text().startswith("z2c-fidelity-gate\t")


def test_run_arm_z2c_gate_pass_runs_once_and_continues(tmp_path):
    log, calls = _gate_run(tmp_path, 0, age=120)
    assert len(calls) == 1                                           # once, not every poll
    assert "Z2CGATE P2__Z2c_s18 PASS model_20500.pt" in log and "KILL" not in log
    assert not (tmp_path / "out" / "STOP_QUEUE").exists() and not (tmp_path / "systemctl.calls").exists()


def test_run_arm_z2c_gate_error_notifies_once_and_continues(tmp_path):
    log, calls = _gate_run(tmp_path, 1, age=120)
    assert len(calls) == 1 and log.count("NOTIFY Z2CGATE P2__Z2c_s18 rc 1") == 1
    assert "KILL" not in log and not (tmp_path / "out" / "STOP_QUEUE").exists()


def test_run_arm_z2c_gate_waits_for_a_whole_checkpoint(tmp_path):
    log, calls = _gate_run(tmp_path, 3, age=0)                       # just written: not read yet
    assert calls == [] and "Z2CGATE" not in log and "KILL" not in log


def test_run_arm_gate_never_runs_for_other_arms(tmp_path):
    for chain in ("P2__Z2_s18", "P2__ST2_s18"):
        sub = tmp_path / chain
        sub.mkdir()
        log, calls = _gate_run(sub, 3, chain=chain, age=120)
        assert calls == [] and "Z2CGATE" not in log, chain


def test_run_arm_refuses_a_z2c_chain_without_a_held_out_gate_bank(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2_"))}
    camp = tmp_path / "camp"
    (camp / "logs").mkdir(parents=True)
    bank = tmp_path / "es2.npz"
    bank.write_bytes(b"x")
    es1 = tmp_path / "es1.npz"
    es1.write_bytes(b"x")
    (camp / "candidates.yaml").write_text(json.dumps({"chains": {"P2__Z2c_s18": {"stages": [
        {"overlay": {"KRABBY_BC_BANK": f"{es1},{bank}"}}]}, "P2__Z2cB_s18": {"stages": [
        {"overlay": {"KRABBY_BC_BANK": str(es1)}}]}}}))
    env.update(P2_OUT=str(tmp_path / "out"), CAMPAIGN_DIR=str(camp), P2_CAP="", P2_PY=sys.executable,
               P2_RUN_ARM_LIB="1")

    def refusal(chain, gate_bank):
        e = dict(env)
        if gate_bank is not None:
            e["P2_Z2C_GATE_BANK"] = str(gate_bank)
        r = subprocess.run(["bash", "-c", f"source {RUN_ARM}\nz2c_gate_refusal {chain}"], env=e,
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip()
    assert "is not set" in refusal("P2__Z2c_s18", None)
    assert "is not a file" in refusal("P2__Z2c_s18", tmp_path / "nope.npz")
    assert "one of the chain's KRABBY_BC_BANK" in refusal("P2__Z2c_s18", bank)
    assert refusal("P2__Z2cB_s18", bank) == ""
    src = RUN_ARM.read_text()
    run_ref = src.split("run_ref() {")[1].split("\n}\n")[0]
    # the refusal sits before the launch, with the other refused-before-launch exits (7)
    assert run_ref.index("z2c_gate_refusal") < run_ref.index('launch "$ref"')
    assert 'Z2c fidelity gate :: ' in run_ref and "return 7" in run_ref.split("z2c_gate_refusal")[1][:300]


def test_dose_file_restance_is_compared_as_parsed_constants(live):
    """A dose file written before eps_step joined format_restance names the same R as the chain's new spelling."""
    rst = live.restance_mod()
    old = "lift=-0.5,rho=1,up_frac=0.5,down_frac=0.4,k_c=2,k_h=0.3,trim=0.03,eps_c=0.1,eps_air=0.05,tilt_max=0.1"
    assert live.judge_z2_dose_matches({"KRABBY_RESTANCE": old}, {"restance_knob": old}) == []
    new = rst.format_restance(rst.parse_restance(old))
    assert live.judge_z2_dose_matches({"KRABBY_RESTANCE": new}, {"restance_knob": old}) == []
    assert live.judge_z2_dose_matches({"KRABBY_RESTANCE": old + ",eps_step=0.3"}, {"restance_knob": old})
    assert live.judge_z2_dose_matches({"KRABBY_RESTANCE": old}, {"restance_knob": "theta=1"})
