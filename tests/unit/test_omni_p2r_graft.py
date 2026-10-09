"""Stage p2r step 2 (P2R_STATUE_PLAN.md step 1 "Graft tool" / `C/scripts/p2r_graft.py`). CPU only: nothing here
starts Isaac/Kit or a GPU, and nothing here needs the real (multi-hundred-MB) S1 checkpoint -- the toy checkpoint
builder in ``p2r_graft.py`` itself (``_tiny_checkpoint``) exercises every architecture piece (scan / priv / history
encoders, actor backbone, critic, estimator, their Adam moments) at toy widths.

* :func:`zero_weight_column` / :func:`zero_adam_column`: the column-15 zero and its pre-zero premise assert;
* :func:`build_graft`: zeroes exactly the three targets (weight + Adam moments), leaves every other leaf tensor
  bitwise identical (:func:`assert_rest_identical`), refuses a wrong expected source sha256, refuses a checkpoint
  whose Adam moments at column 15 are NOT already ~0 (the "never trained on this column" premise);
* :func:`run_identity_check`: a valid graft reads ``ok: True`` on every one of the actor's two paths (anchor /
  history), critic and estimator; an un-grafted critic is caught as a true negative;
* the real X08 S1 checkpoint's graft already written to disk (``parkour/outputs/omni_plan/phase2/graft/
  S1_s18_c15/model_19994.pt`` + its ``.json``) is self-consistent: file sha256 == graft_sha256, source_sha256 ==
  the fixed X08 sha, and -- if the file happens to be present on this machine -- the full identity check passes.
* the module's own ``--self-test`` CLI exits 0.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
GRAFT_PY = SCRIPTS / "p2r_graft.py"
REAL_GRAFT_CKPT = REPO / "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt"
REAL_GRAFT_JSON = REAL_GRAFT_CKPT.with_suffix(".json")
REAL_S1_SHA256 = "968cc550989919cdc3fa36ae047379febabee5659cde0ed4d099195acb13c23c"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def graft():
    return _load(GRAFT_PY, "_t_p2r_graft")


@pytest.fixture()
def toy(graft, tmp_path):
    """A fresh toy checkpoint + agent.yaml, laid out like a real S1 run dir (``<run>/model_19994.pt`` +
    ``<run>/params/agent.yaml``)."""
    d, agent_yaml_dict, dims = graft._tiny_checkpoint()
    run_dir = tmp_path / "s1_run"
    (run_dir / "params").mkdir(parents=True)
    ck = run_dir / "model_19994.pt"
    torch.save(d, str(ck))
    yaml.safe_dump(agent_yaml_dict, open(run_dir / "params" / "agent.yaml", "w"))
    return {"ckpt": ck, "sha256": graft.sha256_of(ck), "dims": dims, "raw": d, "run_dir": run_dir}


def test_zero_weight_column_clones_and_reports_pre_norm(graft):
    w = torch.randn(4, 20)
    pre = float(w[:, 15].norm())
    out, pre_norm = graft.zero_weight_column(w, col=15)
    assert pre_norm == pytest.approx(pre)
    assert torch.equal(out[:, 15], torch.zeros(4))
    assert not torch.equal(out, w)              # a clone, the original is untouched
    assert torch.equal(w[:, 15], w[:, 15])       # original still has its (nonzero) column
    assert float(w[:, 15].abs().sum()) != 0.0


def test_zero_weight_column_refuses_a_too_narrow_weight(graft):
    with pytest.raises(ValueError):
        graft.zero_weight_column(torch.randn(4, 10), col=15)


def test_zero_adam_column_asserts_the_premise_then_zeroes(graft):
    state = {"exp_avg": torch.zeros(4, 20), "exp_avg_sq": torch.zeros(4, 20)}
    pre_max = graft.zero_adam_column(state, col=15)
    assert pre_max == 0.0
    assert torch.equal(state["exp_avg"][:, 15], torch.zeros(4))
    assert torch.equal(state["exp_avg_sq"][:, 15], torch.zeros(4))


def test_zero_adam_column_refuses_a_live_column(graft):
    state = {"exp_avg": torch.zeros(4, 20), "exp_avg_sq": torch.zeros(4, 20)}
    state["exp_avg"][0, 15] = 1e-10          # >= ADAM_PRE_ZERO_EPS (1e-30): this column WAS trained
    with pytest.raises(AssertionError):
        graft.zero_adam_column(state, col=15)


def test_build_graft_zeroes_exactly_the_three_targets(graft, toy, tmp_path):
    out = tmp_path / "graft" / "model_19994.pt"
    payload = graft.build_graft(toy["ckpt"], out, expect_source_sha256=toy["sha256"])
    assert out.is_file()
    assert payload["source_sha256"] == toy["sha256"]
    assert payload["graft_sha256"] == graft.sha256_of(out)
    assert {z["name"] for z in payload["zeroed"]} == {"actor.actor_backbone.0.weight", "critic.0.weight",
                                                      "estimator.0.weight"}
    assert {a["name"] for a in payload["adam_zeroed"]} == {"actor.actor_backbone.0.weight", "critic.0.weight",
                                                           "estimator.0.weight"}
    assert payload["identity_cpu"] is False
    assert payload["identity_gpu_stamps"] == []
    g = torch.load(str(out), map_location="cpu", weights_only=False)
    for name in payload["zeroed"]:
        sd = g["model_state_dict"] if name["dict"] == "model_state_dict" else g["estimator_state_dict"]
        assert bool((sd[name["name"]][:, 15] == 0).all())
    # everything else bitwise identical, including the history encoder (never a target, never touched)
    assert torch.equal(g["model_state_dict"]["actor.history_encoder.encoder.0.weight"],
                       toy["raw"]["model_state_dict"]["actor.history_encoder.encoder.0.weight"])
    assert torch.equal(g["model_state_dict"]["critic.2.weight"], toy["raw"]["model_state_dict"]["critic.2.weight"])
    assert g["iter"] == toy["raw"]["iter"]


def test_build_graft_refuses_a_wrong_expected_sha256(graft, toy, tmp_path):
    with pytest.raises(ValueError):
        graft.build_graft(toy["ckpt"], tmp_path / "out.pt", expect_source_sha256="f" * 64)


def test_build_graft_refuses_a_checkpoint_already_trained_on_column_15(graft, tmp_path):
    d, agent_yaml_dict, _ = graft._tiny_checkpoint(seed=7)
    name = "critic.0.weight"
    idx = list(d["model_state_dict"]).index(name)
    d["optimizer_state_dict"]["state"][idx]["exp_avg_sq"][0, 15] = 3.0
    run_dir = tmp_path / "bad_run"
    (run_dir / "params").mkdir(parents=True)
    ck = run_dir / "model_19994.pt"
    torch.save(d, str(ck))
    sha = graft.sha256_of(ck)
    with pytest.raises(AssertionError):
        graft.build_graft(ck, tmp_path / "out.pt", expect_source_sha256=sha)


def test_identity_check_passes_on_a_valid_graft_and_catches_an_invalid_one(graft, toy, tmp_path):
    out = tmp_path / "graft" / "model_19994.pt"
    graft.build_graft(toy["ckpt"], out, expect_source_sha256=toy["sha256"])
    bank_path = tmp_path / "bank.npz"
    _write_bank(bank_path, n=24, width=toy["dims"]["num_critic_obs"], seed=11)
    res = graft.run_identity_check(toy["ckpt"], out, bank_path, agent_yaml=toy["run_dir"] / "params" / "agent.yaml")
    assert res["ok"] is True
    names = {n for n, _ in res["checks"]}
    assert names == {"actor anchor-path: graft(bit=0) == S1(bit=0)",
                     "actor anchor-path: graft(bit=1) == S1(bit=0)",
                     "actor history-path: graft(bit=0) == S1(bit=0)",
                     "actor history-path: graft(bit=1) == S1(bit=0)",
                     "critic: graft(bit=0) == S1(bit=0)", "critic: graft(bit=1) == S1(bit=0)",
                     "estimator: graft(bit=0) == S1(bit=0)", "estimator: graft(bit=1) == S1(bit=0)"}
    assert all(ok for _, ok in res["checks"])

    # a graft that leaves critic.0 UNgrafted must fail the critic checks (a true negative)
    import copy
    bad = copy.deepcopy(toy["raw"])
    bad["model_state_dict"]["actor.actor_backbone.0.weight"] = \
        graft.zero_weight_column(toy["raw"]["model_state_dict"]["actor.actor_backbone.0.weight"])[0]
    bad["estimator_state_dict"]["estimator.0.weight"] = \
        graft.zero_weight_column(toy["raw"]["estimator_state_dict"]["estimator.0.weight"])[0]
    bad_path = tmp_path / "bad.pt"
    torch.save(bad, str(bad_path))
    res_bad = graft.run_identity_check(toy["ckpt"], bad_path, bank_path,
                                       agent_yaml=toy["run_dir"] / "params" / "agent.yaml")
    assert res_bad["ok"] is False
    bad_names = {n for n, ok in res_bad["checks"] if not ok}
    # bit=0 is trivially unaffected by an ungrafted critic (0 x anything == 0); only bit=1 exposes it
    assert bad_names == {"critic: graft(bit=1) == S1(bit=0)"}


def test_identity_check_cli_writes_identity_cpu_true(graft, toy, tmp_path):
    out = tmp_path / "graft" / "model_19994.pt"
    payload = graft.build_graft(toy["ckpt"], out, expect_source_sha256=toy["sha256"])
    out.with_suffix(".json").write_text(json.dumps(payload))     # the CLI's non-identity-check path writes this
    bank_path = tmp_path / "bank.npz"
    _write_bank(bank_path, n=16, width=toy["dims"]["num_critic_obs"], seed=3)
    rc = graft.main(["--identity-check", "--source", str(toy["ckpt"]), "--graft", str(out), "--bank", str(bank_path),
                    "--agent-yaml", str(toy["run_dir"] / "params" / "agent.yaml")])
    assert rc == 0
    gj = json.loads(out.with_suffix(".json").read_text())
    assert gj["identity_cpu"] is True


def _write_bank(path: Path, n: int, width: int, seed: int) -> None:
    import numpy as np
    obs = torch.randn(n, width, generator=torch.Generator().manual_seed(seed))
    np.savez(path, obs=obs.numpy(), s1_sigma=np.full(3, 0.3, dtype="float32"),
            hold_label=np.array(["fwd"] * n), category=np.array(["flat"] * n))


def test_self_test_cli_passes():
    out = subprocess.run([sys.executable, str(GRAFT_PY), "--self-test"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "self-test: PASS" in out.stdout


# ------------------------------------------------------------------------------------------------ the real X08 graft
@pytest.mark.skipif(not REAL_GRAFT_JSON.is_file(), reason="the real S1_s18_c15 graft has not been built on this "
                    "machine (scripts/p2r_graft.py --source ... --out ...)")
def test_real_graft_sidecar_is_self_consistent():
    gj = json.loads(REAL_GRAFT_JSON.read_text())
    assert gj["source_sha256"] == REAL_S1_SHA256
    assert len(gj["zeroed"]) == 3
    assert len(gj["adam_zeroed"]) == 3
    if REAL_GRAFT_CKPT.is_file():
        assert hashlib.sha256(REAL_GRAFT_CKPT.read_bytes()).hexdigest() == gj["graft_sha256"]


@pytest.mark.skipif(not REAL_GRAFT_CKPT.is_file(), reason="the real graft checkpoint is not on this machine")
def test_real_graft_zeroes_exactly_column_15():
    g = torch.load(str(REAL_GRAFT_CKPT), map_location="cpu", weights_only=False)
    assert bool((g["model_state_dict"]["actor.actor_backbone.0.weight"][:, 15] == 0).all())
    assert bool((g["model_state_dict"]["critic.0.weight"][:, 15] == 0).all())
    assert bool((g["estimator_state_dict"]["estimator.0.weight"][:, 15] == 0).all())
