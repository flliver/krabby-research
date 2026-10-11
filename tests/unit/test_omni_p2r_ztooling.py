"""Stage p2r step 2's Z arm (P2R_STATUE_PLAN.md step 1 "tooling" / "Left for implementation"; CHARTER 12), the
files owned by this agent outside ``code_sha``: CPU only, nothing here starts Isaac/Kit or a GPU.

* ``scripts/run_arm.sh``: ``verify_head_sha``'s graft branch (file sha == graft.json's graft_sha256, identity_cpu
  true + a non-empty identity_gpu_stamps, then the UNDERLYING source_sha256 reused through the existing
  anchors.yaml / P2__S1_s<seed> branches verbatim), ``fork_source_rd`` resolving a graft fork to the graft's own
  SOURCE run dir, and the ``P2__ST[0-9]*_s*|P2__Z[0-9]*_s*`` guard-case pattern;
* ``scripts/p2_live.py``: ``z_arm`` / ``point_at``, the Z +200 wiring judge (the three agent knobs, anchor_rows /
  prior_rows bands, anchor_mb0 fork-exactness, both losses finite and > 0, the realised-dose band) and the Z
  preflight judge (graft identity, no p2s replay-gate dependency) -- the module's own ``--self-test`` carries the
  bulk of this; this file re-checks the handful most load-bearing for the shell / CLI integration;
* ``scripts/p2_read.py``: the clock-bit INVALID rule, the best-effort raw-bit cross-check (with its lag
  tolerance), and the fidelity-table field names;
* ``scripts/p2r_early.py``: imported and exercised directly (its own ``--self-test`` covers the pure judges in
  depth; this file adds the matched-iteration real-checkpoint smoke already validated by hand);
* ``candidates.yaml`` / ``scripts/candidates_v12.yaml``: the ``P2__Z1_s18`` chain -- stage p2r, the S1 overlay
  carried verbatim plus exactly the four new p2r knobs, and the two files staying identical.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = OMNI / "experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
RUN_ARM = SCRIPTS / "run_arm.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def live():
    return _load(SCRIPTS / "p2_live.py", "_t_p2r_z_live")


@pytest.fixture(scope="module")
def read():
    return _load(SCRIPTS / "p2_read.py", "_t_p2r_z_read")


@pytest.fixture(scope="module")
def early():
    return _load(SCRIPTS / "p2r_early.py", "_t_p2r_z_early")


# ============================================================================================= p2_live.py: Z arms
def test_z_arm_and_point_at(live):
    assert live.z_arm("P2__Z1_s18") == "Z1"
    assert live.z_arm("P2__Z1R2_s16") == "Z1"
    assert live.z_arm("P2__ST2_s18") == ""
    assert live.arm_of("P2__Z1_s18") == "Z1"
    series = [(100, 1.0), (101, 2.0), (103, 3.0)]
    assert live.point_at(series, 101) == 2.0
    assert live.point_at(series, 102) is None       # no point AT that exact step (102 was never logged)


def _z_overlay(**over):
    ov = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_ANCHOR_W": "1.0", "KRABBY_STAND_PRIOR_W": "1.0",
         "KRABBY_ANCHOR_CKPT": "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt",
         "KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_ACTION_RATE_W": "-0.05"}
    ov.update(over)
    return ov


def _z_series(live, first, anchor_rows=0.88, prior_rows=0.12, mb0_first=0.0, loss_anchor=0.01, loss_prior=2.3,
             n=215):
    ser = {live.P2R_ANCHOR_ROWS_TAG: [(first + i, anchor_rows) for i in range(n)],
          live.P2R_PRIOR_ROWS_TAG: [(first + i, prior_rows) for i in range(n)],
          live.P2R_LOSS_ANCHOR: [(first + i, loss_anchor) for i in range(n)],
          live.P2R_LOSS_PRIOR: [(first + i, loss_prior) for i in range(n)],
          live.P2R_ANCHOR_MB0_TAG: [(first + i, mb0_first if i == 0 else 0.05) for i in range(n)],
          live.TAGS["reach_obst"]: [(first + i, 0.3) for i in range(n)],
          live.TAGS["clock_hidden"]: [(first + i, 1.0) for i in range(n)]}
    return ser


def _z_facts(live):
    return {"have_env": True, "curriculum_null": True,
           "weights": dict(live.ELEMENT_WEIGHTS, reward_clock_schedule=0.001, reward_clock_swing_apex=0.001,
                           reward_action_rate=-0.05), "term_funcs": {}, "dr_events": {e: True for e in live.DR_EVENTS},
           "push_x": [-0.5, 0.5], "terrain_fns": ["omni_field_terrain"], "num_rows": 10, "dropped_ppo_keys": []}


def test_p2r_wiring_a_clean_z_arm_passes(live):
    F0 = 14996
    audit_ok = [("overlay KRABBY_TERRAIN_PROMOTE=0.75:0.4 in params/env.yaml", True)]
    v = live.judge_wiring("Z1", _z_overlay(), _z_facts(live), _z_series(live, F0), F0, audit_ok, None)
    assert v["rc"] == live.RC_PASS, v


def test_p2r_wiring_flags_each_failure_mode_independently(live):
    F0 = 14996
    audit_ok = [("overlay KRABBY_TERRAIN_PROMOTE=0.75:0.4 in params/env.yaml", True)]
    base_ser = _z_series(live, F0)

    def v(ov=None, ser=None):
        return live.judge_wiring("Z1", ov or _z_overlay(), _z_facts(live), ser or base_ser, F0, audit_ok,
                                 None)["rc"]
    assert v(ov=_z_overlay(KRABBY_ANCHOR_W="0")) == live.RC_FAIL
    assert v(ov=_z_overlay(KRABBY_STAND_PRIOR_W="-1")) == live.RC_FAIL
    assert v(ov={k: x for k, x in _z_overlay().items() if k != "KRABBY_ANCHOR_CKPT"}) == live.RC_FAIL
    assert v(ser=_z_series(live, F0, anchor_rows=0.5)) == live.RC_FAIL
    assert v(ser=_z_series(live, F0, prior_rows=0.01)) == live.RC_FAIL
    assert v(ser=_z_series(live, F0, mb0_first=0.1)) == live.RC_FAIL
    assert v(ser=_z_series(live, F0, loss_anchor=0.0)) == live.RC_FAIL
    assert v(ser=_z_series(live, F0, loss_prior=0.0)) == live.RC_FAIL      # also fails the dose band
    assert v(ser=_z_series(live, F0, loss_prior=0.9)) == live.RC_FAIL      # dose band [1.2, 4.6]
    assert v(ser=_z_series(live, F0, loss_prior=1.2)) == live.RC_PASS      # boundary


def test_p2r_preflight_graft_identity_and_knobs(live):
    good_graft = {"identity_cpu": True, "identity_gpu_stamps": ["GRAFT_S1_s18/suite__omni/seed001/x"]}
    em_ok = {"terrain_levels_obst": ["x.py"], "promote_count": ["x.py"]}
    ck = _z_overlay()["KRABBY_ANCHOR_CKPT"]
    assert live.judge_preflight("Z1", em_ok, None, None, _z_overlay(), p2r_graft=good_graft, fork_ckpt=ck)["rc"] == \
        live.RC_PASS
    assert live.judge_preflight("Z1", em_ok, None, None, _z_overlay(), p2r_graft=None, fork_ckpt=ck)["rc"] == \
        live.RC_FAIL
    assert live.judge_preflight("Z1", em_ok, None, None, _z_overlay(),
                                p2r_graft=dict(good_graft, identity_cpu=False), fork_ckpt=ck)["rc"] == live.RC_FAIL
    assert live.judge_preflight("Z1", em_ok, None, None, _z_overlay(),
                                p2r_graft=dict(good_graft, identity_gpu_stamps=[]), fork_ckpt=ck)["rc"] == \
        live.RC_FAIL
    assert live.judge_preflight("Z1", em_ok, None, None, _z_overlay(), p2r_graft=good_graft,
                                fork_ckpt="some/other/head.pt")["rc"] == live.RC_FAIL
    # no p2s replay-gate dependency at all -- never even consulted for a Z arm
    assert live.judge_preflight("Z1", em_ok, None, None, _z_overlay(), p2r_graft=good_graft, fork_ckpt=ck,
                                p2s_gate=None, p2s_ref=None)["rc"] == live.RC_PASS


def test_a_z_chain_shares_the_p2s_guards(live):
    """--at p2s dispatch (module docstring: Z arms share the three guards, read against the same reference)."""
    assert live.z_arm("P2__Z1_s18")
    EL, CF, SD = live.TAGS["ep_len"], live.TAGS["crab_failure"], live.TAGS["std"]
    ref = {"kind": "run", "name": "P2__C0_s18", "first": 19994,
          "ser": {t: [(19994 + i, v) for i in range(5000)] for t, v in ((EL, 1800.0), (CF, 0.2), (SD, 0.82))}}
    ser = {t: [(19994 + i, v) for i in range(1200)] for t, v in ((EL, 1800.0), (CF, 0.2), (SD, 0.82))}
    assert live.judge_p2s_guards(ser, 19994, ref)["rc"] == live.RC_PASS


def test_p2_live_self_test_passes():
    r = subprocess.run([sys.executable, str(SCRIPTS / "p2_live.py"), "--self-test"], capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


# ============================================================================================= p2_read.py: Z reads
def test_z_name_and_clock_invalid_reasons(read, tmp_path):
    assert read.z_name(18, env={}) == "P2__Z1_s18"
    assert read.z_name(18, env={"P2R_Z_S18": "P2__Z1R2_s18"}) == "P2__Z1R2_s18"
    good = tmp_path / "good"
    good.mkdir()
    (good / "run_meta.json").write_text(json.dumps({"env_vars": {"KRABBY_OMNI_STAND_OBS": "clock"}}))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "run_meta.json").write_text(json.dumps({"env_vars": {"KRABBY_OMNI_STAND_OBS": "1"}}))
    pt = {"name": "P2__Z1_s18", "stamps": {"suite@1": str(good), "suite@2": str(bad)}}
    reasons = read.z_clock_invalid_reasons(pt)
    assert len(reasons) == 1 and "suite@2" in reasons[0] and "clock" in reasons[0]
    pt_ok = {"name": "P2__Z1_s18", "stamps": {"suite@1": str(good)}}
    assert read.z_clock_invalid_reasons(pt_ok) == []


def _z_stamp(tmp_path, name, bits_by_ep, seg, labels=("stand", "fwd", "stand")):
    """A synthetic Z stamp: run_meta.hold_labels + raw episodes carrying segment_id and clock_obs (T, 2)."""
    import json

    import numpy as np

    stamp = tmp_path / name
    raw = stamp / "raw"
    raw.mkdir(parents=True)
    (stamp / "run_meta.json").write_text(json.dumps({"hold_labels": list(labels)}))
    for ep, bit in enumerate(bits_by_ep):
        clock = np.zeros((len(bit), 2))
        clock[:, 0] = bit
        np.savez(raw / f"episode_{ep:02d}.npz", clock_obs=clock, segment_id=np.asarray(seg))
    return str(stamp)


def test_z_raw_bit_reasons_lag_tolerance(read, tmp_path):
    """The raw bit is checked against the hold labels (the driver's narrowed clock check): exact and one-step-late
    bits pass; a bit wrong far from any boundary fails (fable review 2026-10-09: ground truth from the labels, so
    a plain Z policy eval IS checked)."""
    import numpy as np

    seg = np.array([-1] * 3 + [0] * 6 + [1] * 6 + [2] * 5)          # prologue, stand, fwd, stand
    label = (seg == 0) | (seg == 2)
    exact = label.astype(float)
    late = np.roll(label, 1).astype(float)
    late[:3] = 0.0
    good = _z_stamp(tmp_path, "good", [exact, late], seg)
    bad_bit = exact.copy()
    bad_bit[11] = 1.0                                                 # mid-fwd, two steps from any boundary
    bad = _z_stamp(tmp_path, "bad", [bad_bit], seg)
    fails, checked = read.z_raw_bit_reasons({"name": "P2__Z1_s18", "stamps": {"suite@1": good}})
    assert checked == 1 and fails == []
    fails, checked = read.z_raw_bit_reasons({"name": "P2__Z1_s18", "stamps": {"suite@1": bad}})
    assert checked == 1 and len(fails) == 1 and "raw stand bit vs labels" in fails[0]


def test_z_raw_bit_reasons_fails_a_stamp_without_the_raw_bit(read, tmp_path):
    """A stamp whose raws carry no clock_obs cannot show the bit: INVALID (fails), never silently skipped."""
    import json

    import numpy as np

    stamp = tmp_path / "noclock"
    (stamp / "raw").mkdir(parents=True)
    (stamp / "run_meta.json").write_text(json.dumps({"hold_labels": ["stand"]}))
    np.savez(stamp / "raw" / "episode_00.npz", obs=np.zeros((5, 20)), segment_id=np.zeros(5, dtype=int))
    fails, checked = read.z_raw_bit_reasons({"name": "P2__Z1_s18", "stamps": {"suite@1": str(stamp)}})
    assert checked == 1 and len(fails) == 1


def test_p2r_z_fidelity_field_names(read, monkeypatch):
    """p2r_z_fidelity reads steps_per_s / taps_per_s from the same p2s_stand field names p2s itself reads."""
    monkeypatch.setattr(read, "_rec", lambda ref: None)       # P2R_HOLD_s18 unrecorded on this machine -> missing
    z_pt = {"name": "P2__Z1_s18", "p2s": {"suite": {1: {"steps_per_s": 0.2, "taps_per_s": 1.0},
                                                    2: {"steps_per_s": 0.3, "taps_per_s": 1.2}},
                                         "walkstop": {1: {"pooled": {"steps_per_s": 0.1, "taps_per_s": 0.5}}}}}
    fid = read.p2r_z_fidelity(z_pt)
    assert fid["z"]["suite"][1]["steps_per_s"] == 0.2
    assert fid["z"]["walkstop"][1]["steps_per_s"] == 0.1
    assert fid["hold_missing"] is True


def test_p2s_self_test_passes_with_the_z_additions():
    r = subprocess.run([sys.executable, str(SCRIPTS / "p2_read.py"), "--p2s-self-test"], capture_output=True,
                       text=True, timeout=120,
                       env=dict(os.environ, PYTHONPATH=str(REPO / "parkour/parkour_tasks") + ":" + str(REPO / "parkour")))
    assert r.returncode == 0, r.stdout + r.stderr


# ============================================================================================= p2r_early.py
def test_p2r_early_self_test_passes():
    r = subprocess.run([sys.executable, str(SCRIPTS / "p2r_early.py"), "--self-test"], capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


def test_p2r_early_pure_judges_directly(early):
    assert early.judge_drift(0.5, 1.4)["rc"] == early.RC_OK
    assert early.judge_drift(0.8, 1.4)["rc"] == early.RC_NOTIFY
    F0 = 19995
    flat = [(F0 + i, 2.3) for i in range(1100)]
    assert early.judge_formation(flat, F0)["rc"] == early.RC_NOTIFY


# ============================================================================================= run_arm.sh: shell
def _bash(script: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=120)


def _env(tmp: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2_"))}
    camp = tmp / "camp"
    (camp / "logs").mkdir(parents=True, exist_ok=True)
    (tmp / "code").mkdir(exist_ok=True)
    env.update(P2_OUT=str(tmp / "out"), CAMPAIGN_DIR=str(camp), P2_CAP="", CODE_ROOT=str(tmp / "code"))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def test_run_arm_scripts_parse():
    assert subprocess.run(["bash", "-n", str(RUN_ARM)], capture_output=True).returncode == 0


def _graft_fixture(tmp_path: Path, env: dict, seed: int = 18, identity_cpu=True, identity_gpu_stamps=None):
    """A graft checkpoint + its sidecar json, and the S1 head it claims to be a graft of (with the eval-stamp
    fixture :func:`test_omni_p2b_queue._s1_head_fixture` already proves ``verify_head_sha`` resolves)."""
    out = Path(env["P2_OUT"])
    s1_head = tmp_path / f"s1_head_{seed}.pt"
    s1_head.write_bytes(f"the S1 p2a head, seed {seed}".encode())
    s1_sha = hashlib.sha256(s1_head.read_bytes()).hexdigest()
    arms = out / "arms"
    arms.mkdir(parents=True, exist_ok=True)
    (arms / f"P2__S1_s{seed}.record.json").write_text(json.dumps(
        {"ref": f"P2__S1_s{seed}", "chain": f"P2__S1_s{seed}", "seed": seed, "checkpoint": str(s1_head),
         "run_dir": str(tmp_path / f"run_s{seed}"), "driver_stamp": str(tmp_path / f"stamp_s{seed}")}))
    stamp = out / "lr" / f"P2__S1_s{seed}" / "field_hard__omni" / "seed001" / "2026-10-07_00-00-00"
    stamp.mkdir(parents=True)
    (stamp / "p2_complete.json").write_text("{}")
    (stamp / "run_meta.json").write_text(json.dumps({"checkpoint_sha256": s1_sha}))
    graft_ck = tmp_path / "graft" / f"model_19994_{seed}.pt"
    graft_ck.parent.mkdir(parents=True, exist_ok=True)
    graft_ck.write_bytes(f"the graft of s1_head_{seed}".encode())
    graft_sha = hashlib.sha256(graft_ck.read_bytes()).hexdigest()
    graft_ck.with_suffix(".json").write_text(json.dumps(
        {"source_path": str(s1_head), "source_sha256": s1_sha, "graft_sha256": graft_sha,
         "identity_cpu": identity_cpu,
         "identity_gpu_stamps": [] if identity_gpu_stamps is None else identity_gpu_stamps}))
    return graft_ck, s1_head


def test_verify_head_sha_accepts_a_graft_with_a_full_identity_record(tmp_path):
    env = _env(tmp_path, P2_RUN_ARM_LIB=1)
    graft_ck, _ = _graft_fixture(tmp_path, env, seed=18,
                                 identity_gpu_stamps=["GRAFT_S1_s18/suite__omni/seed001/x"])
    camp = Path(env["CAMPAIGN_DIR"])
    camp.mkdir(parents=True, exist_ok=True)
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__Z1_s18": {"stages": [{"resume_from": {"checkpoint": str(graft_ck)}}]}}}))
    (camp / "anchors.yaml").write_text(json.dumps({"anchors": {}}))
    r = _bash(f"source {RUN_ARM}\nverify_head_sha P2__Z1_s18; echo \"rc=$?\"", env)
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    assert "via graft" in r.stdout and "P2__S1_s18 p2a head" in r.stdout


def test_verify_head_sha_refuses_a_graft_with_no_gpu_stamp(tmp_path):
    env = _env(tmp_path, P2_RUN_ARM_LIB=1)
    graft_ck, _ = _graft_fixture(tmp_path, env, seed=18, identity_gpu_stamps=[])
    camp = Path(env["CAMPAIGN_DIR"])
    camp.mkdir(parents=True, exist_ok=True)
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__Z1_s18": {"stages": [{"resume_from": {"checkpoint": str(graft_ck)}}]}}}))
    (camp / "anchors.yaml").write_text(json.dumps({"anchors": {}}))
    r = _bash(f"source {RUN_ARM}\nverify_head_sha P2__Z1_s18; echo \"rc=$?\"", env)
    assert "rc=1" in r.stdout, r.stdout + r.stderr
    assert "identity_gpu_stamps is empty" in r.stdout


def test_verify_head_sha_refuses_a_graft_without_cpu_identity(tmp_path):
    env = _env(tmp_path, P2_RUN_ARM_LIB=1)
    graft_ck, _ = _graft_fixture(tmp_path, env, seed=18, identity_cpu=False,
                                 identity_gpu_stamps=["GRAFT_S1_s18/suite__omni/seed001/x"])
    camp = Path(env["CAMPAIGN_DIR"])
    camp.mkdir(parents=True, exist_ok=True)
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__Z1_s18": {"stages": [{"resume_from": {"checkpoint": str(graft_ck)}}]}}}))
    (camp / "anchors.yaml").write_text(json.dumps({"anchors": {}}))
    r = _bash(f"source {RUN_ARM}\nverify_head_sha P2__Z1_s18; echo \"rc=$?\"", env)
    assert "rc=1" in r.stdout, r.stdout + r.stderr
    assert "identity_cpu is False" in r.stdout


def test_verify_head_sha_refuses_a_swapped_graft_file(tmp_path):
    """The file on disk no longer matches graft.json's own graft_sha256 (swapped / re-built since)."""
    env = _env(tmp_path, P2_RUN_ARM_LIB=1)
    graft_ck, _ = _graft_fixture(tmp_path, env, seed=18,
                                 identity_gpu_stamps=["GRAFT_S1_s18/suite__omni/seed001/x"])
    graft_ck.write_bytes(b"a DIFFERENT file than the one graft.json was written for")
    camp = Path(env["CAMPAIGN_DIR"])
    camp.mkdir(parents=True, exist_ok=True)
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__Z1_s18": {"stages": [{"resume_from": {"checkpoint": str(graft_ck)}}]}}}))
    (camp / "anchors.yaml").write_text(json.dumps({"anchors": {}}))
    r = _bash(f"source {RUN_ARM}\nverify_head_sha P2__Z1_s18; echo \"rc=$?\"", env)
    assert "rc=1" in r.stdout, r.stdout + r.stderr
    assert "graft.json records graft_sha256" in r.stdout


def test_fork_source_rd_resolves_a_graft_to_its_source_run_dir(tmp_path):
    env = _env(tmp_path, P2_RUN_ARM_LIB=1)
    graft_ck, s1_head = _graft_fixture(tmp_path, env, seed=18,
                                       identity_gpu_stamps=["GRAFT_S1_s18/suite__omni/seed001/x"])
    camp = Path(env["CAMPAIGN_DIR"])
    camp.mkdir(parents=True, exist_ok=True)
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__Z1_s18": {"stages": [{"resume_from": {"checkpoint": str(graft_ck)}}]}},
         "references": {"P2__Z1_s18": {"chain": "P2__Z1_s18", "seed": 18}}}))
    r = _bash(f"source {RUN_ARM}\nfork_source_rd P2__Z1_s18", env)
    assert r.stdout.strip() == str(s1_head.parent), r.stdout + r.stderr


def test_fork_source_rd_leaves_a_plain_fork_unchanged(tmp_path):
    """A chain with no graft sidecar (e.g. a plain ST2/S1 fork) resolves to the checkpoint's own parent, exactly
    as before this change."""
    env = _env(tmp_path, P2_RUN_ARM_LIB=1)
    head = tmp_path / "run_dir" / "model_19994.pt"
    head.parent.mkdir(parents=True)
    head.write_bytes(b"a plain head, no graft sidecar")
    camp = Path(env["CAMPAIGN_DIR"])
    camp.mkdir(parents=True, exist_ok=True)
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__ST2_s18": {"stages": [{"resume_from": {"checkpoint": str(head)}}]}},
         "references": {"P2__ST2_s18": {"chain": "P2__ST2_s18", "seed": 18}}}))
    r = _bash(f"source {RUN_ARM}\nfork_source_rd P2__ST2_s18", env)
    assert r.stdout.strip() == str(head.parent), r.stdout + r.stderr


def test_watch_arm_guard_case_matches_z_chains():
    """The ``P2__ST[0-9]*_s*|P2__Z[0-9]*_s*`` case in watch_arm (grep, since watch_arm runs in the background and
    is not independently callable without a live unit)."""
    text = RUN_ARM.read_text()
    assert "P2__ST[0-9]*_s*|P2__Z[0-9]*_s*)" in text


# ============================================================================================= candidates.yaml v12
def test_candidates_yaml_has_the_z1_chain_with_the_s1_overlay_verbatim():
    cands_path = CAMP / "candidates.yaml"
    d = yaml.safe_load(open(cands_path))
    assert "P2__Z1_s18" in d["chains"]
    assert "P2__Z1_s18" in d["references"]
    stage = d["chains"]["P2__Z1_s18"]["stages"][0]
    assert stage["stage"] == "p2r"
    assert stage["resume_from"]["checkpoint"] == "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt"
    new_knobs = {"KRABBY_OMNI_STAND_OBS": "clock", "KRABBY_ANCHOR_W": "1.0", "KRABBY_STAND_PRIOR_W": "1.0",
                "KRABBY_ANCHOR_CKPT": "parkour/outputs/omni_plan/phase2/graft/S1_s18_c15/model_19994.pt"}
    ov = dict(stage["overlay"])
    for k, v in new_knobs.items():
        assert ov.pop(k) == v
    s1_ov = d["chains"]["P2__S1_s18"]["stages"][0]["overlay"]
    assert ov == s1_ov, "the S1 overlay was not carried verbatim (ST2's stand terms, or something else, leaked in)"


def test_candidates_v12_copy_is_identical_to_the_campaign_file():
    # NOTE(p2r Z2, 2026-10-10): the campaign file moved to v13 (P2__Z2_s18 added), then v14 (NOTE(p2r Z2 cams-only,
    # 2026-10-10): P2__Z2_s18 drops the held proxy); its copy is now scripts/candidates_v14.yaml, and v12's Z1 chain /
    # reference must survive into it unchanged.
    cands = (CAMP / "candidates.yaml").read_bytes()
    v14 = (SCRIPTS / "candidates_v19.yaml").read_bytes()  # v17 = v16 + the Z2d chain (2026-10-11)
    assert cands == v14
    v12 = yaml.safe_load(open(SCRIPTS / "candidates_v12.yaml"))
    cur = yaml.safe_load(open(CAMP / "candidates.yaml"))
    assert cur["chains"]["P2__Z1_s18"] == v12["chains"]["P2__Z1_s18"]
    assert cur["references"]["P2__Z1_s18"] == v12["references"]["P2__Z1_s18"]
