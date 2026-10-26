"""Stage p2b (P2B_PLAN.md / CHARTER 10), the files owned by this agent: CPU only, nothing here starts Isaac/Kit or a
GPU.

* ``scripts/run_arm.sh``: ``verify_head_sha`` accepting a fork source that is an S1 p2a head (in addition to its
  unchanged anchors.yaml path), and ``--row-eval`` taking env_seeds 3 / 4 (not hard-limited to 1 / 2);
* ``scripts/p2_live.py``: the CL / TG term-liveness and preflight judges, ``arm_of`` / ``seed_of`` on the p2b arm
  names, and ``p2b_checkpoints`` (the self-test in the module itself carries the bulk of the pure-judge coverage;
  this file re-checks the handful most load-bearing for the shell integration below);
* NEW ``scripts/p2b_queue.sh``: step order (refs -> inertness -> screen -> STOP -> reps -> recipe -> STOP),
  ``FROM_STEP`` resume, the lever whitelist (``P2B_LEVERS``) and the shared flock with ``p2_queue.sh``.

The shell scripts are exercised through stubs: ``P2_PY`` / ``P2_OUT`` / ``CAMPAIGN_DIR`` in tmp, ``RUN_ARM`` /
``READ`` stubs for the queue, ``P2_CAP=""`` (no systemd scope) and ``P2_RUN_ARM_LIB=1`` to source run_arm.sh's
functions without running its command line.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
OMNI = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task"
CAMP = OMNI / "experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
RUN_ARM = SCRIPTS / "run_arm.sh"
P2B_QUEUE = SCRIPTS / "p2b_queue.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def live():
    return _load(SCRIPTS / "p2_live.py", "_t_p2b_live")


# ===================================================================================== p2_live.py: pure judges
def test_arm_of_and_seed_of_the_p2b_arm_names(live):
    assert [live.arm_of(c) for c in ("P2__C0_s18", "P2__CL_s16", "P2__TG_s17")] == ["C0", "CL", "TG"]
    assert [live.seed_of(c) for c in ("P2__CL_s16", "P2__TG_s17", "P2__C0_s18")] == [16, 17, 18]
    assert live.seed_of("no-seed-here") is None


def test_p2b_checkpoints_is_the_fork_first_plus_the_round_offsets(live):
    assert live.p2b_checkpoints(19994) == [21000, 22500, 24000]
    assert live.P2B_MCK_OFFSETS == (1005, 2505, 4005)


def test_cl_liveness_needs_both_terms_sign_aware_and_within_3x(live):
    base = {"Metrics/clock_hidden_frac": 1.0, "Metrics/base_parkour/reach_obst_frac": 0.3}
    first = 14996

    def ser(**over):
        d = dict(base, **over)
        return {t: [(first + i, v) for i in range(215)] for t, v in d.items()}
    facts = {"have_env": True, "curriculum_null": True,
             "weights": {"reward_clock_schedule": 0.001, "reward_clock_swing_apex": 0.001, "reward_action_rate": 0.0,
                         "reward_collision": -2.0, "reward_feet_edge": -0.3, "reward_feet_stumble": -1.0,
                         "reward_feet_air_time_positive": 0.8, "reward_stride_length": 0.5},
             "term_funcs": {}, "dr_events": {e: True for e in live.DR_EVENTS}, "push_x": [-0.5, 0.5],
             "terrain_fns": ["omni_field_terrain"], "num_rows": 10, "dropped_ppo_keys": []}
    ov = {"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_ACTION_RATE_W": "0"}
    audit_ok = [("overlay x", True)]
    replay = {"p2b": {"CL": {"terms": {"reward_foot_clearance": {"episode_reward": 0.02},
                                       "penalty_swing_min_clearance": {"episode_reward": -0.01}}}}}
    good = ser(**{"Episode_Reward/reward_foot_clearance": 0.021, "Episode_Reward/penalty_swing_min_clearance": -0.011})
    r = live.judge_wiring("CL", ov, facts, good, first, audit_ok, replay)
    assert r["rc"] == live.RC_PASS, r["fails"]
    dead = ser(**{"Episode_Reward/reward_foot_clearance": 0.0, "Episode_Reward/penalty_swing_min_clearance": -0.011})
    assert live.judge_wiring("CL", ov, facts, dead, first, audit_ok, replay)["rc"] == live.RC_FAIL
    wrong_sign = ser(**{"Episode_Reward/reward_foot_clearance": -0.02, "Episode_Reward/penalty_swing_min_clearance": -0.011})
    assert live.judge_wiring("CL", ov, facts, wrong_sign, first, audit_ok, replay)["rc"] == live.RC_FAIL
    assert live.judge_wiring("CL", ov, facts, good, first, audit_ok, None)["rc"] == live.RC_FAIL  # no replay: fails closed


def test_tg_liveness_needs_the_seed_tail_floor(live):
    base = {"Metrics/clock_hidden_frac": 1.0, "Metrics/base_parkour/reach_obst_frac": 0.3}
    first = 14996

    def ser(**over):
        d = dict(base, **over)
        return {t: [(first + i, v) for i in range(215)] for t, v in d.items()}
    facts = {"have_env": True, "curriculum_null": True,
             "weights": {"reward_clock_schedule": 0.001, "reward_clock_swing_apex": 0.001, "reward_action_rate": 0.0,
                         "reward_collision": -2.0, "reward_feet_edge": -0.3, "reward_feet_stumble": -1.0,
                         "reward_feet_air_time_positive": 0.8, "reward_stride_length": 0.5},
             "term_funcs": {}, "dr_events": {e: True for e in live.DR_EVENTS}, "push_x": [-0.5, 0.5],
             "terrain_fns": ["omni_field_terrain"], "num_rows": 10, "dropped_ppo_keys": []}
    ov = {"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_ACTION_RATE_W": "0"}
    audit_ok = [("overlay x", True)]
    replay = {"p2b": {"TG": {"terms": {"reward_orientation": {"episode_reward": -0.05}}}}}
    s = ser(**{"Episode_Reward/reward_orientation": -0.045})
    assert live.judge_wiring("TG", ov, facts, s, first, audit_ok, replay,
                             ref_tail={"reward_orientation": -0.02})["rc"] == live.RC_PASS
    assert live.judge_wiring("TG", ov, facts, s, first, audit_ok, replay,
                             ref_tail={"reward_orientation": -0.03})["rc"] == live.RC_FAIL      # under the 2x floor
    assert live.judge_wiring("TG", ov, facts, s, first, audit_ok, replay, ref_tail=None)["rc"] == live.RC_FAIL


def test_c0_has_no_arm_specific_term(live):
    first = 14996
    ser = {"Metrics/clock_hidden_frac": [(first + i, 1.0) for i in range(215)],
           "Metrics/base_parkour/reach_obst_frac": [(first + i, 0.3) for i in range(215)]}
    facts = {"have_env": True, "curriculum_null": True,
             "weights": {"reward_clock_schedule": 0.001, "reward_clock_swing_apex": 0.001, "reward_action_rate": 0.0,
                         "reward_collision": -2.0, "reward_feet_edge": -0.3, "reward_feet_stumble": -1.0,
                         "reward_feet_air_time_positive": 0.8, "reward_stride_length": 0.5},
             "term_funcs": {}, "dr_events": {e: True for e in live.DR_EVENTS}, "push_x": [-0.5, 0.5],
             "terrain_fns": ["omni_field_terrain"], "num_rows": 10, "dropped_ppo_keys": []}
    ov = {"KRABBY_FLAT_TERRAIN_GEOM": "field", "KRABBY_ACTION_RATE_W": "0"}
    r = live.judge_wiring("C0", ov, facts, ser, first, [("overlay x", True)], None)
    assert r["rc"] == live.RC_PASS and "no arm-specific term" in r["notes"][0]


def test_preflight_cl_tg_c0(live):
    em_ok = {"terrain_levels_obst": ["x.py"], "promote_count": ["x.py"]}
    assert live.judge_preflight("C0", em_ok, None, None)["rc"] == live.RC_PASS
    assert live.judge_preflight("CL", em_ok, None, None)["rc"] == live.RC_FAIL
    cl_replay = {"p2b": {"CL": {"verdict": "PASS", "terms": {
        "reward_foot_clearance": {"episode_reward": 0.02}, "penalty_swing_min_clearance": {"episode_reward": -0.01}}}}}
    assert live.judge_preflight("CL", em_ok, cl_replay, None)["rc"] == live.RC_PASS
    fail_no_overrule = {"p2b": {"CL": dict(cl_replay["p2b"]["CL"], verdict="FAIL")}}
    assert live.judge_preflight("CL", em_ok, fail_no_overrule, None)["rc"] == live.RC_FAIL
    fail_overrule = {"p2b": {"CL": dict(fail_no_overrule["p2b"]["CL"], overrule={"by": "user"})}}
    assert live.judge_preflight("CL", em_ok, fail_overrule, None)["rc"] == live.RC_PASS
    tg_replay = {"p2b": {"TG": {"verdict": "PASS", "terms": {"reward_orientation": {"episode_reward": -0.05}}}}}
    assert live.judge_preflight("TG", em_ok, tg_replay, None)["rc"] == live.RC_PASS
    assert live.judge_preflight("TG", em_ok, None, None)["rc"] == live.RC_FAIL


def test_p2_live_self_test_passes():
    r = subprocess.run([sys.executable, str(SCRIPTS / "p2_live.py"), "--self-test"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_p2b_checkpoints_cli():
    r = subprocess.run([sys.executable, str(SCRIPTS / "p2_live.py"), "--p2b-checkpoints", "19994"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "21000 22500 24000", r.stdout + r.stderr


# ===================================================================================== run_arm.sh: shell plumbing
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


def test_scripts_parse():
    for f in (RUN_ARM, P2B_QUEUE):
        assert subprocess.run(["bash", "-n", str(f)], capture_output=True).returncode == 0, f


def test_verify_head_sha_keeps_the_anchors_yaml_path(tmp_path):
    """verify_head_sha's anchors.yaml path is unchanged: a fork of an anchor head still checks against its table."""
    head = tmp_path / "anchor_model.pt"
    head.write_bytes(b"an anchor head")
    sha = hashlib.sha256(head.read_bytes()).hexdigest()
    env = _env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1)
    camp = Path(env["CAMPAIGN_DIR"])
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__A1_s18": {"stages": [{"resume_from": {"checkpoint": str(head)}}]}}}))
    (camp / "anchors.yaml").write_text(json.dumps({"anchors": {18: {"checkpoint": str(head), "sha256": sha}}}))
    script = f"""
        source {RUN_ARM}
        verify_head_sha P2__A1_s18; echo "rc=$?"
    """
    r = _bash(script, env)
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    assert "(anchors.yaml)" in r.stdout


def test_verify_head_sha_fails_an_anchor_mismatch(tmp_path):
    head = tmp_path / "anchor_model.pt"
    head.write_bytes(b"an anchor head")
    env = _env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1)
    camp = Path(env["CAMPAIGN_DIR"])
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__A1_s18": {"stages": [{"resume_from": {"checkpoint": str(head)}}]}}}))
    (camp / "anchors.yaml").write_text(json.dumps({"anchors": {18: {"checkpoint": str(head), "sha256": "0" * 64}}}))
    script = f"source {RUN_ARM}\nverify_head_sha P2__A1_s18; echo \"rc=$?\""
    r = _bash(script, env)
    assert "rc=1" in r.stdout, r.stdout + r.stderr


def _s1_head_fixture(tmp_path: Path, env: dict, seed: int = 16) -> tuple[Path, str]:
    """An S1 p2a head file + its arms record + a complete field_hard__omni p2a eval stamp recording its sha256."""
    head = tmp_path / f"s1_head_{seed}.pt"
    head.write_bytes(f"the S1 p2a head, seed {seed}".encode())
    sha = hashlib.sha256(head.read_bytes()).hexdigest()
    out = Path(env["P2_OUT"])
    arms = out / "arms"
    arms.mkdir(parents=True, exist_ok=True)
    (arms / f"P2__S1_s{seed}.record.json").write_text(json.dumps(
        {"ref": f"P2__S1_s{seed}", "chain": f"P2__S1_s{seed}", "seed": seed, "checkpoint": str(head),
         "run_dir": str(tmp_path / f"run_s{seed}"), "driver_stamp": str(tmp_path / f"stamp_s{seed}")}))
    stamp = out / "lr" / f"P2__S1_s{seed}" / "field_hard__omni" / "seed001" / "2026-10-07_00-00-00"
    stamp.mkdir(parents=True)
    (stamp / "p2_complete.json").write_text("{}")
    (stamp / "run_meta.json").write_text(json.dumps({"checkpoint_sha256": sha}))
    return head, sha


def test_verify_head_sha_accepts_an_s1_p2a_head_fork(tmp_path):
    """P2B_PLAN.md step 1 / CHARTER 10: a p2b chain's {checkpoint} fork of an S1 p2a head verifies against that
    head's newest complete field_hard__omni eval stamp's run_meta.checkpoint_sha256 -- not against anchors.yaml."""
    env = _env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1)
    head, sha = _s1_head_fixture(tmp_path, env, seed=16)
    camp = Path(env["CAMPAIGN_DIR"])
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__CL_s16": {"stages": [{"resume_from": {"checkpoint": str(head)}}]}}}))
    (camp / "anchors.yaml").write_text(json.dumps({"anchors": {}}))
    script = f"source {RUN_ARM}\nverify_head_sha P2__CL_s16; echo \"rc=$?\""
    r = _bash(script, env)
    assert "rc=0" in r.stdout, r.stdout + r.stderr
    assert "P2__S1_s16 p2a head" in r.stdout


def test_verify_head_sha_refuses_a_stale_s1_fork(tmp_path):
    """The checkpoint file on disk no longer matches the recorded eval stamp's sha256: refused, not accepted."""
    env = _env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1)
    head, sha = _s1_head_fixture(tmp_path, env, seed=17)
    head.write_bytes(b"a DIFFERENT head (e.g. the run was re-forked)")
    camp = Path(env["CAMPAIGN_DIR"])
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__TG_s17": {"stages": [{"resume_from": {"checkpoint": str(head)}}]}}}))
    (camp / "anchors.yaml").write_text(json.dumps({"anchors": {}}))
    script = f"source {RUN_ARM}\nverify_head_sha P2__TG_s17; echo \"rc=$?\""
    r = _bash(script, env)
    assert "rc=1" in r.stdout, r.stdout + r.stderr


def test_verify_head_sha_refuses_an_unrecorded_checkpoint(tmp_path):
    """A fork source that is neither an anchors.yaml head nor a recorded P2__S1_s<seed> checkpoint refuses."""
    env = _env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1)
    (Path(env["P2_OUT"]) / "arms").mkdir(parents=True, exist_ok=True)
    stray = tmp_path / "stray.pt"
    stray.write_bytes(b"not a recorded head")
    camp = Path(env["CAMPAIGN_DIR"])
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__CL_s16": {"stages": [{"resume_from": {"checkpoint": str(stray)}}]}}}))
    (camp / "anchors.yaml").write_text(json.dumps({"anchors": {}}))
    script = f"source {RUN_ARM}\nverify_head_sha P2__CL_s16; echo \"rc=$?\""
    r = _bash(script, env)
    assert "rc=1" in r.stdout, r.stdout + r.stderr
    assert "neither an anchors.yaml head nor a recorded" in r.stdout


def test_fork_source_rd_resolves_an_s1_head_checkpoint(tmp_path):
    """fork_source_rd (unchanged) already resolves any {checkpoint} fork generically, incl. a p2b fork of an S1
    p2a head: it only needs resume_from.checkpoint, never anchors.yaml."""
    env = _env(tmp_path, P2_RUN_ARM_LIB=1)
    head, _ = _s1_head_fixture(tmp_path, env, seed=18)
    camp = Path(env["CAMPAIGN_DIR"])
    (camp / "candidates.yaml").write_text(json.dumps(
        {"chains": {"P2__CL_s18": {"stages": [{"resume_from": {"checkpoint": str(head)}}]}},
         "references": {"P2__CL_s18": {"chain": "P2__CL_s18", "seed": 18}}}))
    script = f"source {RUN_ARM}\nfork_source_rd P2__CL_s18"
    r = _bash(script, env)
    assert r.stdout.strip() == str(head.parent), r.stdout + r.stderr


def test_row_eval_accepts_env_seed_3_and_4_not_just_1_and_2(tmp_path):
    """--row-eval is not hard-limited to env_seed 1/2 (P2B_PLAN.md step 3 needs field_hard at env_seeds 1-4)."""
    code = tmp_path / "code"
    ev = code / "parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/scripts/eval_crab_hex_gait.py"
    ev.parent.mkdir(parents=True)
    ev.write_text(textwrap.dedent("""\
        import json, sys, pathlib
        a = sys.argv
        out = pathlib.Path(a[a.index('--output-root') + 1]) / a[a.index('--scenario') + 1] / \
            ('seed%03d' % int(a[a.index('--env-seed') + 1])) / 'ts'
        out.mkdir(parents=True)
        (out / 'complete.json').write_text('{}')
        (out / 'argv.json').write_text(json.dumps(a))
        """))
    (code / "parkour").mkdir(exist_ok=True)
    env = _env(tmp_path, P2_PY=sys.executable, P2_RUN_ARM_LIB=1)
    st = tmp_path / "stamp"
    st.mkdir()
    (st / "run_meta.json").write_text(json.dumps({"env_vars": {}}))
    script = f"""
        source {RUN_ARM}
        code_guard() {{ return 0; }}
        drivers_idle() {{ :; }}
        write_sidecar() {{ echo '{{}}' > "$1/code_state.json"; }}
        row_eval N3 {tmp_path}/m.pt {st} {tmp_path}/man.yaml field_hard__omni 3; echo "rc3=$? dir3=$ROW_DIR"
        row_eval N4 {tmp_path}/m.pt {st} {tmp_path}/man.yaml field_hard__omni 4; echo "rc4=$? dir4=$ROW_DIR"
    """
    r = _bash(script, env)
    assert "rc3=0" in r.stdout and "rc4=0" in r.stdout, r.stdout + r.stderr
    assert str(tmp_path / "out/lr/N3/field_hard__omni/seed003") in r.stdout
    assert str(tmp_path / "out/lr/N4/field_hard__omni/seed004") in r.stdout


# ===================================================================================== p2b_queue.sh
def _read_stub(tmp: Path, rcs: dict) -> Path:
    """A fake p2_read.py: logs every call's argv, prints a stub json line, exits the configured rc per flag (0
    for any flag not named in ``rcs``)."""
    p = tmp / "read_stub.py"
    lines = ["import sys", f"open({str(tmp / 'read.calls')!r}, 'a').write(' '.join(sys.argv[1:]) + chr(10))",
             "a = sys.argv[1:]", "print('{\"stub\": true}')"]
    for flag, rc in rcs.items():
        lines.append(f"if {flag!r} in a: sys.exit({rc})")
    lines.append("sys.exit(0)")
    p.write_text("\n".join(lines) + "\n")
    return p


def _run_arm_stub(tmp: Path) -> Path:
    """A fake run_arm.sh: logs argv; a --row-eval / --lr-eval call exits $ROWRC; any other (plain-ref training)
    call exits $ARMRC and, on success, writes a dummy arms record (checkpoint + driver_stamp) so the queue's
    post-training reads (recf) have something to read."""
    p = tmp / "run_arm_stub.sh"
    p.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        echo "$@" >> {tmp}/run_arm.calls
        case "$1" in
          --row-eval|--lr-eval) exit "${{ROWRC:-0}}" ;;
        esac
        rc=${{ARMRC:-5}}
        if [ "$rc" = 0 ]; then
          mkdir -p {tmp}/out/arms
          printf '{{"checkpoint": "%s/dummy_model.pt", "driver_stamp": "%s/dummy_stamp"}}' {tmp} {tmp} \\
            > "{tmp}/out/arms/$1.record.json"
        fi
        exit "$rc"
        """))
    p.chmod(0o755)
    return p


def _p2b_env(tmp: Path, rcs: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2_", "P2B_", "ROWRC", "ARMRC"))}
    env.update(P2_OUT=str(tmp / "out"), P2_PY=sys.executable, READ=str(_read_stub(tmp, rcs or {})),
               RUN_ARM=str(_run_arm_stub(tmp)), LIVE=str(SCRIPTS / "p2_live.py"))
    return env


def _seed_s1_records(tmp: Path) -> None:
    arms = tmp / "out" / "arms"
    arms.mkdir(parents=True, exist_ok=True)
    for seed in (16, 17, 18):
        (arms / f"P2__S1_s{seed}.record.json").write_text(json.dumps(
            {"checkpoint": str(tmp / f"s1_s{seed}.pt"), "driver_stamp": str(tmp / f"s1_stamp_{seed}")}))


def _run(env: dict, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(P2B_QUEUE)], env=env, capture_output=True, text=True, timeout=timeout)


def test_refs_then_inertness_stop_on_a_failed_inertness_check(tmp_path):
    """refs runs all three seeds (18 first, then 16, 17) before the inertness read; a non-zero --p2b-inertness
    stops the queue without ever reaching the screen."""
    env = _p2b_env(tmp_path, {"--p2b-inertness": 4})
    env["ROWRC"] = "0"
    _seed_s1_records(tmp_path)
    r = _run(env)
    assert r.returncode == 4, r.stdout + r.stderr
    calls = (tmp_path / "run_arm.calls").read_text()
    assert "REF_S1_s18" in calls and "REF_S1_s16" in calls and "REF_S1_s18" in calls.split("REF_S1_s16")[0], calls
    assert "field_hard__omni 4" in calls                                    # field_hard widened to env_seeds 1-4
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "P2B INERTNESS rc 4" in log
    assert "STOP p2b_queue: the REF code-inertness check" in log and "NOTIFY" in log
    assert "P2B SCREEN" not in log                                          # never reached the screen


def test_inertness_pass_starts_the_screen_with_c0_first(tmp_path):
    """A passed inertness check reaches the screen; the screen trains C0 before CL/TG, and a training failure
    there stops the queue immediately (no reads, no CL/TG)."""
    env = _p2b_env(tmp_path, {"--p2b-inertness": 0})
    env["ROWRC"] = "0"
    env["ARMRC"] = "5"                                                      # the first real arm (training) call fails
    _seed_s1_records(tmp_path)
    r = _run(env)
    assert r.returncode == 4, r.stdout + r.stderr
    calls = (tmp_path / "run_arm.calls").read_text().splitlines()
    plain = [c for c in calls if not c.startswith(("--row-eval", "--lr-eval"))]
    assert plain == ["P2__C0_s18"]                                          # CL/TG never launched
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "STOP p2b_queue: rc 5 in the arm P2__C0_s18 step" in log


def test_screen_trains_c0_cl_tg_in_order_then_always_stops(tmp_path):
    """With every call succeeding, the screen trains all three arms, reads field_hard at env_seeds 1-4 and the
    p2b checkpoints, then STOPS unconditionally (the user may prune levers) regardless of --p2b-screen's rc."""
    env = _p2b_env(tmp_path, {"--p2b-inertness": 0, "--p2b-screen": 1})      # a non-zero screen rc still stops once
    env["ROWRC"] = "0"
    env["ARMRC"] = "0"
    _seed_s1_records(tmp_path)
    r = _run(env)
    assert r.returncode == 4, r.stdout + r.stderr
    calls = (tmp_path / "run_arm.calls").read_text().splitlines()
    plain = [c for c in calls if not c.startswith(("--row-eval", "--lr-eval"))]
    assert plain == ["P2__C0_s18", "P2__CL_s18", "P2__TG_s18"]
    assert "field_hard__omni 4" in "\n".join(calls)
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "P2B SCREEN rc 1" in log
    assert "STOP p2b_queue: s18 screen done (C0, CL, TG)" in log and "NOTIFY" in log
    assert "p2b_queue replicates" not in log                                # reps never started


def test_from_step_reps_skips_refs_inertness_and_screen(tmp_path):
    """FROM_STEP=reps runs no refs/inertness/screen calls at all: the replicates start cold."""
    env = _p2b_env(tmp_path, {})
    env["ROWRC"] = "0"
    env["ARMRC"] = "5"
    env["FROM_STEP"] = "reps"
    r = _run(env)
    assert r.returncode == 4, r.stdout + r.stderr
    calls = (tmp_path / "run_arm.calls").read_text().splitlines()
    assert calls == ["P2__CL_s16"]                                          # default P2B_LEVERS="CL TG", s16 first
    assert not (tmp_path / "read.calls").exists()                           # no inertness/screen reads either


def test_from_step_reps_only_runs_levers_in_p2b_levers(tmp_path):
    """A pruned P2B_LEVERS (e.g. only TG survived the screen) means CL is never launched at either seed."""
    env = _p2b_env(tmp_path, {})
    env["ROWRC"] = "0"
    env["ARMRC"] = "5"
    env["FROM_STEP"] = "reps"
    env["P2B_LEVERS"] = "TG"
    r = _run(env)
    assert r.returncode == 4, r.stdout + r.stderr
    calls = (tmp_path / "run_arm.calls").read_text().splitlines()
    assert calls == ["P2__TG_s16"]
    assert "CL" not in " ".join(calls)


def test_from_step_reps_runs_s16_then_s17_for_every_lever(tmp_path):
    """With training (and every read) succeeding, both levers run at s16, then both at s17."""
    env = _p2b_env(tmp_path, {})
    env["ROWRC"] = "0"
    env["ARMRC"] = "0"
    env["FROM_STEP"] = "reps"
    r = _run(env)
    assert r.returncode == 4, r.stdout + r.stderr
    calls = (tmp_path / "run_arm.calls").read_text().splitlines()
    plain = [c for c in calls if not c.startswith(("--row-eval", "--lr-eval"))]
    assert plain == ["P2__CL_s16", "P2__TG_s16", "P2__CL_s17", "P2__TG_s17"]
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "p2b_queue replicates: levers [CL TG] at s16 then s17" in log
    assert "P2B RECIPE" in log and "STOP p2b_queue: replicates done" in log and "NOTIFY" in log


def test_from_step_recipe_runs_only_the_recipe_read(tmp_path):
    env = _p2b_env(tmp_path, {"--p2b-recipe": 0})
    env["FROM_STEP"] = "recipe"
    r = _run(env)
    assert r.returncode == 4, r.stdout + r.stderr
    assert not (tmp_path / "run_arm.calls").exists()
    assert (tmp_path / "read.calls").read_text().strip() == "--p2b-recipe"
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "STOP p2b_queue: replicates done" in log and "NOTIFY" in log


def test_queue_refuses_a_second_instance(tmp_path):
    import fcntl
    env = _p2b_env(tmp_path, {})
    env["FROM_STEP"] = "recipe"
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "out" / "p2_queue.lock", "w") as fh:        # the SAME lock file p2_queue.sh uses
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = _run(env)
    assert r.returncode == 3 and "already running" in r.stderr
    assert not (tmp_path / "run_arm.calls").exists()
    assert "another p2_queue holds" in (tmp_path / "out" / "queue.log").read_text()


def test_bad_from_step_refuses(tmp_path):
    env = _p2b_env(tmp_path, {})
    env["FROM_STEP"] = "nonsense"
    r = _run(env)
    assert r.returncode == 2 and "bad FROM_STEP" in r.stderr


def test_p2b_liveness_reads_the_calibrated_expectation_when_present():
    """CHARTER 10 (2026-10-08): a term with episode_reward_live_expect is read at that value, else at episode_reward."""
    live = _load(SCRIPTS / "p2_live.py", "_t_p2_live_calib")
    rg = {"p2b": {"TG": {"terms": {"reward_orientation": {"episode_reward": -0.0112,
                                                          "episode_reward_live_expect": -0.0241}}},
                  "CL": {"terms": {"reward_foot_clearance": {"episode_reward": 0.0267}}}}}
    assert live.p2b_replay_magnitude(rg, "TG", "reward_orientation") == -0.0241
    assert live.p2b_replay_magnitude(rg, "CL", "reward_foot_clearance") == 0.0267
    assert live.p2b_replay_magnitude(rg, "CL", "penalty_swing_min_clearance") is None
