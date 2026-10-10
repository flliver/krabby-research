"""Stage p2s (P2S_PLAN.md steps 5-7 / CHARTER 11): ``scripts/p2s_queue.sh`` and ``candidates.yaml`` v10. CPU only:
nothing here starts Isaac/Kit or a GPU.

* the queue, through stubs (``RUN_ARM`` / ``READ`` / ``P2_OUT`` in tmp): step order refs (REFS_S1_s18 then C0R_s18,
  every p2b row + suite es2 + walkstop at 1 / 2) -> inertness -> st2 -> st1 ONLY if ST2 reads STILL -> screen STOP (either
  way) -> refsrep (REFS_S1_s16 / _s17 + their inertness) -> rep16 -> rep17 -> recipe STOP; FROM_STEP resume; the shared
  flock with p2_queue.sh / p2b_queue.sh; the guard re-pointed to REFS_S1_s18; --accept-code-change passed ONCE; a
  killed (or already trained) reference refused; the notify-only p2s_flag stopping after its arm;
* ``run_arm.sh``'s p2s plumbing (the --at p2s watcher block, the rc-5 notify-only flag, the preflight's --arms);
* candidates v10: the six p2s chains -- stage p2s, base 2a, 5000 its, the p2b forks of each seed's S1 p2a head, the S1
  overlay VERBATIM + the two lever knobs at the planning doses -- and its scripts/ copy.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CAMP = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-07_omni_phase2"
SCRIPTS = CAMP / "scripts"
RUN_ARM = SCRIPTS / "run_arm.sh"
P2S_QUEUE = SCRIPTS / "p2s_queue.sh"
P2S_CHAINS = ("P2__ST2_s18", "P2__ST1_s18", "P2__ST2_s16", "P2__ST2_s17", "P2__ST1_s16", "P2__ST1_s17")
DOSES = {"ST2": ("1.1", "-1.25"), "ST1": ("0.33", "-0.50")}  # v11: the replay gate's solved doses


# ===================================================================================== candidates v10
@pytest.fixture(scope="module")
def cands():
    return yaml.safe_load((CAMP / "candidates.yaml").read_text())


def test_v10_copy_is_the_live_file():
    # v12 (CHARTER 12 / P2R_STATUE_PLAN.md step 1) added P2__Z1_s18 on top of v11's six p2s chains; v13 (P2R_Z2_PLAN.md
    # step 1, 2026-10-10) added P2__Z2_s18; v14 (the cams-only amendment, 2026-10-10: P2__Z2_s18 without the held
    # proxy) is now the newest snapshot, so it is the one that must equal the live file.
    assert (SCRIPTS / "candidates_v15.yaml").read_bytes() == (CAMP / "candidates.yaml").read_bytes()  # v15 (final R constants)


@pytest.mark.parametrize("chain", P2S_CHAINS)
def test_p2s_chain_shape(cands, chain):
    dose, seed = chain.split("__")[1].split("_s")
    st = cands["chains"][chain]["stages"]
    assert len(st) == 1 and st[0]["stage"] == "p2s" and st[0]["base"] == "2a" and st[0]["iters"] == 5000
    assert st[0]["from_scratch"] is False and st[0]["eval"] == ["suite"]
    assert cands["references"][chain] == {"chain": chain, "seed": int(seed)}
    # the {checkpoint} fork is EXACTLY the p2b chains' fork of that seed's S1 p2a head (model_19994)
    p2b = "P2__C0_s18" if seed == "18" else f"P2__CL_s{seed}"
    assert st[0]["resume_from"] == cands["chains"][p2b]["stages"][0]["resume_from"]
    assert st[0]["resume_from"]["checkpoint"].endswith("/model_19994.pt")
    # the S1 overlay VERBATIM (same keys, values and order), then the two lever knobs at the planning dose
    ov = dict(st[0]["overlay"])
    w = (ov.pop("KRABBY_STAND_SUPPORT_W"), ov.pop("KRABBY_STAND_SLIDE_W"))
    assert list(ov.items()) == list(cands["chains"][f"P2__S1_s{seed}"]["stages"][0]["overlay"].items()) or \
        list(ov.items()) == list(cands["chains"]["P2__S1_s18"]["stages"][0]["overlay"].items())
    assert w == DOSES[dose]
    assert list(st[0]["overlay"])[-2:] == ["KRABBY_STAND_SUPPORT_W", "KRABBY_STAND_SLIDE_W"]   # one place, at the end


def test_p2s_overlay_equals_the_p2b_continuation_plus_the_knobs(cands):
    """C0 (p2b) is the S1 recipe unchanged; every p2s chain is C0's overlay + exactly the two lever knobs."""
    c0 = cands["chains"]["P2__C0_s18"]["stages"][0]["overlay"]
    for chain in P2S_CHAINS:
        ov = cands["chains"][chain]["stages"][0]["overlay"]
        assert {k: v for k, v in ov.items() if k not in ("KRABBY_STAND_SUPPORT_W", "KRABBY_STAND_SLIDE_W")} == c0


def test_earlier_chains_are_untouched(cands):
    # v12 (CHARTER 12 / P2R_STATUE_PLAN.md step 1) adds exactly one new chain, P2__Z1_s18, on top of v11's six
    # p2s chains -- excluded here the same way the p2s chains themselves are, since v9 predates both.
    excl = P2S_CHAINS + ("P2__Z1_s18", "P2__Z2_s18")      # + v13's Z2 chain (P2R_Z2_PLAN.md step 1; v14 amends it)
    v9 = yaml.safe_load((SCRIPTS / "candidates_v9.yaml").read_text())
    assert {k: v for k, v in cands["chains"].items() if k not in excl} == v9["chains"]
    assert {k: v for k, v in cands["references"].items() if k not in excl} == v9["references"]


# ===================================================================================== the queue (stubs)
def _read_stub(tmp: Path, rcs: dict) -> Path:
    """A fake p2_read.py: logs every call's argv, prints a json line, exits the configured rc per flag (0 otherwise)."""
    p = tmp / "read_stub.py"
    lines = ["import sys", f"open({str(tmp / 'read.calls')!r}, 'a').write(' '.join(sys.argv[1:]) + chr(10))",
             "import os, json", "a = sys.argv[1:]",
             "print(json.dumps({'stub': True, 'verdict': os.environ.get('READ_VERDICT', '')}))"]
    for flag, rc in rcs.items():
        lines.append(f"if {flag!r} in a: sys.exit({rc})")
    lines.append("sys.exit(0)")
    p.write_text("\n".join(lines) + "\n")
    return p


def _run_arm_stub(tmp: Path) -> Path:
    """A fake run_arm.sh: logs argv (+ the env the queue hands a training launch); --row-eval / --lr-eval exit $ROWRC;
    a training call exits $ARMRC and, on success, writes a dummy record; $FLAG_REF gets a p2s_flag."""
    p = tmp / "run_arm_stub.sh"
    p.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        echo "$@" >> {tmp}/run_arm.calls
        case "$1" in
          --row-eval|--lr-eval) exit "${{ROWRC:-0}}" ;;
        esac
        echo "$1 ACCEPT_CODE_WHY=${{ACCEPT_CODE_WHY:-}} GUARD=${{P2_GUARD_REF:-}} STOP_ON_REF_STOP=${{STOP_ON_REF_STOP:-}}" \\
          >> {tmp}/train.env
        rc=${{ARMRC:-5}}
        if [ "$rc" = 0 ]; then
          mkdir -p {tmp}/out/arms
          printf '{{"checkpoint": "%s/m.pt", "driver_stamp": "%s/st"}}' {tmp} {tmp} > "{tmp}/out/arms/$1.record.json"
          [ "${{FLAG_REF:-}}" = "$1" ] && printf '+700\\tflag\\n' > "{tmp}/out/arms/$1.p2s_flag"
        fi
        exit "$rc"
        """))
    p.chmod(0o755)
    return p


def _env(tmp: Path, rcs: dict | None = None, **extra) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("KRABBY_", "P2_", "P2B_", "P2S_", "ROWRC", "ARMRC", "FLAG_REF", "FROM_STEP"))}
    env.update(P2_OUT=str(tmp / "out"), P2_PY=sys.executable, READ=str(_read_stub(tmp, rcs or {})),
               RUN_ARM=str(_run_arm_stub(tmp)))
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _seed_records(tmp: Path) -> None:
    arms = tmp / "out" / "arms"
    arms.mkdir(parents=True, exist_ok=True)
    for ref in ("P2__S1_s16", "P2__S1_s17", "P2__S1_s18", "P2__C0_s18"):
        (arms / f"{ref}.record.json").write_text(json.dumps({"checkpoint": str(tmp / f"{ref}.pt"),
                                                             "driver_stamp": str(tmp / f"{ref}_stamp")}))


def _run(env: dict, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(P2S_QUEUE)], env=env, capture_output=True, text=True, timeout=timeout)


def _calls(tmp: Path) -> list[str]:
    f = tmp / "run_arm.calls"
    return f.read_text().splitlines() if f.exists() else []


def _trained(tmp: Path) -> list[str]:
    return [c for c in _calls(tmp) if not c.startswith(("--row-eval", "--lr-eval"))]


def _log(tmp: Path) -> str:
    return (tmp / "out" / "queue.log").read_text()


def test_script_parses():
    assert subprocess.run(["bash", "-n", str(P2S_QUEUE)], capture_output=True).returncode == 0


def test_refs_rows_then_inertness_stop(tmp_path):
    """refs: REFS_S1_s18 (from P2__S1_s18) first, then C0R_s18 (from P2__C0_s18): lr, suite es1 + es2, the p2b field
    rows (field_hard 1-4) and walkstop at 1 and 2 from scenarios_p2s.yaml; a failed inertness read stops before any
    training."""
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {"--p2s-inertness": 4}))
    assert r.returncode == 4, r.stdout + r.stderr
    calls = _calls(tmp_path)
    names = [c.split()[1] for c in calls]
    first_c0r = names.index("C0R_s18")
    assert set(names[:first_c0r]) == {"REFS_S1_s18"} and set(names[first_c0r:]) == {"C0R_s18"}
    refs = [c for c in calls if c.split()[1] == "REFS_S1_s18"]
    assert refs[0].startswith("--lr-eval") and refs[1].split()[5:7] == ["suite__omni", "1"]   # the guard's reference
    rows = {(c.split()[5], c.split()[6]) for c in refs if c.startswith("--row-eval")}
    assert rows == {("suite__omni", "1"), ("suite__omni", "2"), ("field_light__omni", "1"), ("field_light__omni", "2"),
                    ("ring_obst__omni", "1"), ("ring_obst__omni", "2"), ("field_hard__omni", "1"),
                    ("field_hard__omni", "2"), ("field_hard__omni", "3"), ("field_hard__omni", "4"),
                    ("field_turn__omni", "1"), ("field_suite__omni", "1"), ("walkstop__omni", "1"),
                    ("walkstop__omni", "2")}
    ws = [c for c in refs if "walkstop__omni" in c]
    assert all(c.split()[4].endswith("scenarios_p2s.yaml") for c in ws)
    assert all(str(tmp_path / "P2__S1_s18.pt") in c for c in refs)                 # the S1 head's own checkpoint
    assert all(str(tmp_path / "P2__C0_s18.pt") in c for c in calls if " C0R_s18 " in f" {c} ")
    assert (tmp_path / "read.calls").read_text().split("\n")[1].strip() == "--p2s-inertness --p2s-seeds 18"
    assert "STOP p2s_queue: the reference inertness check at seeds 18" in _log(tmp_path)
    assert not _trained(tmp_path)


def test_the_guard_is_repointed_and_smokes_ok_is_only_logged(tmp_path):
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {"--p2s-inertness": 0, "--p2s-is-still": 4}, ARMRC=0, READ_VERDICT="NO EFFECT"))
    assert r.returncode == 4, r.stdout + r.stderr
    env = (tmp_path / "train.env").read_text()
    assert "GUARD=REFS_S1_s18" in env and "STOP_ON_REF_STOP=1" in env
    assert "SMOKES_OK check rc" in _log(tmp_path) and "not a gate" in _log(tmp_path)


def test_st2_not_still_skips_st1_and_stops(tmp_path):
    """ST1 only if ST2 reads STILL: --p2s-is-still rc 4 -> ST1 is never launched; the screen read runs and the queue
    STOPS (either way)."""
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {"--p2s-inertness": 0, "--p2s-is-still": 4}, ARMRC=0, READ_VERDICT="NO EFFECT"))
    assert r.returncode == 4, r.stdout + r.stderr
    assert _trained(tmp_path) == ["P2__ST2_s18"]
    log = _log(tmp_path)
    assert "reads NO EFFECT (not STILL) -- ST1 is not run" in log
    assert "P2S SCREEN rc 0" in log and "STOP p2s_queue: s18 screen done" in log and "NOTIFY" in log
    reads = (tmp_path / "read.calls").read_text()
    assert "--p2s-is-still P2__ST2_s18" in reads and "--p2s-screen" in reads
    # the arm's own reads: lr, suite es2 (es1 is its driver stamp), the field rows, walkstop 1 / 2
    st2 = [c for c in _calls(tmp_path) if c.startswith(("--row-eval", "--lr-eval")) and c.split()[1] == "P2__ST2_s18"]
    rows = [tuple(c.split()[5:7]) for c in st2 if c.startswith("--row-eval")]
    assert ("suite__omni", "1") not in rows and ("suite__omni", "2") in rows
    assert sum("walkstop__omni" in c for c in st2) == 2 and sum("field_hard__omni" in c for c in st2) == 4


def test_st2_still_runs_st1_then_stops(tmp_path):
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {"--p2s-inertness": 0, "--p2s-is-still": 0}, ARMRC=0))
    assert r.returncode == 4, r.stdout + r.stderr
    assert _trained(tmp_path) == ["P2__ST2_s18", "P2__ST1_s18"]
    assert "STOP p2s_queue: s18 screen done (P2__ST2_s18, P2__ST1_s18)" in _log(tmp_path)
    assert "p2s_queue replicates" not in _log(tmp_path)


def test_accept_code_change_is_passed_once(tmp_path):
    _seed_records(tmp_path)
    _run(_env(tmp_path, {"--p2s-inertness": 0, "--p2s-is-still": 0}, ARMRC=0))
    env = (tmp_path / "train.env").read_text().splitlines()
    assert env[0].startswith("P2__ST2_s18 ACCEPT_CODE_WHY=p2s step-3 code move")
    assert env[1].startswith("P2__ST1_s18 ACCEPT_CODE_WHY= ")


def test_a_failed_arm_stops_immediately(tmp_path):
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {"--p2s-inertness": 0}, ARMRC=5))
    assert r.returncode == 4
    assert _trained(tmp_path) == ["P2__ST2_s18"]
    assert "STOP p2s_queue: rc 5 in the arm P2__ST2_s18 step" in _log(tmp_path)
    assert "--p2s-is-still" not in (tmp_path / "read.calls").read_text()


def test_a_p2s_flag_stops_after_the_arms_reads(tmp_path):
    """The notify-only termination flag never kills; the queue asks after the flagged arm's reads."""
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {"--p2s-inertness": 0, "--p2s-is-still": 0}, ARMRC=0, FLAG_REF="P2__ST2_s18"))
    assert r.returncode == 4
    assert _trained(tmp_path) == ["P2__ST2_s18"]
    assert any("walkstop__omni 2" in c for c in _calls(tmp_path) if " P2__ST2_s18 " in f" {c} ")   # reads done
    log = _log(tmp_path)
    assert "STOP p2s_queue: P2__ST2_s18 carries the notify-only p2s termination flag" in log
    assert "FROM_STEP=st1" in log


def test_a_killed_reference_is_refused_fresh_names_run(tmp_path):
    _seed_records(tmp_path)
    (tmp_path / "out" / "arms" / "P2__ST2_s18.record.json").write_text(json.dumps({"killed": "p2s-guard"}))
    r = _run(_env(tmp_path, {}, ARMRC=0, FROM_STEP="st2"))
    assert r.returncode == 4
    assert not _trained(tmp_path)
    assert "P2__ST2_s18 was killed (p2s-guard)" in _log(tmp_path) and "fresh chain name" in _log(tmp_path)
    # the fresh name goes through the environment and runs
    r = _run(_env(tmp_path, {"--p2s-is-still": 4}, ARMRC=0, FROM_STEP="st2", P2S_ST2_S18="P2__ST2R1_s18"))
    assert _trained(tmp_path) == ["P2__ST2R1_s18"]
    assert "--p2s-is-still P2__ST2R1_s18" in (tmp_path / "read.calls").read_text()


def test_an_already_trained_reference_is_refused(tmp_path):
    _seed_records(tmp_path)
    (tmp_path / "out" / "arms" / "P2__ST2_s16.record.json").write_text(json.dumps({"checkpoint": "/x.pt"}))
    r = _run(_env(tmp_path, {}, ARMRC=0, FROM_STEP="rep16"))
    assert r.returncode == 4 and not _trained(tmp_path)
    assert "P2__ST2_s16 already has a record" in _log(tmp_path)


def test_refsrep_then_reps_at_s16_then_s17_then_recipe(tmp_path):
    """FROM_STEP=refsrep: REFS_S1_s16 / _s17 (from the S1 heads), their inertness (16,17), the kept dose at s16 then
    s17, the recipe read, STOP."""
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {}, ARMRC=0, FROM_STEP="refsrep", P2S_DOSE="ST1"))
    assert r.returncode == 4, r.stdout + r.stderr
    names = [c.split()[1] if c.startswith("--") else c.split()[0] for c in _calls(tmp_path)]
    assert names.index("REFS_S1_s16") < names.index("REFS_S1_s17") < names.index("P2__ST1_s16")
    assert _trained(tmp_path) == ["P2__ST1_s16", "P2__ST1_s17"]
    reads = (tmp_path / "read.calls").read_text().splitlines()
    assert [r for r in reads if r.startswith("--p2s-")] == ["--p2s-inertness --p2s-seeds 16,17",
                                                            f"--p2s-recipe --p2s-md {tmp_path}/out/p2s_recipe.md"]
    log = _log(tmp_path)
    assert "p2s_queue replicates: dose ST1" in log and "STOP p2s_queue: replicates done" in log and "NOTIFY" in log


def test_refsrep_inertness_difference_stops(tmp_path):
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {"--p2s-inertness": 4}, ARMRC=0, FROM_STEP="refsrep"))
    assert r.returncode == 4 and not _trained(tmp_path)
    assert "the reference inertness check at seeds 16,17" in _log(tmp_path)


def test_from_step_rep17_runs_only_s17(tmp_path):
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {}, ARMRC=0, FROM_STEP="rep17"))
    assert r.returncode == 4
    assert _trained(tmp_path) == ["P2__ST2_s17"]


def test_from_step_recipe_runs_only_the_recipe(tmp_path):
    r = _run(_env(tmp_path, {"--p2s-recipe": 0}, FROM_STEP="recipe"))
    assert r.returncode == 4
    assert not _calls(tmp_path)
    reads = [ln for ln in (tmp_path / "read.calls").read_text().splitlines() if ln.startswith("--p2s")]
    assert reads == [f"--p2s-recipe --p2s-md {tmp_path}/out/p2s_recipe.md"]


def test_queue_refuses_a_second_instance(tmp_path):
    env = _env(tmp_path, {}, FROM_STEP="recipe")
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "out" / "p2_queue.lock", "w") as fh:        # the SAME lock file p2_queue.sh / p2b_queue.sh use
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = _run(env)
    assert r.returncode == 3 and "already running" in r.stderr
    assert not _calls(tmp_path)
    assert "another p2_queue holds" in _log(tmp_path)


def test_bad_from_step_refuses(tmp_path):
    r = _run(_env(tmp_path, {}, FROM_STEP="reps"))
    assert r.returncode == 2 and "bad FROM_STEP" in r.stderr


def test_stop_queue_file_stops_at_start(tmp_path):
    (tmp_path / "out").mkdir(parents=True)
    (tmp_path / "out" / "STOP_QUEUE").write_text("")
    r = _run(_env(tmp_path, {}))
    assert r.returncode == 4 and not _calls(tmp_path)


# ===================================================================================== run_arm.sh p2s plumbing
def test_run_arm_p2s_watcher_block_and_preflight_arms():
    src = RUN_ARM.read_text()
    # the case pattern grew to also match stage-p2r Z arms (CHARTER 12 / P2R_STATUE_PLAN.md step 1: Z arms share
    # the same three guards, read against the same reference)
    assert "P2__ST[0-9]*_s*|P2__Z[0-9]*_s*)" in src and "--at p2s" in src and '--arms "$ARMS"' in src
    assert '"$rc" = 5' in src and ".p2s_flag" in src
    assert src.count("--preflight --chain") == 1 and "--candidates \"$C/candidates.yaml\" --arms \"$ARMS\" 2>&1) ||" in src
    assert subprocess.run(["bash", "-n", str(RUN_ARM)], capture_output=True).returncode == 0


def _watch_env(tmp: Path, live_rc: int) -> tuple[dict, str]:
    """run_arm.sh sourced (P2_RUN_ARM_LIB=1) with a stub p2_live.py: --iters -> "19994 20400"; --check ... --at p2s
    -> exit live_rc; every other read 0. systemctl is a stub that reports the unit active twice, then inactive."""
    bin_ = tmp / "bin"
    bin_.mkdir()
    (bin_ / "systemctl").write_text(textwrap.dedent(f"""\
        #!/usr/bin/env bash
        n=$(cat {tmp}/sc.n 2>/dev/null || echo 0); n=$((n+1)); echo $n > {tmp}/sc.n
        case "$*" in *is-active*) [ "$n" -le 2 ] && exit 0 || exit 3 ;; esac
        echo "$@" >> {tmp}/systemctl.calls; exit 0
        """))
    (bin_ / "systemctl").chmod(0o755)
    (bin_ / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_ / "sleep").chmod(0o755)
    scripts = tmp / "scripts"
    scripts.mkdir()
    live = scripts / "p2_live.py"
    live.write_text(textwrap.dedent(f"""\
        import sys
        a = sys.argv[1:]
        open({str(tmp / 'live.calls')!r}, 'a').write(' '.join(a) + chr(10))
        if '--iters' in a: print('19994 20400'); sys.exit(0)
        if '--at' in a and a[a.index('--at') + 1] == 'p2s': print('{{"p2s": "stub"}}'); sys.exit({live_rc})
        print('{{}}'); sys.exit(0)
        """))
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2_"))}
    camp = tmp / "camp"
    (camp / "logs").mkdir(parents=True)
    env.update(P2_OUT=str(tmp / "out"), CAMPAIGN_DIR=str(camp), P2_CAP="", P2_PY=sys.executable,
               P2_RUN_ARM_LIB="1", PATH=f"{bin_}:{env['PATH']}")
    script = f"""
        source {RUN_ARM}
        S={scripts}
        PREC={tmp}
        fork_source_rd() {{ echo {tmp}/src; }}
        printf 'import sys\\nprint("{{}}")\\nsys.exit(2)\\n' > {tmp}/resume_read.py
        watch_arm P2__ST2_s18 {tmp}/rd unit-x P2__ST2_s18 {tmp}/train.log
        echo "watch rc=$?"
    """
    return env, script


def test_run_arm_watcher_rc5_is_notify_only(tmp_path):
    env, script = _watch_env(tmp_path, 5)
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert "watch rc=0" in r.stdout, r.stdout + r.stderr
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "NOTIFY P2__ST2_s18 p2s termination flag at +406 (notify-only, the run continues" in log
    assert log.count("p2s termination flag") == 1                                    # notified once, not every poll
    assert (tmp_path / "out" / "arms" / "P2__ST2_s18.p2s_flag").read_text().startswith("+406\t")
    assert not (tmp_path / "out" / "STOP_QUEUE").exists() and "KILL" not in log      # never a kill
    assert not (tmp_path / "systemctl.calls").exists()
    assert any("--at p2s" in c and "--arms" in c for c in (tmp_path / "live.calls").read_text().splitlines())


def test_run_arm_watcher_rc3_kills(tmp_path):
    env, script = _watch_env(tmp_path, 3)
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert "watch rc=0" in r.stdout, r.stdout + r.stderr
    log = (tmp_path / "out" / "queue.log").read_text()
    assert "CHECKP2S P2__ST2_s18 rc 3" in log and "KILL P2__ST2_s18 (p2s-guard at +406" in log
    assert (tmp_path / "out" / "STOP_QUEUE").exists()
    assert "stop unit-x" in (tmp_path / "systemctl.calls").read_text()


def test_run_arm_write_record_carries_the_flag(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "P2_"))}
    camp = tmp_path / "camp"
    (camp / "logs").mkdir(parents=True)
    env.update(P2_OUT=str(tmp_path / "out"), CAMPAIGN_DIR=str(camp), P2_CAP="", P2_RUN_ARM_LIB="1")
    arms = tmp_path / "out" / "arms"
    arms.mkdir(parents=True)
    (arms / "P2__ST2_s18.p2s_flag").write_text("+700\tcrab_failure 0.29 > 0.20 + 0.08\n")
    script = f"""
        source {RUN_ARM}
        CHAIN=P2__ST2_s18 SEED=18 SMOKE=0 UNIT=u RD=/rd CK=/rd/m.pt ST=/st TRAINLOG=/t
        write_record P2__ST2_s18
        write_record P2__ST1_s18
    """
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    rec = json.loads((arms / "P2__ST2_s18.record.json").read_text())
    assert rec["p2s_flag"] == {"at": "+700", "why": "crab_failure 0.29 > 0.20 + 0.08"}
    assert "p2s_flag" not in json.loads((arms / "P2__ST1_s18.record.json").read_text())   # no flag: no key


def test_st2_still_read_not_final_stops(tmp_path):
    """fable review 2026-10-08: --p2s-is-still rc 4 WITHOUT a final verdict (PENDING / INVALID / a reader crash) STOPS
    the queue -- it must never read as "not STILL" and silently skip ST1."""
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {"--p2s-inertness": 0, "--p2s-is-still": 4}, ARMRC=0, READ_VERDICT="PENDING"))
    log = _log(tmp_path)
    assert r.returncode == 4, r.stdout + r.stderr
    assert _trained(tmp_path) == ["P2__ST2_s18"]
    assert "is not a final verdict" in log and "--p2s-screen" not in (tmp_path / "read.calls").read_text()


def test_until_step_stops_after_the_references(tmp_path):
    """UNTIL_STEP=inertness runs the references and the inertness read, then STOPS before any training launch."""
    _seed_records(tmp_path)
    r = _run(_env(tmp_path, {"--p2s-inertness": 0}, ARMRC=0, UNTIL_STEP="inertness"))
    assert r.returncode == 4, r.stdout + r.stderr
    assert _trained(tmp_path) == []
    assert "UNTIL_STEP=inertness reached" in _log(tmp_path)
