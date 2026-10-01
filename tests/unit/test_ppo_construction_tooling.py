"""PPO-construction campaign tooling (CPU only): the arm runner's kill / record / hand-class shell
functions and the verdict + rung-0b read-outs (plan foamy-wondering-ullman, review fixes F1-F10).

The shell tests lift the REAL functions out of ``scripts/run_arm.sh`` and run them under bash with
``systemctl``, ``sleep`` and the python interpreter stubbed, so a regression in the queue's kill
semantics (a wiring/dose kill that does not stop the queue, a killed run that leaves no record, a
retry that never happens) fails here instead of quietly during a 30-GPU-hour screen.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-10-01_ppo_construction"
SCRIPTS = CAMPAIGN / "scripts"
RUN_ARM = SCRIPTS / "run_arm.sh"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


verdict = _load(SCRIPTS / "verdict.py", "_ppo_construction_verdict")
rung0b = _load(SCRIPTS / "rung0b_read.py", "_ppo_construction_rung0b")


def _shell_fn(name: str) -> str:
    """One function of run_arm.sh, verbatim (they all close with a column-0 ``}``; one whose body is a
    python heredoc closes after its ``PY`` terminator -- the heredoc itself may hold a column-0 ``}``)."""
    src = RUN_ARM.read_text()
    start = re.search(rf"^{re.escape(name)}\(\) \{{", src, re.M)
    assert start, name
    body = src[start.start():]
    end = re.search(r"^PY\n\}\n" if "<<'PY'" in body.split("\n}\n", 1)[0] else r"^\}\n", body, re.M)
    assert end, name
    return body[:end.end()]


def _bash(script: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], cwd=cwd, capture_output=True, text=True, timeout=60)


# ============================================================ the scripts' own self-tests
def test_verdict_self_test_passes(capsys):
    assert verdict.self_test() == 0, capsys.readouterr().out


def test_rung0b_self_test_passes(capsys):
    assert rung0b.self_test() == 0, capsys.readouterr().out


# ============================================================ F3: hand class fails closed
@pytest.mark.parametrize("feet, along, want", [
    ("FAST", "FAST", "FAST"), ("FAST", "SLOW", "SLOW"), ("FAST", "INTERMEDIATE", "INTERMEDIATE"),
    ("FAST", "", "NA"),                       # along_cmd.py crashed / no json: never FAST
    ("", "FAST", "NA"),                       # no feet class: never FAST
    ("", "", "NA"), ("SLOW", "", "SLOW"), ("SLOW", "FAST", "SLOW"), ("DEAD", "FAST", "DEAD"),
])
def test_runner_and_verdict_hand_class_agree_and_fail_closed(tmp_path, feet, along, want):
    r = _bash(_shell_fn("hand_fast") + f'hand_fast "{feet}" "{along}"\n', tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == want
    assert verdict.hand_class(feet or None, along or None) == want


# ============================================================ F1/F2/F5: the live kills
FAKE_PY = r"""#!/usr/bin/env bash
# stand-in for "$PY": verdict.py --last-iter / --check-run, early_read.py
case "$2" in
  --last-iter) echo 1100; exit 0 ;;
  --check-run)
    at=""; while [ $# -gt 0 ]; do [ "$1" = --at ] && at=$2; shift; done
    var="RC_${at}"; seq=${!var:-0}
    n=$(cat "$STATE/$at" 2>/dev/null || echo 0); echo $((n + 1)) > "$STATE/$at"
    set -- $seq; i=$((n + 1)); [ "$i" -gt $# ] && i=$#
    rc=${!i}
    echo "{\"at\": \"$at\", \"rc\": $rc}"; exit "$rc" ;;
esac
echo "EARLY note"
"""


def _watch(tmp_path: Path, rc200: str, rc1000: str, rclive: str = "0", polls: int = 6, auto_kill: int = 1) -> dict:
    """Run the real kill_kind/kill_arm/watch_arm with stubs; return the log, the kill file, STOP_QUEUE, stops."""
    out, state = tmp_path / "out", tmp_path / "state"
    (out / "arms").mkdir(parents=True)
    state.mkdir()
    fake = tmp_path / "fakepy"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    script = textwrap.dedent(f"""\
        set -u
        OUT={out}; ARMS={out}/arms; LOG={out}/queue.log; S={tmp_path}; PY={fake}; AUTO_KILL={auto_kill}
        export STATE={state} RC_200="{rc200}" RC_1000="{rc1000}" RC_live="{rclive}"
        say() {{ echo "$*" >> "$LOG"; }}
        sleep() {{ :; }}
        N=0
        systemctl() {{
          case "$2" in
            is-active) N=$((N + 1)); [ "$N" -le {polls} ] ;;
            stop) echo "STOPPED $3" >> "$LOG"; N=999 ;;
          esac
        }}
        """) + _shell_fn("kill_kind") + _shell_fn("kill_arm") + _shell_fn("watch_arm") + \
        'watch_arm REF RD UNIT C_lr__A TL\n'
    r = _bash(script, tmp_path)
    assert r.returncode == 0, r.stderr
    log = (out / "queue.log").read_text() if (out / "queue.log").exists() else ""
    killed = out / "arms" / "REF.killed"
    return {"log": log, "killed": killed.read_text() if killed.exists() else None,
            "stop_queue": (out / "STOP_QUEUE").exists(), "calls": {p.name: int(p.read_text()) for p in state.iterdir()}}


def test_wiring_kill_at_200_stops_the_queue_and_leaves_a_kill_record(tmp_path):
    w = _watch(tmp_path, rc200="3", rc1000="0")
    assert w["killed"].split("\t")[:2] == ["wiring", "200"]
    assert w["stop_queue"] and "STOPPED UNIT" in w["log"] and "NOTIFY REF killed for wiring" in w["log"]


def test_dose_kill_at_1000_stops_the_queue(tmp_path):
    w = _watch(tmp_path, rc200="0", rc1000="4")
    assert w["killed"].split("\t")[:2] == ["dose", "1000"] and w["stop_queue"]


@pytest.mark.parametrize("rc1000, rclive, kind, at", [("5", "0", "dead", "1000"), ("0", "6", "diverged", "live")])
def test_dead_and_divergence_kills_let_the_queue_continue(tmp_path, rc1000, rclive, kind, at):
    w = _watch(tmp_path, rc200="0", rc1000=rc1000, rclive=rclive)
    assert w["killed"].split("\t")[:2] == [kind, at]
    assert not w["stop_queue"] and f"NOTIFY REF killed {kind}" in w["log"]


def test_not_enough_telemetry_is_retried_not_latched(tmp_path):
    """F5: rc 2 at the first poll must not latch the read (it used to be skipped for good)."""
    w = _watch(tmp_path, rc200="2 2 3", rc1000="0")
    assert w["calls"]["200"] == 3                               # retried until it decided
    assert w["killed"].split("\t")[:2] == ["wiring", "200"]
    assert w["log"].count("CHECK200 REF rc 2") == 1             # logged once, not every poll


def test_a_passing_run_is_never_killed_and_each_read_latches_once(tmp_path):
    w = _watch(tmp_path, rc200="0", rc1000="0", polls=5)
    assert w["killed"] is None and not w["stop_queue"]
    assert w["calls"]["200"] == 1 and w["calls"]["1000"] == 1 and w["calls"]["live"] == 5


def test_auto_kill_off_only_advises(tmp_path):
    w = _watch(tmp_path, rc200="3", rc1000="5", auto_kill=0, polls=4)
    assert w["killed"] is None and not w["stop_queue"] and "STOPPED" not in w["log"]
    assert "KILL-ADVISED REF (wiring" in w["log"] and "KILL-ADVISED REF (dead" in w["log"]


def test_kill_codes_match_verdict():
    for rc, kind in verdict.KILL_KIND.items():
        r = subprocess.run(["bash", "-c", _shell_fn("kill_kind") + f"kill_kind {rc}"], capture_output=True, text=True)
        assert r.stdout.strip() == kind
    assert (verdict.RC_WIRING, verdict.RC_DOSE, verdict.RC_DEAD, verdict.RC_DIVERGED) == (3, 4, 5, 6)
    src = RUN_ARM.read_text()
    # launch refused moved off 5 (now DEAD); the main loop exits on 6 and on STOP_QUEUE
    assert 'return 6; }' in src and '[ "$rc" = 6 ] && exit 6' in src and "exit 4" in src
    assert '[ "$rc" = 5 ] && exit 5' not in src


# ============================================================ F2: every killed run leaves a record verdict counts
def _write_record(tmp_path: Path, ref: str, chain: str, seed: int, killed: str | None) -> dict:
    arms = tmp_path / "arms"
    arms.mkdir(exist_ok=True)
    kf = ""
    if killed:
        (arms / f"{ref}.killed").write_text(killed)
        kf = str(arms / f"{ref}.killed")
    two = int(chain.startswith("C_lr"))
    script = textwrap.dedent(f"""\
        set -u
        ARMS={arms}; CHAIN={chain}; TWO={two}; SCREEN=1; SEED={seed}; UNIT=u; RD=/nonexistent; CK=""; ST=""
        TRAINLOG=/nonexistent.log; CODE_ROOT=/x
        """) + _shell_fn("write_record") + f'write_record {ref} "{kf}"\n'
    r = _bash(script, tmp_path)
    assert r.returncode == 0, r.stderr
    return json.loads((arms / f"{ref}.record.json").read_text())


def test_killed_runs_are_recorded_and_counted_or_excluded(tmp_path):
    cands = {"chains": {"C_lr__A": {"stages": [{"overlay": {"KRABBY_ACTION_BOUND_LOSS": "0.001"}}]},
                        "C_lr__C0": {"stages": [{"overlay": {}}]}}}
    dead = _write_record(tmp_path, "C_lr__A_s12", "C_lr__A", 12, "dead\t1000\t{\"why\": [\"DEAD\"]}\n")
    assert (dead["killed_kind"], dead["killed_at"], dead["seed"]) == ("dead", 1000, 12)
    wiring = _write_record(tmp_path, "C_lr__A_s11", "C_lr__A", 11, "wiring\t200\tno consumed line\n")
    rerun = _write_record(tmp_path, "C_lr__A_s11_r2", "C_lr__A", 11, None)   # seed from SEED, not the name
    assert rerun["seed"] == 11 and "killed" not in rerun
    facts = [verdict.facts_from_record(r, cands, with_tb=False) for r in (dead, wiring, rerun)]
    for f in facts:
        if f["ref"] == "C_lr__A_s11_r2":            # a completed run: give it its stamp reads
            f["hands"] = {"l": "SLOW", "r": "SLOW"}
    res = verdict.evaluate(facts)
    by = {f["ref"]: f for f in res["runs"]}
    assert by["C_lr__A_s12"]["excluded"] is None and by["C_lr__A_s12"]["fast"] is False      # DEAD counts
    assert "wiring" in by["C_lr__A_s11"]["excluded"]
    assert by["C_lr__A_s11_r2"]["excluded"] is None
    assert any("replaces" in n for n in by["C_lr__A_s11_r2"]["notes"])
    assert res["beds"]["C_lr"]["A"]["n_arm"] == 2


# ============================================================ F6: consumed lines are exact
def test_consumed_lines_reject_not_consumed(tmp_path):
    log = tmp_path / "t.log"
    log.write_text("[KNOB] KRABBY_RSI_CAM_STAGE=1 NOT consumed: RSI is off (no events...)\n"
                   "[KNOB] KRABBY_INIT_STD=0.8 consumed\n")
    assert verdict.consumed_lines(log, ["KRABBY_RSI_CAM_STAGE", "KRABBY_INIT_STD"]) == {
        "KRABBY_RSI_CAM_STAGE": False, "KRABBY_INIT_STD": True}


# ============================================================ F9: the live candidates file
def test_live_candidates_is_v2_and_every_c_lr_rsi_chain_states_its_staging():
    live = (CAMPAIGN / "candidates.yaml").read_text()
    assert live == (SCRIPTS / "candidates_v3.yaml").read_text()
    chains = yaml.safe_load(live)["chains"]
    rsi = {n: c for n, c in chains.items() if n.startswith("C_lr_rsi")}
    assert len(rsi) == 8
    for name, c in rsi.items():
        want = "1" if name.startswith("C_lr_rsi_st__") else "0"
        assert [str(st["overlay"].get("KRABBY_RSI_CAM_STAGE")) for st in c["stages"]] == [want], name
    # every chain that is not a C_lr_rsi one leaves the knob unset (X16's twins differ in this key only)
    assert not any("KRABBY_RSI_CAM_STAGE" in (st.get("overlay") or {})
                   for n, c in chains.items() if not n.startswith("C_lr_rsi") for st in c["stages"])


# ============================================================ rung 0b: plain schedule stamps (GPU smoke fix 2026-10-01)
def _schedule_stamp(root: Path, rid: str, n_eps: int, n_raw: int, complete: bool = False) -> Path:
    """A rung-0b stamp as ``report.write_run`` lays out a PLAIN schedule row: run_meta.json, then one raw npz per
    episode, and NO complete.json (that marker is written for category-suite rows only)."""
    d = root / rid / rid / "seed001" / "2026-10-01_00-00-00"
    (d / "raw").mkdir(parents=True)
    (d / "run_meta.json").write_text(json.dumps({"n_steps_per_env": [300] * n_eps, "termination_reason": []}))
    for i in range(n_raw):
        (d / "raw" / f"episode_{i:02d}.npz").write_bytes(b"x")
    if complete:
        (d / "complete.json").write_text("{}")
    return d


def test_rung0b_finds_plain_schedule_stamps_without_complete_json(tmp_path):
    """The first GPU rung-0b row wrote a full stamp and was refused as 'no complete stamp': the reader (and
    rung0b.sh, through the same function) demanded the suite-only complete.json."""
    rid = rung0b.row_id("rsi_on", 1)
    d = _schedule_stamp(tmp_path, rid, n_eps=3, n_raw=3)
    assert rung0b.stamp_complete(d)
    assert rung0b.find_stamps(tmp_path) == {rid: d}


def test_rung0b_rejects_a_partial_schedule_stamp(tmp_path):
    rid = rung0b.row_id("rsi_off", 2)
    d = _schedule_stamp(tmp_path, rid, n_eps=3, n_raw=2)          # the raw write was cut short
    assert not rung0b.stamp_complete(d)
    assert rung0b.find_stamps(tmp_path) == {}
    assert rung0b.stamp_complete(_schedule_stamp(tmp_path / "suite", rid, n_eps=3, n_raw=0, complete=True))


def test_rung0b_sh_uses_the_reader_completeness_check():
    src = (SCRIPTS / "rung0b.sh").read_text()
    assert "stamp_complete" in src and '[ ! -f "$d/complete.json" ]' not in src
