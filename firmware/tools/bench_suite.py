#!/usr/bin/env python3
"""Bench validation suite for board roles + SET/GET + joint motion (M17 Task 1).

Run against the real three-board rig to show, in one pass, that config commands
work on every board and every leg moves. Phases:

  1. config   — GET role/serial/version on FRONT and, via SET_LEFT/SET_RIGHT
                routing, on both followers; SET round-trip (rewrites each
                board's role to its current value, then reads it back).
  2. telemetry— all 18 joints reporting; pot values in ADC range.
  3. wiggle   — each leg in turn, ONE JOINT AT A TIME via per-joint J commands
                (the same path the GUI's buttons use); operator confirms motion
                (pot deltas printed as supporting evidence).
  4. multileg — two legs on different boards jogged simultaneously via one
                batch (B) command — the simultaneity/load test.
  5. powercycle (optional) — operator power-cycles the rig; roles must
                persist and telemetry must resume.

Usage (on the bench host, from the directory containing firmware/):
    python3 -m firmware.tools.bench_suite                 # local port autodetect
    python3 -m firmware.tools.bench_suite --port /dev/ttyUSB0
    python3 -m firmware.tools.bench_suite --remote orin1  # from a dev machine
    python3 -m firmware.tools.bench_suite --skip-motion   # config checks only

Motion phases move motors: keep the rig clear. PWM and pulse length are small
by default (--pwm, --pulse to adjust).

Every run is recorded to bench_runs.sqlite3 next to this script: one `runs` row
(who/where/when/args/outcome + board roles/serials/versions), a `checks` row per
check, and a `wiggles` row per joint per attempt with its pot before/after/delta
and the operator's answer. Inspect with e.g.:
    sqlite3 firmware/tools/bench_runs.sqlite3 'SELECT * FROM runs ORDER BY id DESC LIMIT 5'
"""
from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from firmware.krabby_mcu import KrabbyMCUSDK

# Leg -> its three joints. FRONT board drives FL+FR, LEFT drives RL+ML, RIGHT drives RR+MR.
LEGS = {
    "FL": ["FLHY", "FLHL", "FLKL"],
    "FR": ["FRHY", "FRHL", "FRKL"],
    "RL": ["RLHY", "RLHL", "RLKL"],
    "ML": ["MLHY", "MLHL", "MLKL"],
    "RR": ["RRHY", "RRHL", "RRKL"],
    "MR": ["MRHY", "MRHL", "MRKL"],
}
ALL_JOINTS = [j for joints in LEGS.values() for j in joints]
BOARDS_EXPECTED = {None: "FRONT", "left": "LEFT", "right": "RIGHT"}

DB_PATH = Path(__file__).resolve().parent / "bench_runs.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TEXT NOT NULL,
    finished_at     TEXT,
    host            TEXT,               -- machine the suite ran on
    remote          TEXT,               -- --remote ssh host, if used
    port            TEXT,               -- resolved serial port / socket URL
    git_commit      TEXT,               -- repo HEAD if available
    pwm             INTEGER,
    pulse           REAL,
    skip_motion     INTEGER,
    skip_powercycle INTEGER,
    aborted         INTEGER,
    abort_reason    TEXT,               -- operator's stated reason, when aborted
    passed          INTEGER,
    failed          INTEGER,
    boards_json     TEXT                -- role/serial/version per board (from config phase)
);
CREATE TABLE IF NOT EXISTS checks (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id  INTEGER NOT NULL REFERENCES runs(id),
    seq     INTEGER NOT NULL,
    ts      TEXT NOT NULL,
    phase   TEXT,
    name    TEXT NOT NULL,
    status  TEXT NOT NULL,              -- PASS / FAIL
    detail  TEXT
);
CREATE TABLE IF NOT EXISTS board_snapshots (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id   INTEGER NOT NULL REFERENCES runs(id),
    ts       TEXT NOT NULL,
    moment   TEXT NOT NULL,              -- 'initial' / 'post_powercycle'
    board    TEXT NOT NULL,              -- front / left / right
    responding INTEGER NOT NULL,         -- did it answer GET at all
    role     TEXT,
    serial   TEXT,
    version  TEXT
);
CREATE TABLE IF NOT EXISTS wiggles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    ts          TEXT NOT NULL,
    check_name  TEXT NOT NULL,          -- e.g. "wiggle FL", "multi-leg simultaneous motion"
    attempt     INTEGER NOT NULL,       -- 1..n ('a'=again re-runs bump this)
    joint       TEXT NOT NULL,          -- motor name, e.g. FLHL
    pot_before  INTEGER,
    pot_after   INTEGER,
    pot_delta   INTEGER,
    pwm         INTEGER,
    pulse       REAL,
    answer      TEXT                    -- operator answer for this attempt: y/n/a/q
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


class Recorder:
    """Appends every run/check/wiggle to bench_runs.sqlite3 as it happens."""

    def __init__(self, args: argparse.Namespace):
        self.db = sqlite3.connect(DB_PATH)
        self.db.executescript(_SCHEMA)
        # Migrate databases created before abort_reason existed.
        cols = {row[1] for row in self.db.execute("PRAGMA table_info(runs)")}
        if "abort_reason" not in cols:
            self.db.execute("ALTER TABLE runs ADD COLUMN abort_reason TEXT")
        self.phase = "setup"
        self.seq = 0
        self.boards: dict[str, dict] = {}
        cur = self.db.execute(
            "INSERT INTO runs (started_at, host, remote, pwm, pulse, skip_motion,"
            " skip_powercycle, git_commit) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (_now(), platform.node(), args.remote, args.pwm, args.pulse,
             int(args.skip_motion), int(args.skip_powercycle), _git_commit()))
        self.run_id = cur.lastrowid
        self.db.commit()

    def set_port(self, port: str) -> None:
        self.db.execute("UPDATE runs SET port = ? WHERE id = ?", (port, self.run_id))
        self.db.commit()

    def check(self, status: str, name: str, detail: str) -> None:
        self.seq += 1
        self.db.execute(
            "INSERT INTO checks (run_id, seq, ts, phase, name, status, detail)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self.run_id, self.seq, _now(), self.phase, name, status, detail))
        self.db.commit()

    def wiggle(self, check_name: str, attempt: int, joints: list[str],
               before: dict[str, int], after: dict[str, int],
               pwm: int, pulse: float) -> None:
        rows = [(self.run_id, _now(), check_name, attempt, j,
                 before.get(j), after.get(j),
                 (after[j] - before[j]) if j in before and j in after else None,
                 pwm, pulse)
                for j in joints]
        self.db.executemany(
            "INSERT INTO wiggles (run_id, ts, check_name, attempt, joint,"
            " pot_before, pot_after, pot_delta, pwm, pulse)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
        self.db.commit()

    def board_snapshot(self, moment: str, board: str, got: dict | None) -> None:
        self.db.execute(
            "INSERT INTO board_snapshots (run_id, ts, moment, board, responding,"
            " role, serial, version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (self.run_id, _now(), moment, board, int(got is not None),
             (got or {}).get("role"), (got or {}).get("serial"),
             (got or {}).get("version")))
        self.db.commit()

    def wiggle_answer(self, check_name: str, attempt: int, answer: str) -> None:
        self.db.execute(
            "UPDATE wiggles SET answer = ? WHERE run_id = ? AND check_name = ? AND attempt = ?",
            (answer, self.run_id, check_name, attempt))
        self.db.commit()

    def finish(self, aborted: bool, passed: int, failed: int,
               abort_reason: str | None = None) -> None:
        self.db.execute(
            "UPDATE runs SET finished_at = ?, aborted = ?, abort_reason = ?,"
            " passed = ?, failed = ?, boards_json = ? WHERE id = ?",
            (_now(), int(aborted), abort_reason, passed, failed,
             json.dumps(self.boards) if self.boards else None, self.run_id))
        self.db.commit()
        self.db.close()


RECORDER: Recorder | None = None
results: list[tuple[str, str, str]] = []  # (status, name, detail)


class BenchAbort(Exception):
    """Operator chose to abort — stop all remaining phases (hold-all still runs)."""


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if RECORDER:
        RECORDER.check(status, name, detail)


def get_retry(mcu: KrabbyMCUSDK, *keys: str, board=None, tries: int = 3, timeout: float = 2.0):
    """send_get with retries: a single GET can lose the 300 ms relay window to
    the telemetry stream on the follower path, so one miss is not a failure."""
    for _ in range(tries):
        if (got := mcu.send_get(*keys, board=board, timeout=timeout)) is not None:
            return got
    return None


def _print_board_summary(moment: str, snaps: dict[str, dict | None]) -> None:
    responding = sum(1 for g in snaps.values() if g is not None)
    roles = "  ".join(f"{b}={g['role'] if g else '-'}" for b, g in snaps.items())
    print(f"    boards responding ({moment}): {responding}/{len(snaps)}  {roles}")


def phase_config(mcu: KrabbyMCUSDK) -> None:
    RECORDER.phase = "config"
    snaps: dict[str, dict | None] = {}
    for board, want_role in BOARDS_EXPECTED.items():
        label = board or "front"
        got = get_retry(mcu, "role", "serial", "version", board=board)
        RECORDER.board_snapshot("initial", label, got)
        snaps[label] = got
        if got is None:
            record("FAIL", f"get {label}", "no reply")
            continue
        RECORDER.boards[label] = got
        if got.get("role") != want_role:
            record("FAIL", f"get {label}", f"role={got.get('role')!r}, expected {want_role}")
            continue
        record("PASS", f"get {label}", f"role={got['role']} serial={got.get('serial')} version={got.get('version')}")
        # SET round-trip: rewrite the current role (exercises SET -> EEPROM -> GET
        # on this board, including follower routing, without changing anything).
        mcu.send_set(board=board, role=want_role)
        back = get_retry(mcu, "role", board=board)
        if back and back.get("role") == want_role:
            record("PASS", f"set round-trip {label}")
        else:
            record("FAIL", f"set round-trip {label}", f"read back {back!r}")
    _print_board_summary("initial", snaps)


def phase_telemetry(mcu: KrabbyMCUSDK, wait: float = 3.0) -> None:
    RECORDER.phase = "telemetry"
    deadline = time.time() + wait
    while time.time() < deadline and any(j not in mcu.joints for j in ALL_JOINTS):
        time.sleep(0.1)
    missing = [j for j in ALL_JOINTS if j not in mcu.joints]
    if missing:
        record("FAIL", "telemetry: all 18 joints reporting", f"missing {missing}")
    else:
        record("PASS", "telemetry: all 18 joints reporting")
    bad = [f"{j}={jt.pot}" for j in ALL_JOINTS
           if (jt := mcu.joints.get(j)) and not (0 <= jt.pot <= 1023)]
    if bad:
        record("FAIL", "telemetry: pot values in ADC range", ", ".join(bad))
    else:
        record("PASS", "telemetry: pot values in ADC range")


def _pots(mcu: KrabbyMCUSDK, joints: list[str]) -> dict[str, int]:
    return {j: mcu.joints[j].pot for j in joints if j in mcu.joints}


def _wiggle_joint(mcu: KrabbyMCUSDK, joint: str, pwm: int, pulse: float) -> None:
    """Out and back on ONE joint via per-joint J commands — the exact path the
    GUI's Extend/Retract buttons use (send_command_jog), one motor at a time.
    A stop between directions lets the firmware's PWM ramp (5/10 ms) unwind
    instead of eating the whole reverse pulse."""
    for direction in (pwm, -pwm):
        mcu.send_command_jog(joint, direction)
        time.sleep(pulse)
        mcu.send_command_jog(joint, 0)
        time.sleep(0.2)


def _wiggle_sequential(mcu: KrabbyMCUSDK, joints: list[str], pwm: int, pulse: float) -> None:
    for j in joints:
        _wiggle_joint(mcu, j, pwm, pulse)


def _wiggle_batch(mcu: KrabbyMCUSDK, joints: list[str], pwm: int, pulse: float) -> None:
    """All joints at once via one batch (B) command — only used where
    simultaneity is the thing under test (the multileg phase)."""
    for direction in (pwm, -pwm):
        mcu.send_commands_jog({j: direction for j in joints})
        time.sleep(pulse)
        mcu.send_commands_jog({j: 0 for j in joints})
        time.sleep(0.2)


def _wiggle_and_confirm(mcu: KrabbyMCUSDK, joints: list[str], pwm: int, pulse: float,
                        check_name: str, question: str, batch: bool) -> bool:
    """Wiggle, show pot deltas, ask the operator — 'a' repeats, 'q' aborts."""
    attempt = 0
    while True:
        attempt += 1
        before = _pots(mcu, joints)
        (_wiggle_batch if batch else _wiggle_sequential)(mcu, joints, pwm, pulse)
        time.sleep(0.3)
        after = _pots(mcu, joints)
        RECORDER.wiggle(check_name, attempt, joints, before, after, pwm, pulse)
        deltas = "  ".join(f"{j} Δpot={after.get(j, 0) - before.get(j, 0):+d}" for j in joints)
        print(f"    {deltas}")
        while True:
            ans = (input(f"{question} [y/n/a=again/q=abort] ").strip().lower() or "?")[0]
            if ans in "ynaq":
                break
            print("    please answer y, n, a, or q")
        RECORDER.wiggle_answer(check_name, attempt, ans)
        if ans == "q":
            raise BenchAbort
        if ans != "a":
            return ans == "y"


def phase_wiggle(mcu: KrabbyMCUSDK, pwm: int, pulse: float) -> None:
    RECORDER.phase = "wiggle"
    for leg, joints in LEGS.items():
        input(f"About to wiggle leg {leg} ({', '.join(joints)}), one joint at a time — clear? [enter] ")
        moved = _wiggle_and_confirm(mcu, joints, pwm, pulse,
                                    f"wiggle {leg}", f"Did all three {leg} joints move?",
                                    batch=False)
        record("PASS" if moved else "FAIL", f"wiggle {leg}")


def phase_multileg(mcu: KrabbyMCUSDK, pwm: int, pulse: float) -> None:
    RECORDER.phase = "multileg"
    # Two legs on two different follower boards, driven in one batch command.
    # NOTE: six motors at once is a real load test — if legs move strongly in
    # the sequential wiggles but falter here, suspect 24 V supply sag, not comms.
    joints = LEGS["RL"] + LEGS["MR"]
    input("About to wiggle legs RL and MR SIMULTANEOUSLY — clear? [enter] ")
    moved = _wiggle_and_confirm(mcu, joints, pwm, pulse,
                                "multi-leg simultaneous motion",
                                "Did BOTH legs (RL and MR) move together?",
                                batch=True)
    record("PASS" if moved else "FAIL", "multi-leg simultaneous motion")


def phase_powercycle(mcu: KrabbyMCUSDK) -> None:
    RECORDER.phase = "powercycle"
    ans = input("Power-cycle the whole rig now (all three boards), then press [enter] "
                "once it's back on — or press [s] to skip... ").strip().lower()
    if ans.startswith("s"):
        record("SKIP", "power-cycle persistence", "skipped by operator")
        return
    # The serial device vanished during the cycle. The bridge reopens it on its
    # own, but it drops its client socket when the device dies — which kills our
    # reader thread — so this connection is dead no matter what. Reconnect fresh,
    # and clear the joints dict so pre-cycle telemetry can't satisfy the re-check.
    mcu.close()
    mcu.joints.clear()
    # Wait (up to 90 s) for the board to be reachable again, instead of a few
    # blind open() attempts: USB re-enumeration takes a human-scale moment, the
    # operator may still be plugging things in, and a replug can land the board
    # on a NEW device node (ttyUSB0 -> ttyUSB1 in a different physical port) —
    # so re-detect by USB ID whenever the original path is gone. Only attempt
    # connect() once a plausible device exists, so we don't spew open() errors.
    # Bridge (socket://) URLs never change — the bridge re-detects its own end.
    connected = False
    announced = False
    deadline = time.time() + 90.0
    while time.time() < deadline:
        time.sleep(2.0)
        port_ok = "://" in mcu.port or Path(mcu.port).exists()
        if not port_ok:
            from firmware.mcu_port import default_port
            try:
                new_port = default_port()
            except RuntimeError:
                new_port = mcu.port
            if new_port != mcu.port and Path(new_port).exists():
                print(f"    board re-enumerated: {mcu.port} -> {new_port}")
                mcu.port = new_port
                port_ok = True
            elif not announced:
                print("    waiting for the board to enumerate (up to 90 s)...")
                announced = True
        if port_ok and mcu.connect(hold=True):
            connected = True
            break
    if not connected:
        record("FAIL", "reconnect after power cycle", f"could not reopen {mcu.port}")
        return
    record("PASS", "reconnect after power cycle")
    snaps: dict[str, dict | None] = {}
    for board, want_role in BOARDS_EXPECTED.items():
        label = board or "front"
        got = get_retry(mcu, "role", "serial", "version", board=board, tries=5, timeout=3.0)
        RECORDER.board_snapshot("post_powercycle", label, got)
        snaps[label] = got
        if got and got.get("role") == want_role:
            record("PASS", f"role persisted after power cycle: {label}")
        else:
            record("FAIL", f"role persisted after power cycle: {label}", f"got {got!r}")
    _print_board_summary("post_powercycle", snaps)
    phase_telemetry(mcu, wait=5.0)
    RECORDER.phase = "powercycle"


def main() -> int:
    global RECORDER
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None, help="serial port (default: auto-detect)")
    ap.add_argument("--remote", metavar="HOST", default=None,
                    help="ssh host the rig is attached to (starts the serial/TCP bridge)")
    ap.add_argument("--pwm", type=int, default=60, help="jog PWM for wiggles (default 60, matching the GUI slider max)")
    ap.add_argument("--pulse", type=float, default=0.8,
                    help="seconds per jog direction (default 0.8 — must comfortably exceed the "
                         "firmware's PWM ramp, ~120 ms to reach 60)")
    ap.add_argument("--skip-motion", action="store_true", help="config + telemetry checks only")
    ap.add_argument("--skip-powercycle", action="store_true", help="skip the power-cycle persistence phase")
    args = ap.parse_args()
    if args.remote and args.port:
        ap.error("--remote and --port are mutually exclusive")

    RECORDER = Recorder(args)

    bridge = None
    port = args.port
    if args.remote:
        from firmware.gui.remote import start_bridge
        bridge, port = start_bridge(args.remote)

    mcu = KrabbyMCUSDK(port=port)
    RECORDER.set_port(mcu.port)
    if not mcu.connect(hold=True):
        RECORDER.finish(aborted=True, passed=0, failed=0)
        sys.exit(f"could not open {mcu.port}")
    aborted = False
    abort_reason = None
    try:
        phase_config(mcu)
        phase_telemetry(mcu)
        if not args.skip_motion:
            phase_wiggle(mcu, args.pwm, args.pulse)
            phase_multileg(mcu, args.pwm, args.pulse)
            if not args.skip_powercycle:
                phase_powercycle(mcu)
        elif not args.skip_powercycle:
            phase_powercycle(mcu)
    except BenchAbort:
        aborted = True
        abort_reason = input("Reason for abort? ").strip() or None
        print("Aborted by operator — skipping remaining phases (joints held).")
    finally:
        try:
            mcu.send_command_joints_hold()
        except Exception:
            pass
        mcu.close()
        if bridge:
            bridge.stop()

    fails = [r for r in results if r[0] == "FAIL"]
    RECORDER.finish(aborted=aborted, passed=len(results) - len(fails), failed=len(fails),
                    abort_reason=abort_reason)
    print(f"\n{len(results)} checks: {len(results) - len(fails)} passed, {len(fails)} failed"
          + (" (ABORTED — incomplete)" if aborted else ""))
    for _, name, detail in fails:
        print(f"  FAILED: {name}" + (f" — {detail}" if detail else ""))
    print(f"run #{RECORDER.run_id} recorded in {DB_PATH}")
    return 1 if fails or aborted else 0


if __name__ == "__main__":
    sys.exit(main())
