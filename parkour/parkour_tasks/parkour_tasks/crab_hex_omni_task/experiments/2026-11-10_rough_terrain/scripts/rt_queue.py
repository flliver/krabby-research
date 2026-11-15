#!/usr/bin/env python3
"""The rough-terrain campaign's ONE queue runner (PLAN steps 5, 6, 8, 9; memory: gpu-queue-chaining).

    rt_queue.py --block smoke|baseline|train|gate [--new-ckpt ABS] [--from LABEL] [--only LABEL ...] [--dry] [--list]

Launch it detached with ``rt_launch.sh BLOCK [args]`` (setsid nohup; survives the session) and watch the log with
``rt_wait.sh LOG FROM_LINE`` (a log sentinel; never pgrep). The plan of each block is ``rt_machinery.plan`` (one
source; ``--list`` prints it). Every step prints one sentinel ``RTQ <label> rc=<n> [note]`` and the run ends with
``RTQ_DONE <block> fail=<n> [note]`` (an exception anywhere in the run still ends the log with
``RTQ_DONE <block> fail=<n+1> (queue crashed: <Type: msg>)`` and a non-zero exit); a child's own key lines (BUILD_CHECK, TPROW END, KITRUN END, RT TRAIN,
RTQ NOTIFY, RTSMOKE, RT GATES / AUDIT / ENVS ...) are echoed to the queue log, everything else goes to the step's own
log under ``<OUT>/logs/<block>/``.

One Isaac process at a time: the runner holds an exclusive flock on ``<campaign>/.isaac.lock`` for its whole run (a
second queue refuses), and before every Isaac step it waits while any Isaac python runs (distill.sh's anchored
pattern + ``rt_kit_*.py``; ``ps -ww``, never ``pgrep -f``); gate_row.sh / rt_kit_run.sh / rt_train.py wait again
themselves. Before every Isaac step: the Kit DerivedDataCache <= 80 GB and >= RTQ_FREE_MIN_GB (20) GB free disk, and
after it the cache grew < 3 GB (the corruption signature balloons ~2.4 GB/s); any breach stops the queue.

A gate row whose output slot already holds a complete stamp (tp_row.json, rc 0) AT THE PINNED BUILD (its
code_state.json: code_sha / eval_sha == results/pin.json's, dirty "False") is SKIPPED (``--from`` restarts and
relaunches are safe); a complete stamp at any other build (a pre-re-pin smoke) is ignored, and the row relaunches.
``@stamp:`` placeholders and the post-run stamp check read pinned-build stamps only; with no pin (a dry pass before
``build_check.py pin``) nothing counts as complete and nothing is skipped. A failed row is counted and the queue goes on; a failed check step (``stop_on_fail``: the build
pin, the env checks, the trainer) stops it. ``touch <campaign>/results/STOP_QUEUE`` stops it between steps.

``--dry``: nothing launches and nothing is written. Every gate row runs gate_row.sh with TP_ROW_DRY=1 (it prints the
composed env, the stripped keys and the harness command), every Kit step rt_kit_run.sh with KIT_RUN_DRY=1 (the
composed env and the capped command), the trainer rt_train.py --dry-run (its pre-flight table, the composed child env,
the wrapped command); read-only CPU checks run, the other CPU steps print their command.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import fcntl  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rt_machinery as M  # noqa: E402

ECHO_RE = re.compile(r"^(BUILD_CHECK|BUILD_PIN|TPROW END|KITRUN END|RT TRAIN|RTQ NOTIFY|RTSMOKE|RT GATES|RT AUDIT|"
                     r"RT ENVS|RT TRAINENV|RT GATEIDENT|RT PARAMS (OK|BAD|DISALLOWED)|RT VALIDATE|RT wrote|RT THRESHOLDS|\[rt smoke\] (PASS|FAIL|wrote)|"
                     r"\[rt probe\])")
BUSY_POLL_S = 30


def say(line: str) -> None:
    print(line, flush=True)


#: the RTQ_* guard knobs the launchers honour too (rt_kit_run.sh; rt_train.py's PF7 through rt_machinery): passed through
RTQ_PASS = ("RTQ_DDC_MAX_GB", "RTQ_FREE_MIN_GB")


def runner_env(step: M.Step, dry: bool) -> dict:
    """The child's env: this process's env minus every KRABBY_* / TP_* / KIT_* / RTQ_* key (the launchers compose the
    row env from their env file only) EXCEPT the two guard knobs :data:`RTQ_PASS` (a raised RTQ_DDC_MAX_GB must reach
    rt_kit_run.sh and rt_train.py, else the queue's guard passes and the launcher's refuses), plus the step's own
    launcher variables."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("KRABBY_", "TP_", "KIT_", "RTQ_")) or k in RTQ_PASS}
    env.update(step.env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if step.how == "cpu" or dry:
        env["CUDA_VISIBLE_DEVICES"] = ""            # CPU steps, and every step of a dry pass (nothing launches)
    elif env.get("CUDA_VISIBLE_DEVICES") == "":
        del env["CUDA_VISIBLE_DEVICES"]             # a CPU-only parent shell must not hide the GPU from Kit
    if dry and step.how == "gate":
        env["TP_ROW_DRY"] = "1"
    if dry and step.how == "kit":
        env["KIT_RUN_DRY"] = "1"
    return env


def pinned_stamps(slot, pin: dict | None) -> list:
    """The complete stamps of a slot recorded at the pinned build, oldest first; none at all without a pin."""
    return [] if pin is None else M.complete_stamps(slot, pin=pin)


def resolve(argv: list, steps_by_label: dict, pin: dict | None) -> tuple[list, str | None]:
    """``@stamp:<label>`` -> the newest complete stamp of that gate step's slot at the pinned build."""
    out = []
    for a in argv:
        if isinstance(a, str) and a.startswith("@stamp:"):
            src = steps_by_label.get(a[len("@stamp:"):])
            if src is None or not src.slot:
                return argv, f"{a}: no such gate step"
            tp_out, name, sc, es = src.slot
            st = pinned_stamps(M.slot_dir(tp_out, name, sc, es), pin)
            if not st:
                return argv, (f"{a}: no complete stamp at the pinned build in {M.slot_dir(tp_out, name, sc, es)}"
                              + ("" if pin is not None else f" (no pin {M.PIN_PATH})"))
            out.append(str(st[-1]))
        else:
            out.append(a)
    return out, None


def run_child(argv: list, env: dict, log: Path | None) -> int:
    fh = open(log, "a") if log else None
    try:
        if fh:
            fh.write(f"### {M.now_iso()} {' '.join(argv)}\n")
            fh.flush()
        p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             text=True, bufsize=1, cwd=str(M.W))
        for line in p.stdout:
            if fh:
                fh.write(line)
            if fh is None or ECHO_RE.match(line):
                # child lines are indented in the queue log (never read as the queue's own sentinels), except the
                # iteration-1500 NOTIFY line, which rt_wait.sh must see at column 0
                lead = "" if (fh is None or line.startswith("RTQ NOTIFY ")) else "  "
                sys.stdout.write(lead + line)
                sys.stdout.flush()
        return p.wait()
    finally:
        if fh:
            fh.close()


def guards() -> tuple[float, list]:
    ddc = M.ddc_gb()
    free = {str(p): M.free_gb(p) for p in (M.MAIN, M.W) if Path(p).exists()}
    return ddc, M.guard_problems(ddc, free)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block", required=True, choices=M.BLOCKS)
    ap.add_argument("--new-ckpt", default=None)
    ap.add_argument("--thresholds", type=Path, default=M.THRESHOLDS)
    ap.add_argument("--train-name", default=M.TRAIN_NAME)
    ap.add_argument("--from", dest="from_label", default=None)
    ap.add_argument("--only", action="append", default=None, help="run only these labels (checks always run)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    st = {"fail": 0}
    try:
        return run(a, st)
    except (Exception, KeyboardInterrupt) as exc:      # any crash still ends the log with the sentinel rt_wait.sh reads
        why = " ".join(f"{type(exc).__name__}: {exc}".split())[:300]
        say(M.done_line(a.block, st["fail"] + 1, f"(queue crashed: {why})"))
        return 1


def run(a, st: dict) -> int:
    """The queue proper; ``st["fail"]`` is the failure tally (read by :func:`main` when this raises)."""
    try:
        steps = M.plan(a.block, a.new_ckpt, a.thresholds, a.train_name)
    except (ValueError, RuntimeError, OSError) as exc:
        say(M.done_line(a.block, 1, f"(plan refused: {exc})"))
        return 1
    by_label = {s.label: s for s in steps}
    pin = M.read_pin()          # read once: the build every stamp the queue skips on / substitutes must carry
    if a.list:
        M._print_plan(steps)
        return 0
    for lab in [a.from_label] + list(a.only or []):
        if lab and lab not in by_label:
            say(M.done_line(a.block, 1, f"(unknown label {lab!r})"))
            return 2
    say(f"RTQ_START block={a.block} dry={int(a.dry)} steps={len(steps)} from={a.from_label or ''} "
        f"only={','.join(a.only or [])} out={M.OUT} at={M.now_iso()}")
    lock = None
    if not a.dry:
        M.LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        lock = open(M.LOCK_PATH, "a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            say(M.done_line(a.block, 1, f"(another queue holds {M.LOCK_PATH})"))
            return 3
    logd = M.LOGS / a.block
    if not a.dry:
        logd.mkdir(parents=True, exist_ok=True)
    st["fail"] = 0
    on = a.from_label is None
    for s in steps:
        is_check = s.label.endswith((".check", ".envs", ".trainenv", ".envcheck", ".thresholds", ".ident"))
        if s.label == a.from_label:
            on = True
        if not is_check and (not on or (a.only and s.label not in a.only)):
            continue
        if not a.dry and M.STOP_FILE.exists():
            say(M.done_line(a.block, st["fail"], f"(stopped by {M.STOP_FILE} before {s.label})"))
            return 1
        if s.note:
            say(f"# [{s.label}] {s.note}")
        if s.slot:
            done = pinned_stamps(M.slot_dir(*s.slot), pin)
            if done:
                say(M.sentinel(s.label, 0, f"(slot complete at the pinned build: {done[-1]}; skipped)"))
                continue
            other = len(M.complete_stamps(M.slot_dir(*s.slot))) - len(done)
            if other:
                say(f"# [{s.label}] {other} complete stamp(s) in the slot at another build"
                    f"{'' if pin is not None else ' (no pin)'}: ignored, the row relaunches")
        argv_r, why = resolve(s.argv, by_label, pin)
        if why:
            if a.dry:
                say(f"# [{s.label}] (dry) {why}: a real run substitutes it")
                argv_r = s.argv
            else:
                say(M.sentinel(s.label, 7, f"({why})"))
                st["fail"] += 1
                if s.stop_on_fail:
                    say(M.done_line(a.block, st["fail"], f"(stopped at {s.label})"))
                    return 1
                continue
        env = runner_env(s, a.dry)
        if a.dry:
            shown = " ".join(f"{k}={v}" for k, v in sorted(s.env.items()))
            say(f"# [{s.label}] {s.how}: {shown + ' ' if shown else ''}{' '.join(map(str, argv_r))}")
            if s.how == "cpu" and not s.dry_exec:
                say(M.sentinel(s.label, 0, "(dry: not run)"))
                continue
            if s.how == "train":
                argv_r = [*argv_r, "--dry-run"]
            rc = run_child(argv_r, env, None)
            say(M.sentinel(s.label, rc, "(dry)"))
            if rc != 0:
                st["fail"] += 1
            continue
        ddc0 = None
        if s.isaac:
            ddc0, probs = guards()
            if probs:
                say(M.sentinel(s.label, 5, f"(guard: {'; '.join(probs)})"))
                say(M.done_line(a.block, st["fail"] + 1, f"(guard stopped the queue before {s.label})"))
                return 1
            waited = 0
            while M.busy(exclude=(os.getpid(),)):
                if waited % 600 == 0:
                    say(f"# [{s.label}] waiting: another Isaac process runs {M.busy(exclude=(os.getpid(),))[:1]}")
                time.sleep(BUSY_POLL_S)
                waited += BUSY_POLL_S
        log = logd / f"{s.label}.log"
        t0 = time.time()
        rc = run_child([str(x) for x in argv_r], env, log)
        note = f"({time.time() - t0:.0f} s; log {log})"
        if s.slot and rc == 0:
            stamps = pinned_stamps(M.slot_dir(*s.slot), pin)     # not ``st``: that name is the failure tally
            if not stamps:
                rc, note = 8, (f"(gate_row rc 0 but no complete stamp at the pinned build in {M.slot_dir(*s.slot)}; "
                               f"log {log})")
            else:
                note = f"(stamp {stamps[-1]}; {time.time() - t0:.0f} s)"
        say(M.sentinel(s.label, rc, note))
        if rc != 0:
            st["fail"] += 1
        if s.isaac:
            ddc1, _p = guards()
            if ddc0 is not None and ddc1 - ddc0 > M.DDC_GROWTH_MAX_GB:
                say(M.done_line(a.block, st["fail"] + 1, f"(DerivedDataCache grew {ddc1 - ddc0:.0f} GB in {s.label}: "
                                                   "quarantine it before any Kit start)"))
                return 1
        if rc != 0 and s.stop_on_fail:
            say(M.done_line(a.block, st["fail"], f"(stopped at {s.label})"))
            return 1
    say(M.done_line(a.block, st["fail"]))
    return 0 if not st["fail"] else 1


if __name__ == "__main__":
    sys.exit(main())
