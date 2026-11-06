#!/usr/bin/env python3
"""DR stress sweep driver (CHARTER §3.4): one serial queue of gate rows, phase by phase, decide-record-notify.

Phases (each row is ``tools/pipeline/gate_row.sh`` of this checkout; one Isaac process at a time, gate_row waits):

1. ``nominal``  0 kg es1 + es2 twice (launches A, B) and L090 es1 + es2 (A); then the BASELINE GATE
                (``stress_read.baseline``). A failed gate writes ``DRS STOP baseline`` and exits 3.
2. ``grid``     every cell at es1 (table order).
3. ``es2``      the cells ``stress_read.es2_plan`` names (band edges, in-band es1 failures, terms 13-15 transitions).
4. ``closing``  0 kg es1 once more (launch Z, then Z2, Z3 ... whenever rows ran after the last closing): its 18 line
                values must equal launch A's or B's es1 (the bitwise outcome classes); anything else is FLAGGED
                (``DRS FLAG``, ``results/closing.json`` with its status), never a stop.
5. ``table``    ``results/table.json`` + ``results/table.md``; ``DRS END``.

Preconditions (non-dry), checked at start AND before every row: the code / eval paths of this checkout are CLEAN (the
gate_row ``code_state`` test) and their hashes equal the launch pin ``results/launch.json`` (written at the first
launch); a change stops the queue (``DRS STOP dirty_tree`` / ``code_changed_since_launch``). Starting ``--from`` a later
phase needs a passed baseline (``results/baseline.json``). One driver at a time (``results/run_sweep.lock``). A row is
skipped only when its newest complete stamp passes the reader's launch checks; a stamp that fails them is re-run
(``DRS RERUN``). ``touch STOP_QUEUE`` in the campaign dir, or SIGTERM / SIGINT / SIGHUP to the driver, stops before
the next row (the running row finishes). Every exit writes a sentinel to ``results/run_sweep.log``: ``DRS PHASE``,
``DRS ROW``, ``DRS BASELINE``, ``DRS ES2PLAN``, ``DRS CLOSING``, ``DRS FLAG``, ``DRS STOP <why>``, ``DRS END``.

Run with the Isaac venv python (the reader needs numpy):
    /home/nickmagus/krabby/isaac_venv/bin/python -I run_sweep.py [--from nominal|grid|es2|closing|table] [--dry]
    ... run_sweep.py --prepare-only     writes envs/, levels/ and the zero-depth checkpoint copy, then exits
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cells as C  # noqa: E402
import stress_read as SR  # noqa: E402

RESULTS = SR.RESULTS
LOG = RESULTS / "run_sweep.log"
ROWLOGS = C.TP_OUT / "rowlogs"
STOP_FILE = C.CAMPAIGN / "STOP_QUEUE"
PHASES = ("nominal", "grid", "es2", "closing", "table")
ROW_TIMEOUT_S = 5400
PY = "/home/nickmagus/krabby/isaac_venv/bin/python"
TOOLS = C.EXPERIMENTS / "tools"
INVOCATION = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
STOP_REQUESTED: list[str] = []


def _on_signal(signum, _frame) -> None:
    """Stop before the next row; the running row finishes (no orphaned Kit, no partial stamp)."""
    STOP_REQUESTED.append(signal.Signals(signum).name)


class Stop(Exception):
    def __init__(self, why: str, code: int):
        super().__init__(why)
        self.why, self.code = why, code


def say(msg: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    line = f"{msg} {datetime.datetime.now().isoformat(timespec='seconds')}"
    with LOG.open("a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def code_state() -> dict:
    """The gate_row ``code_state`` of this checkout (code_sha, eval_sha, dirty, dirty_paths)."""
    snippet = (
        "import json, subprocess, sys\n"
        "sys.dont_write_bytecode = True\n"
        "sys.path.insert(0, '.')\n"
        "import run_candidates as r\n"
        "st = subprocess.run(['git', '-C', str(r.REPO), 'status', '--porcelain', '--', *r.CODE_PATHSPECS, "
        "*r.EVAL_PATHSPECS], capture_output=True, text=True).stdout.split('\\n')\n"
        "dirty = [l[3:] for l in st if l.strip() and not r.code_drop(l[3:])]\n"
        "print(json.dumps({'code_sha': r.code_sha(), 'eval_sha': r.eval_sha(), 'dirty': dirty}))\n"
    )
    out = subprocess.run([PY, "-I", "-c", snippet], cwd=str(TOOLS), capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def check_tree() -> dict:
    cs = code_state()
    if cs["dirty"]:
        raise Stop(f"dirty_tree {cs['dirty']}", 5)
    pin = SR.launch_pin()
    if pin is None:
        pin = {"code_sha": cs["code_sha"], "eval_sha": cs["eval_sha"],
               "pinned_at": datetime.datetime.now().isoformat(timespec="seconds")}
        SR.LAUNCH.write_text(json.dumps(pin, indent=1))
        say(f"DRS PIN code_sha={pin['code_sha']} eval_sha={pin['eval_sha']}")
    elif (pin["code_sha"], pin["eval_sha"]) != (cs["code_sha"], cs["eval_sha"]):
        raise Stop(f"code_changed_since_launch pin={pin} now={cs}", 5)
    return pin


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare() -> None:
    """Per-cell env files (both rows), the payload level files, the zero-depth checkpoint copy (verified or rebuilt)."""
    (C.CAMPAIGN / "envs").mkdir(parents=True, exist_ok=True)
    (C.CAMPAIGN / "levels").mkdir(parents=True, exist_ok=True)
    for base in ("L000", "L090"):
        for la in ("A", "B", "Z"):
            for row in C.ROWS:
                C.env_file(C.nominal_id(base, la), row).write_text(C.env_text(None, None, row))
    for cell in C.cells():
        t = C.TERM[cell["term"]]
        for row in C.ROWS:
            C.env_file(cell["id"], row).write_text(C.env_text(t, cell["level"], row))
        if t["kind"] in ("payload_kg", "payload_com"):
            C.level_file_for(t, cell["level"], cell["base"]).write_text(C.level_text(t, cell["level"]))
    zd = C.ZERO_DEPTH_DIR
    agent_src = (C.CKPT.parent / "params" / "agent.yaml").read_text()
    if agent_src.count("zero_depth: false") != 1:
        raise RuntimeError("agent.yaml: expected exactly one 'zero_depth: false' to flip")
    want_agent = agent_src.replace("zero_depth: false", "zero_depth: true")
    ck = zd / C.CKPT.name
    env_src = (C.CKPT.parent / "params" / "env.yaml").read_bytes()
    ok = (ck.is_file() and not ck.is_symlink() and _sha256(ck) == _sha256(C.CKPT)
          and (zd / "params" / "env.yaml").is_file() and (zd / "params" / "env.yaml").read_bytes() == env_src
          and (zd / "params" / "agent.yaml").is_file() and (zd / "params" / "agent.yaml").read_text() == want_agent)
    if not ok:
        (zd / "params").mkdir(parents=True, exist_ok=True)
        if ck.exists() or ck.is_symlink():
            ck.unlink()
        shutil.copy2(C.CKPT, ck)
        shutil.copy2(C.CKPT.parent / "params" / "env.yaml", zd / "params" / "env.yaml")
        (zd / "params" / "agent.yaml").write_text(want_agent)


def row_problems(row: dict, term, level, base: str) -> list[str] | None:
    """None when the row has no complete stamp; else the reader's launch checks on its newest stamp ([] = valid)."""
    stamp = C.stamp_dir(row["name"], row["scenario"], row["es"])
    if stamp is None:
        return None
    want_zd = (term is not None and term["kind"] == "zero_depth") if row["policy"] == "student" else None
    return SR.stamp_problems(stamp, C.expected_env(term, level, base), want_zd, SR.launch_pin())


def run_row(row: dict, dry: bool, term=None, level=None, base: str = "L000", attempt: int = 1) -> int:
    if not dry:
        probs = row_problems(row, term, level, base)
        if probs == []:
            say(f"DRS SKIP {row['name']} {row['scenario']} es{row['es']} (valid stamp exists)")
            return 0
        if probs:
            say(f"DRS RERUN {row['name']} {row['scenario']} es{row['es']} (stamp fails the launch checks: {probs})")
    if STOP_FILE.exists():
        raise Stop("user (STOP_QUEUE)", 4)
    if STOP_REQUESTED:
        raise Stop(f"signal {STOP_REQUESTED[0]}", 4)
    if not dry:
        check_tree()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "TP_"))}
    env.update({"TP_OUT": str(C.TP_OUT), "TP_ROW_TIMEOUT": str(ROW_TIMEOUT_S)})
    if row["level_file"]:
        env["TP_LEVEL_FILE"] = row["level_file"]
    if dry:
        env["TP_ROW_DRY"] = "1"
    cmd = ["bash", str(C.GATE_ROW), row["name"], row["checkpoint"], row["env_file"], row["manifest"],
           row["scenario"], str(row["es"]), *row["args"]]
    ROWLOGS.mkdir(parents=True, exist_ok=True)
    log = ROWLOGS / f"{row['name']}.{row['scenario']}.es{row['es']}.{INVOCATION}.try{attempt}.log"
    with log.open("w") as fh:
        rc = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    stamp = C.stamp_dir(row["name"], row["scenario"], row["es"])
    say(f"DRS ROW {row['name']} {row['scenario']} es{row['es']} try{attempt} rc={rc} stamp={stamp}")
    return rc


def run_cell(cid: str, term, level, base: str, es: int, dry: bool) -> None:
    if term is None:  # nominal launches (A, B, Z, Z2 ...): their env files are written on demand
        for row in C.ROWS:
            if not C.env_file(cid, row).is_file():
                C.env_file(cid, row).write_text(C.env_text(None, None, row))
    for row in C.rows_for(cid, term, level, base, es):
        rc = run_row(row, dry, term, level, base)
        if rc != 0 and not dry:
            run_row(row, dry, term, level, base, attempt=2)  # one retry (Task 4's rule); a second failure: unread


def nominal(dry: bool) -> None:
    for base, launches in (("L000", ("A", "B")), ("L090", ("A",))):
        for la in launches:
            for es in (1, 2):
                run_cell(C.nominal_id(base, la), None, None, base, es, dry)


CLOSING_LAUNCHES = ("Z",) + tuple(f"Z{i}" for i in range(2, 10))


def _stamp_time(name: str, es: int) -> float | None:
    """The newest tp_row.json mtime of a cell-seed's two rows (None while either is missing)."""
    ts = []
    for spec in C.ROWS.values():
        st = C.stamp_dir(name, spec["scenario"], es)
        if st is None:
            return None
        ts.append((st / "tp_row.json").stat().st_mtime)
    return max(ts)


def newest_sweep_time() -> float:
    """The newest stamp time of every non-closing sweep cell-seed (nominal A/B/L090 and the grid)."""
    names = [(C.gate_name(C.nominal_id(b, la)), es) for b, las in (("L000", ("A", "B")), ("L090", ("A",)))
             for la in las for es in (1, 2)]
    names += [(C.gate_name(c["id"]), es) for c in C.cells() for es in (1, 2)]
    ts = [t for t in (_stamp_time(n, es) for n, es in names) if t is not None]
    return max(ts) if ts else 0.0


def _closing_valid(la: str) -> bool:
    """Both rows of a closing launch are stamped and pass the reader's launch checks."""
    cid = C.nominal_id("L000", la)
    return all(row_problems(row, None, None, "L000") == [] for row in C.rows_for(cid, None, None, "L000", 1))


def latest_closing() -> tuple[str | None, float | None]:
    """The most recent VALID closing launch (both rows stamped and passing the launch checks), and its stamp time."""
    best = (None, None)
    for la in CLOSING_LAUNCHES:
        t = _stamp_time(C.gate_name(C.nominal_id("L000", la)), 1)
        if t is not None and _closing_valid(la) and (best[1] is None or t > best[1]):
            best = (la, t)
    return best


def next_closing_launch() -> str | None:
    """The closing launch to run now: none while the latest closing is newer than every sweep stamp."""
    la, t = latest_closing()
    if la is not None and t >= newest_sweep_time():
        return None
    used = [x for x in CLOSING_LAUNCHES if _stamp_time(C.gate_name(C.nominal_id("L000", x)), 1) is not None]
    free = [x for x in CLOSING_LAUNCHES if x not in used]
    if not free:
        raise Stop("closing launches exhausted (Z..Z9)", 8)
    return free[0]


def closing_check() -> dict:
    la, t = latest_closing()
    z = None if la is None else SR.cellseed(C.gate_name(C.nominal_id("L000", la)), 1, None, None, "L000")
    refs = {x: SR.cellseed(C.gate_name(C.nominal_id("L000", x)), 1, None, None, "L000") for x in ("A", "B")}

    def vec(r):
        return None if r is None else {k: v["value"] for k, v in sorted(r["lines"].items())}
    zv = vec(z)
    matches = [x for x, r in refs.items() if r is not None and r["pass"] is not None and zv is not None
               and vec(r) == zv]
    if z is None:
        status = "unread"
    elif z["pass"] is None:
        status = "launch_error"
    elif matches:
        status = "match"
    else:
        status = "mismatch"
    newest = newest_sweep_time()
    res = {"status": status, "match": status == "match", "matches": matches, "launch": la, "stamp_time": t,
           "newest_sweep_time": newest, "stale": bool(t is None or t < newest),
           "z_pass": None if z is None else z["pass"], "z_launch_error": None if z is None else z["launch_error"],
           "z_lines": zv, "ref_lines": {x: vec(r) for x, r in refs.items()}}
    (RESULTS / "closing.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def sweep(a) -> int:
    todo = PHASES[PHASES.index(a.start):]
    say(f"DRS START phases={','.join(todo)} dry={a.dry} out={C.TP_OUT}")
    if not a.dry:
        pin = check_tree()
        say(f"DRS TREE clean code_sha={pin['code_sha']} eval_sha={pin['eval_sha']}")
        if "nominal" not in todo:
            bl = RESULTS / "baseline.json"
            if not bl.is_file() or json.loads(bl.read_text()).get("stop", True):
                raise Stop("baseline_not_passed (start from nominal)", 3)
    if "nominal" in todo:
        say("DRS PHASE nominal")
        nominal(a.dry)
        if not a.dry:
            res = SR.baseline()
            (RESULTS / "baseline.json").write_text(json.dumps(res, indent=1, default=str))
            say(f"DRS BASELINE stop={res['stop']} split={res['split']} reasons={res['reasons']}")
            if res["stop"]:
                raise Stop("baseline", 3)
    if "grid" in todo:
        say("DRS PHASE grid")
        for cell in C.cells():
            run_cell(cell["id"], C.TERM[cell["term"]], cell["level"], cell["base"], 1, a.dry)
    if "es2" in todo:
        say("DRS PHASE es2")
        plan = (SR.es2_plan(SR.grid_reads()) if not a.dry
                else [c["id"] for c in C.cells() if c["level"] in C.band_edges(C.TERM[c["term"]])])
        (RESULTS / "es2_plan.json").write_text(json.dumps(plan, indent=1))
        say(f"DRS ES2PLAN n={len(plan)}")
        by_id = {c["id"]: c for c in C.cells()}
        for cid in plan:
            cell = by_id[cid]
            run_cell(cid, C.TERM[cell["term"]], cell["level"], cell["base"], 2, a.dry)
    if "closing" in todo:
        say("DRS PHASE closing")
        la = "Z" if a.dry else next_closing_launch()
        if la is not None:
            run_cell(C.nominal_id("L000", la), None, None, "L000", 1, a.dry)
        if not a.dry:
            res = closing_check()
            say(f"DRS CLOSING status={res['status']} launch={res['launch']} matches={res['matches']} "
                f"stale={res['stale']}")
            if res["status"] != "match" or res["stale"]:
                say(f"DRS FLAG closing_{res['status']}{'_stale' if res['stale'] else ''} "
                    "(every cell since the last nominal is flagged in the table)")
    if "table" in todo and not a.dry:
        reads = SR.grid_reads()
        table = SR.term_table(reads)
        (RESULTS / "table.json").write_text(json.dumps({**table, "reads": reads}, indent=1, default=str))
        SR.write_md(table, RESULTS / "table.md")
        say(f"DRS TABLE rows={len(table['rows'])} closing_flag={table['closing_flag']}")
    say("DRS END")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=PHASES, default="nominal")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--prepare-only", action="store_true")
    a = ap.parse_args()
    import numpy  # noqa: F401,PLC0415 -- fail fast: the reader needs it (run with the Isaac venv python)

    RESULTS.mkdir(parents=True, exist_ok=True)
    lock = open(RESULTS / "run_sweep.lock", "w")  # noqa: SIM115 -- held for the process lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("DRS another driver holds results/run_sweep.lock", flush=True)
        return 6
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    try:
        prepare()
        if a.prepare_only:
            say("DRS PREPARED")
            return 0
        return sweep(a)
    except Stop as s:
        say(f"DRS STOP {s.why}")
        return s.code
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 -- every exit leaves a sentinel
        say(f"DRS STOP crash {type(exc).__name__}: {exc}")
        (RESULTS / "crash.txt").write_text(traceback.format_exc())
        return 7


if __name__ == "__main__":
    sys.exit(main())
