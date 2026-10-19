#!/usr/bin/env python3
"""PLAN step 8 C7: the FULL-GRAPH DRY PASS -- drives ``TP/pipeline.py run --run dry`` end to end and judges it.

It does what the orchestrator would do at every prompt, by a pre-registered table, and records each action:

* prepare: ``gen --run dry``; REDUCED mode (default) also ``lock`` (the reduced spec's own lock, in its own dir);
  OFFICIAL mode never re-locks PL/pipeline.lock.json (the orchestrator re-locks before C7: the step-8 scripts in
  PL/scripts/ are lock inputs). Then ``<run dir>/INJECT.json`` from ``SC/INJECT.json`` (c4 FAIL, c5 INVALID, c6 GUARD,
  st LIVE, sn LIVE).
* launch ``pipeline.py run``; while the queue runs, sample ``ps`` for the one-Isaac-process rule: the count is of KIT
  processes (``tp_lib.kit_lines``: anchored ISAAC_PAT minus NONKIT_PAT minus the driver supervisor -- a driver stage
  is ``run_candidates.py`` + its one train / eval child, and counting both read every driver stage as two: the
  2026-10-17 C7 "2 at 16:27:09" was s1's driver + its own train.py, Kit log kit_20261017_162700); the lines at the
  max are recorded; ONCE, while a driver stage after s1 is ``training`` with its unit active, SIGTERM the QUEUE
  process (a plain python, never Kit -- the driver unit keeps training) and relaunch ``run``: the resumed queue must
  ATTACH to the running driver (``T4PIPE ATTACH``), not relaunch it.
* when the queue parks: every open HALT gets its table action -- FAIL / LIVE -> ``continue --accept-chance``; the
  injected GUARD (c6) -> ``continue --retry``; the STUDENT HALT (sel: a 50-it student has one candidate and fails its
  block) is EXPECTED and left open (the stand branch must still run); anything else (CRASH, INFRA, DRIVER, STOP-R, an
  INVALID HALT = Gate 0 refusing real stamps twice, a GUARD that was not injected) ends the pass as FAIL.
* verdict (STEPC_RULES C7, pinned): ``SC/c7/dry_verdict.json`` -- PASS iff every check below holds.

    t4_stepC_dry.py [--mode reduced|official] [--poll 60] [--max-hours 30]
Never SIGKILLs anything; the only signal it sends is SIGTERM to the pipeline QUEUE's python (pid verified by the
pipeline's own pidfile + /proc cmdline check).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t4_stepC_lib as C  # noqa: E402

OUTD = C.SC / "c7"
EXPECT = {"s1": ("accepted", "pass"), "c4": ("accepted", "pass"), "c5": ("accepted", "pass"),
          "c6": ("accepted", "pass"), "c6x": ("accepted", "pass"), "p2a": ("accepted", "pass"),
          "p3a": ("accepted", "pass"), "p3b": ("ceiling", "accepted", "pass", "skipped"),
          "p3c": ("skipped", "ceiling", "accepted", "pass"), "tstar": ("pass",), "st": ("pass",),
          "sel": ("halt", "pass"), "sn": ("accepted", "pass"), "sh": ("accepted", "pass"),
          "sn_ref": ("pass",), "sh_ref": ("pass",), "bundle": ("blocked", "pass")}
FORBIDDEN = ("CRASH", "INFRA", "DRIVER", "STOP-R")


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log(rec: dict) -> None:
    OUTD.mkdir(parents=True, exist_ok=True)
    with open(OUTD / "actions.jsonl", "a") as fh:
        fh.write(json.dumps({"at": now(), **rec}, default=str) + "\n")
    print(f"[c7] {rec}", flush=True)


def pl(spec: Path, *args) -> tuple:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("KRABBY_", "SN_WALKER_", "P2_", "P2R_", "SH_", "SN_STAND_", "TP_")) or k == "TP_NOTIFY_CMD"}
    p = subprocess.run([C.PY, "-I", str(C.TP / "pipeline.py"), *args, "--spec", str(spec)], capture_output=True,
                       text=True, env=env)
    return p.returncode, (p.stdout + p.stderr)[-3000:]


def progress(update=None) -> dict:
    p = OUTD / "progress.json"
    d = C.read_json(p, {}) or {}
    if update:
        d.update(update)
        C.write_json(p, d)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=("reduced", "official"), default="reduced")
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--max-hours", type=float, default=30.0)
    a = ap.parse_args(argv)
    spec = (C.SC / "dryspec/pipeline.yaml") if a.mode == "reduced" else C.SPEC
    if not spec.is_file():
        print(f"no spec {spec} (run scripts/t4_make_stepC.py)", file=sys.stderr)
        return 2
    P = C.P()
    S = P.Spec(spec)
    rd = S.run_dir("dry")
    prog = progress({"mode": a.mode, "spec": str(spec), "run_dir": str(rd)})
    if not prog.get("prepared"):
        rc, out = pl(spec, "gen", "--run", "dry")
        log({"step": "gen", "rc": rc, "out": out[-400:]})
        if rc != 0:
            return finish(S, a, fail=[f"gen rc {rc}"])
        if a.mode == "reduced" and not S.lock_path().is_file():
            rc, out = pl(spec, "lock")
            log({"step": "lock", "rc": rc, "out": out[-400:]})
            if rc != 0:
                return finish(S, a, fail=[f"lock rc {rc}"])
        inj = rd / "INJECT.json"
        if not inj.is_file():
            inj.write_text((C.SC / "INJECT.json").read_text())
        progress({"prepared": now()})
    t_end = time.time() + a.max_hours * 3600
    fails = []
    while time.time() < t_end:
        ctx = P.Ctx(S, "dry")
        st = ctx.state
        qpid = P.queue_pid(ctx)
        prog = progress()
        if qpid:
            kl = C.TL().kit_lines()
            n = len(kl)
            if n > int(prog.get("max_isaac", 0)):
                progress({"max_isaac": n, "max_isaac_at": now(), "max_isaac_lines": [ln[:300] for ln in kl],
                          "isaac_lines_at_max": [ln[:300] for ln in C.isaac_lines()]})
            if not prog.get("sigterm_done"):
                for sid in ("c5", "c6x", "c6"):
                    r = (st.get("stages") or {}).get(sid) or {}
                    if r.get("status") == "training" and r.get("unit") and ctx.rt.unit_active(r["unit"]):
                        os.kill(qpid, signal.SIGTERM)
                        for _ in range(120):
                            if P.queue_pid(P.Ctx(S, "dry")) is None:
                                break
                            time.sleep(2)
                        alive = ctx.rt.unit_active(r["unit"])
                        log({"step": "sigterm_queue", "stage": sid, "pid": qpid, "unit": r["unit"],
                             "unit_still_active": alive})
                        if alive:
                            progress({"sigterm_done": now(), "sigterm_stage": sid,
                                      "qlog_offset": (ctx.qlog.stat().st_size if ctx.qlog.is_file() else 0)})
                        else:   # the driver finished inside the window: no attach to test -- try a later stage
                            progress({"sigterm_races": int(prog.get("sigterm_races", 0)) + 1})
                        rc, out = pl(spec, "run", "--run", "dry")
                        log({"step": "relaunch_after_sigterm", "rc": rc, "out": out[-300:]})
                        if rc != 0:
                            fails.append(f"relaunch after SIGTERM refused: {out[-200:]}")
                        break
            time.sleep(a.poll)
            continue
        status = st.get("status")
        if status in ("new", None) and not prog.get("launched"):
            rc, out = pl(spec, "run", "--run", "dry")
            log({"step": "run", "rc": rc, "out": out[-400:]})
            progress({"launched": now()})
            if rc != 0:
                fails.append(f"pipeline.py run refused: {out[-300:]}")
                break
            time.sleep(10)
            continue
        if status == "running":
            # the queue died without parking (outside the SIGTERM test): relaunch, at most 3 times in the pass
            nd = int(prog.get("dead_relaunches", 0)) + 1
            progress({"dead_relaunches": nd})
            if nd > 3:
                fails.append("the queue died without parking more than 3 times (see pipeline_queue.log)")
                break
            rc, out = pl(spec, "run", "--run", "dry")
            log({"step": "relaunch_dead_queue", "rc": rc, "out": out[-300:]})
            if rc != 0:
                fails.append(f"relaunch of a dead queue refused: {out[-200:]}")
                break
            time.sleep(10)
            continue
        opened = {s: h for s, h in (st.get("halts") or {}).items() if h.get("status") == "open"}
        act = None
        for sid, h in sorted(opened.items()):
            cls = h["class"]
            if cls in ("FAIL", "LIVE"):
                act = (sid, "--accept-chance")
            elif cls == "GUARD" and sid == "c6" and "c6:GUARD" in (st.get("injected") or []) \
                    and not prog.get("guard_retried"):
                act = (sid, "--retry")
                progress({"guard_retried": now()})
            elif cls == "STUDENT":
                continue
            else:
                fails.append(f"unexpected HALT {sid} {cls}: {(C.read_json(C.R / h['record'], {}) or {}).get('why', '')[:300]}")
            if act or fails:
                break
        if fails:
            break
        if act:
            sid, flag = act
            rc, out = pl(spec, "continue", "--run", "dry", flag, sid, "--why", "C7 dry pass: pre-registered action")
            log({"step": "continue", "stage": sid, "flag": flag, "rc": rc, "out": out[-300:]})
            if rc != 0:
                fails.append(f"continue {flag} {sid} rc {rc}: {out[-200:]}")
                break
            time.sleep(10)
            continue
        break                                       # parked with only the expected STUDENT HALT, or done
    else:
        fails.append(f"time bound {a.max_hours} h")
    return finish(S, a, fail=fails)


def finish(S, a, fail: list) -> int:
    P = C.P()
    ctx = P.Ctx(S, "dry")
    st = ctx.state
    prog = progress()
    checks = {}
    stages = {s["id"]: ((st.get("stages") or {}).get(s["id"]) or {}).get("status", "pending") for s in S.stages("dry")}
    checks["terminal_statuses"] = {s: {"status": v, "ok": v in EXPECT.get(s, (v,))} for s, v in stages.items()}
    inj = set(st.get("injected") or [])
    want = {f"{s}:{k}" for s, ks in (C.read_json(ctx.dir / "INJECT.json", {}) or {}).items() for k in ks}
    checks["injections_fired"] = {"want": sorted(want), "fired": sorted(inj), "ok": want <= inj}
    halts = sorted((ctx.dir / "halts").glob("*.json")) if (ctx.dir / "halts").is_dir() else []
    hrec = [(p.name, C.read_json(p, {}) or {}) for p in halts]
    notify = (ctx.notify_path.read_text() if ctx.notify_path.is_file() else "")
    qlog = (ctx.qlog.read_text() if ctx.qlog.is_file() else "")
    eff = {
        "c4_fail_line": any(n.startswith("c4.") and r.get("class") == "FAIL"
                            and any(x.get("id") == "c4.injected" for x in r.get("lines") or []) for n, r in hrec),
        "c5_invalid_rerun": "stage=c5 INVALID once" in notify,
        "c6_guard_halt": any(n.startswith("c6.") and r.get("class") == "GUARD" for n, r in hrec),
        "st_live_then_fresh_run": any(n.startswith("st.") and r.get("class") == "LIVE" for n, r in hrec)
                                  and int(((st.get("stages") or {}).get("st") or {}).get("distill_attempt", 1)) >= 2,
        "sn_live_then_fresh_run": any(n.startswith("sn.") and r.get("class") == "LIVE" for n, r in hrec)
                                  and int(((st.get("stages") or {}).get("sn") or {}).get("dagger_attempt", 1)) >= 2,
    }
    checks["injection_effects"] = {**eff, "ok": all(eff.values())}
    bad_h = [(n, r.get("class")) for n, r in hrec if r.get("class") in FORBIDDEN or r.get("class") == "INVALID"
             or (r.get("class") == "GUARD" and not n.startswith("c6."))]
    checks["no_forbidden_halts"] = {"found": bad_h, "ok": not bad_h}
    after = qlog[int(prog.get("qlog_offset", 0)):] if prog.get("sigterm_done") else ""
    sid = prog.get("sigterm_stage")
    checks["sigterm_resume_attach"] = {"done": bool(prog.get("sigterm_done")), "stage": sid,
                                       "ok": bool(prog.get("sigterm_done")) and bool(re.search(rf"ATTACH run=dry stage={sid}\b", after))}
    checks["one_isaac_at_a_time"] = {"max_seen": int(prog.get("max_isaac", 0)), "ok": int(prog.get("max_isaac", 0)) <= 1}
    drv = [s["id"] for s in S.driver_stages("dry")]
    gates = {s: (ctx.dir / "gates" / f"{s}.json").is_file() for s in drv if stages.get(s) not in ("skipped",)}
    checks["gate_records"] = {"present": gates, "ok": all(gates.values())}
    cand = C.TL().load_yaml(ctx.dir / "candidates.yaml") if (ctx.dir / "candidates.yaml").is_file() else {}
    chain_ok, prev = True, None
    for s in drv:
        cid = S.chain("dry", s)
        stg = ((cand.get("chains") or {}).get(cid) or {}).get("stages") or [{}]
        rf = stg[0].get("resume_from")
        exp = "scratch" if prev is None else {"reference": S.chain("dry", prev)}
        chain_ok = chain_ok and rf == exp
        prev = s
    checks["gen_chaining"] = {"ok": chain_ok}
    # terminal_statuses is a per-stage map with no top-level "ok" (CHARTER 2026-10-19: the C7 re-run read FAIL on
    # this alone, every check having passed); it is judged stage by stage below
    ok = not fail and all(v.get("ok") for k, v in checks.items() if k != "terminal_statuses")
    for k, v in checks["terminal_statuses"].items():
        ok = ok and v["ok"]
    v = {"verdict": "PASS" if ok else "FAIL", "mode": a.mode, "spec": str(S.path), "run_dir": str(ctx.dir),
         "driver_failures": fail, "checks": checks, "pipeline_status": st.get("status"),
         "open_halts": {s: h for s, h in (st.get("halts") or {}).items() if h.get("status") == "open"},
         "written_at": now()}
    C.write_json(OUTD / "dry_verdict.json", v)
    print(f"[c7] verdict {v['verdict']}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
