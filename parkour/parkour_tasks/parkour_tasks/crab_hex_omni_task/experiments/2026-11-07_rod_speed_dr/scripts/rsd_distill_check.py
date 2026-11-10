#!/usr/bin/env python3
"""Rod-speed DR campaign: the distill's +10-iteration checks (CHARTER 3.4), its record, and its stop -- CPU only.

    PY=/home/nickmagus/krabby/isaac_venv/bin/python
    $PY -I -B rsd_distill_check.py check --name N --teacher TDR_ABS [--dr 0.6:1.35|none] [--num-envs 192] [--seed 19]
                                         [--log LOG] [--run-dir DIR] [--teacher-run-dir DIR] [--min-iters 10] [--live]
    $PY -I -B rsd_distill_check.py reference      # the same checks on Task 4's T4F19_st_a2 (no DR), a self-test
    $PY -I -B rsd_distill_check.py record --name N --teacher TDR_ABS --env-file ENV   # results/runs/N.distill.json
    $PY -I -B rsd_distill_check.py stop --name N [--unit UNIT]   # SIGTERM / 120 s / SIGTERM / 120 s / SIGKILL last

``check`` (run it once the distill log shows iteration ``iter+10``; ``rsd_distill.sh`` prints the command):
  D1  the ``[KNOB]`` / ``[omni-knobs]`` multiset before the first iteration = ``T4F19_st_a2``'s (Task 4's distill of
      T*, same env file minus the knob) plus exactly the one ``KRABBY_DR_ROD_SPEED`` ``[KNOB]`` line
      (``dr_knobs.py``'s own print for the value; none with ``--dr none``);
  D2  the ``[DR]`` line = the predicted first draw (``rod_speed_dr.py``: seed ``rod_speed_dr_seed(seed)``, envs
      0..num_envs-1, step 0) naming this worktree's module; none with ``--dr none``;
  D3  teacher sha = TDR: the ``Loading model checkpoint from:`` path is TDR and its sha256 is TDR's; the run's
      ``params/agent.yaml`` ``load_checkpoint`` is TDR;
  D4  the first iteration is TDR's iteration and the log has reached +``min-iters``;
  D5  the live-watch tail read from TDR's run dir (``live_watch.read_series`` on its tfevents: last-200 means of
      ``Train/mean_episode_length`` and ``Episode_Termination/crab_failure``) exists -- the student live rules'
      reference;
  D6  (``--live``) ``live_watch.check("student", ...)`` on the student's tfevents with that tail and Task 4's
      ``pipeline.yaml`` live config (rc 3 = a live kill).
A miss means kill (CHARTER 3.4): exit 3, then ``stop``. Exit 2 = not yet at +min-iters (re-run later); 0 = pass.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import rsd_launch as RL  # noqa: E402

EXPERIMENT = "crab_hex_omni_student"
DEFAULT_DR = "0.6:1.35"
TASK4_PIPELINE_YAML = RL.OMNI / "experiments" / "2026-10-16_task4_pipeline" / "pipeline.yaml"
P3B_TEACHER_CKPT = RL.P3B_RUN_DIR / "model_25992.pt"
RC_PASS, RC_WAIT, RC_KILL = 0, 2, 3


def res(cid: str, ok, detail: str, **data) -> dict:
    return {"id": cid, "ok": ok, "detail": detail, "data": data}


def _head(text: str) -> str:
    m = RL.ITER_RE.search(text)
    return text[:m.start()] if m else text


def _read(path: Path, n: int = 64 << 20) -> str:
    with open(path, "rb") as fh:
        return fh.read(n).decode(errors="replace")


def live_cfg() -> dict:
    import yaml  # noqa: PLC0415

    return (yaml.safe_load(TASK4_PIPELINE_YAML.read_text()) or {}).get("live") or {}


def checks(*, name: str, teacher: Path, log: Path, run_dir: Path | None, dr: str | None, num_envs: int, seed: int,
           teacher_run_dir: Path | None, min_iters: int, reference_log: Path, live: bool) -> tuple[int, list[dict]]:
    out = []
    if not log.is_file():
        return RC_WAIT, [res("D0", None, f"no log {log} yet")]
    text = _read(log)
    head = _head(text)
    blocks = RL.iteration_blocks(text)
    m = re.match(r"^model_(\d+)\.pt$", teacher.name)
    t_iter = int(m.group(1)) if m else None
    # D4 first: before +min_iters the other reads may be incomplete
    if not blocks or blocks[-1][0] - blocks[0][0] < min_iters:
        last = blocks[-1][0] if blocks else None
        return RC_WAIT, [res("D4", None, f"log at iteration {last}; need {blocks[0][0] + min_iters if blocks else '?'}")]
    first = blocks[0][0]
    out.append(res("D4", first == t_iter, f"first iteration {first} (TDR iter {t_iter}); reached {blocks[-1][0]}",
                   first=first, last=blocks[-1][0]))
    # D1
    want = RL.knob_lines(_head(_read(reference_log)))
    spec = None
    if dr is not None:
        spec, line = RL.dr_knob_line(dr)
        want = want + [line]
    d = RL.multiset_diff(RL.knob_lines(head), want)
    ok = not d["missing"] and not d["extra"]
    out.append(res("D1", ok, f"{len(want)} [KNOB]/[omni-knobs] lines = {reference_log.name}'s" +
                   (" + the DR line" if dr else "") if ok else f"missing {d['missing']} extra {d['extra']}", **d))
    # D2
    got = RL.dr_lines(head)
    if dr is not None:
        p = RL.predicted_dr(spec, num_envs, seed, 0)
        ok = got == [p["line"]] and f"module={RL.RSD_PY}" in (got[0] if got else "")
        out.append(res("D2", ok, "the [DR] line = the predicted draw (worktree module)" if ok
                       else f"got {got}, want [{p['line']}]", got=got, predicted=p))
    else:
        out.append(res("D2", not got, "no [DR] line (no knob)" if not got else f"unexpected {got}", got=got))
    # D3
    loads = [ln.split("Loading model checkpoint from: ", 1)[1].strip() for ln in head.split("\n")
             if "Loading model checkpoint from: " in ln]
    t_sha = RL.sha256_file(teacher)
    why = []
    if loads != [str(teacher)]:
        why.append(f"loaded {loads}, want [{teacher}]")
    else:
        try:
            if RL.sha256_file(loads[0]) != t_sha:
                why.append("the loaded file's sha256 is not TDR's")
        except OSError as exc:
            why.append(f"cannot hash {loads[0]}: {exc}")
    lc = None
    if run_dir is not None and (run_dir / "params" / "agent.yaml").is_file():
        mm = re.search(r"^load_checkpoint:\s*(.+?)\s*$", (run_dir / "params" / "agent.yaml").read_text(), re.M)
        lc = mm.group(1) if mm else None
        if lc != str(teacher):
            why.append(f"params/agent.yaml load_checkpoint {lc} != TDR")
    else:
        why.append(f"no params/agent.yaml under run dir {run_dir}")
    out.append(res("D3", not why, "; ".join(why) or f"teacher sha {t_sha[:12]} = TDR ({teacher.name})",
                   teacher_sha256=t_sha, loaded=loads, load_checkpoint=lc))
    # D5
    tail = None
    trd = teacher_run_dir or teacher.parent
    try:
        LW = RL.load_by_path(RL.LIVE_WATCH_PY, "rsd_live_watch")
        tser = LW.read_series(trd, (LW.T_EPLEN, LW.T_CRAB))
        tail = {"ep_len": LW.tail_mean(tser.get(LW.T_EPLEN)), "crab_failure": LW.tail_mean(tser.get(LW.T_CRAB))}
        ok = tail["ep_len"] is not None and tail["crab_failure"] is not None
        out.append(res("D5", ok, f"TDR tail from {trd}: ep_len {tail['ep_len']}, crab_failure {tail['crab_failure']}"
                       if ok else f"no T* tail in {trd} (tfevents with {LW.T_EPLEN} / {LW.T_CRAB}?)",
                       teacher_run_dir=str(trd), tail=tail))
    except Exception as exc:  # noqa: BLE001
        out.append(res("D5", False, f"live-watch tail from {trd} failed: {exc}"))
    # D6
    if live:
        if run_dir is None or tail is None:
            out.append(res("D6", False, "no student run dir / no teacher tail"))
        else:
            ser = LW.read_series(run_dir)
            r = LW.check("student", ser, tstar_tail=tail, cfg=live_cfg())
            out.append(res("D6", r["rc"] != LW.RC_KILL, f"live rc {r['rc']} rule {r.get('rule')} fails "
                           f"{r.get('fails')} watch {r.get('watch')}", live=r))
    bad = [r for r in out if r["ok"] is False]
    return (RC_KILL if bad else RC_PASS), out


def cmd_check(a) -> int:
    teacher = Path(a.teacher)
    if not teacher.is_absolute() or not teacher.is_file():
        print(f"rsd_distill_check: --teacher must be an existing absolute path: {a.teacher}", file=sys.stderr)
        return RC_KILL
    log = Path(a.log) if a.log else RL.DISTILL_OUT / f"{a.name}.log"
    run_dir = Path(a.run_dir) if a.run_dir else RL.find_run_dir(a.name, a.experiment)
    dr = None if a.dr.lower() == "none" else a.dr
    rc, out = checks(name=a.name, teacher=teacher, log=log, run_dir=run_dir, dr=dr, num_envs=a.num_envs,
                     seed=a.seed, teacher_run_dir=Path(a.teacher_run_dir) if a.teacher_run_dir else None,
                     min_iters=a.min_iters, reference_log=Path(a.reference_log), live=a.live)
    for r in out:
        tag = "PASS" if r["ok"] else ("WAIT" if r["ok"] is None else "MISS")
        print(f"RSD DISTILL CHECK {a.name} {r['id']} {tag} {r['detail'][:700]}")
    verdict = {RC_PASS: "PASS", RC_WAIT: "WAIT (re-run later)", RC_KILL: "MISS -> kill (rsd_distill_check.py stop)"}
    print(f"RSD DISTILL CHECK {a.name} RESULT {verdict[rc]}")
    if a.json:
        print(json.dumps({"rc": rc, "checks": out}, indent=1, default=str))
    return rc


def cmd_reference(a) -> int:
    ns = argparse.Namespace(name=RL.ST_A2_NAME, teacher=str(P3B_TEACHER_CKPT), log=str(RL.ST_A2_LOG),
                            run_dir=str(RL.ST_A2_RUN_DIR), dr="none", num_envs=192, seed=19,
                            teacher_run_dir=str(RL.P3B_RUN_DIR), min_iters=10, reference_log=str(RL.ST_A2_LOG),
                            live=a.live, json=a.json, experiment=EXPERIMENT)
    return cmd_check(ns)


def cmd_record(a) -> int:
    teacher, envf = Path(a.teacher), Path(a.env_file)
    rec = {"name": a.name, "kind": "distill", "created": RL.now_iso(), "launcher_argv": a.launcher_argv,
           "teacher": str(teacher), "teacher_sha256": RL.sha256_file(teacher),
           "teacher_facts": {k: v for k, v in RL.checkpoint_facts(teacher).items() if k != "keys"},
           "env_file": str(envf), "env_file_sha256": RL.sha256_file(envf),
           "env_file_krabby": dict(RL.read_env_lines(envf)[0]), "head": RL.git_head(),
           "log": str(RL.DISTILL_OUT / f"{a.name}.log"),
           "file_hashes": {str(p): RL.sha256_file(p) for p in (RL.DISTILL_SH, RL.TRAIN, RL.RSD_PY, RL.DR_KNOBS_PY,
                                                               RL.ACTIONS_PY, RL.OMNI_KNOBS_PY)},
           "checks_command": (f"{RL.PY} -I -B {Path(__file__).resolve()} check --name {a.name} "
                              f"--teacher {teacher} --dr {DEFAULT_DR}")}
    try:
        cs = RL.code_state()
        rec["code_sha"], rec["eval_sha"], rec["dirty"] = cs["code_sha"], cs["eval_sha"], cs["dirty"]
    except Exception as exc:  # noqa: BLE001
        rec["code_state_error"] = str(exc)
    path = RL.RUNS / f"{a.name}.distill.json"
    if path.exists():
        print(f"rsd_distill_check: {path} exists (a distill is never relaunched under the same name)", file=sys.stderr)
        return 5
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    print(f"recorded {path}")
    return 0


def cmd_stop(a) -> int:
    unit = a.unit
    if unit is None:
        uf = RL.DISTILL_OUT / f"{a.name}.unit"
        unit = uf.read_text().strip() if uf.is_file() else None
    r = RL.stop_trainer(lambda: RL.trainer_pids(unit, a.name), say=lambda m: print(f"RSD DISTILL STOP {a.name} {m}"))
    print(json.dumps(r, default=str))
    if r.get("sigkill"):
        print(f"RSD DISTILL STOP {a.name} sigkill=1: open the DerivedDataCache watch before the next Kit start")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--name", required=True)
    c.add_argument("--teacher", required=True)
    c.add_argument("--dr", default=DEFAULT_DR, help="the knob value in the env file, or 'none'")
    c.add_argument("--num-envs", type=int, default=192)
    c.add_argument("--seed", type=int, default=19)
    c.add_argument("--log", default=None)
    c.add_argument("--run-dir", default=None)
    c.add_argument("--teacher-run-dir", default=None)
    c.add_argument("--experiment", default=EXPERIMENT)
    c.add_argument("--min-iters", type=int, default=10)
    c.add_argument("--reference-log", default=str(RL.ST_A2_LOG))
    c.add_argument("--live", action="store_true")
    c.add_argument("--json", action="store_true")
    r = sub.add_parser("reference")
    r.add_argument("--live", action="store_true")
    r.add_argument("--json", action="store_true")
    rc_ = sub.add_parser("record")
    rc_.add_argument("--name", required=True)
    rc_.add_argument("--teacher", required=True)
    rc_.add_argument("--env-file", required=True)
    rc_.add_argument("--launcher-argv", nargs=argparse.REMAINDER, default=[])
    s = sub.add_parser("stop")
    s.add_argument("--name", required=True)
    s.add_argument("--unit", default=None)
    a = ap.parse_args(argv)
    return {"check": cmd_check, "reference": cmd_reference, "record": cmd_record, "stop": cmd_stop}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
