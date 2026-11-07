#!/usr/bin/env python3
"""Rod-speed DR campaign gate driver (CHARTER 3.3 G1, 3.4 selection + G2, 3.5 disk guard, 3.6 stops).

One serial queue of Task 4 gate rows (``tools/pipeline/gate_row.sh`` of THIS worktree, one Isaac process at a time:
gate_row waits while one runs), composed as the sweep's driver composed them: the parent's ``KRABBY_*`` / ``TP_*``
stripped, ``TP_OUT`` (``$RSD_TP_OUT``, default the main checkout's ``parkour/outputs/omni_plan/rod_speed_dr``),
``TP_ROW_TIMEOUT=5400``, ``TP_LEVEL_FILE`` on a level row, ``TP_STRIP_KEYS`` on a selection row (Task 4's). Before
EVERY row: STOP_QUEUE / a signal, a clean tree whose code_sha / eval_sha equal ``results/launch.json`` (``pin``), and
the free-space guard (stop below 30 GB free or with the Kit DerivedDataCache above 78 GB). A row is skipped only when
its newest stamp passes every launch check (rsd_lib.row_problems); otherwise it runs, with one retry; a second failure
stops the queue. Sentinels in ``results/rsd_gate.log``: ``RSD ROW|SKIP|RERUN|STOP|END ...`` (plus START / PIN / READ);
a stop also appends ``results/notify.jsonl``. One driver at a time (fcntl ``results/rsd_gate.lock``); ``touch
STOP_QUEUE`` in the campaign dir, or SIGTERM / SIGINT / SIGHUP, stops before the next row (the running row finishes).

    PY=/home/nickmagus/krabby/isaac_venv/bin/python;  G="$PY -B -I rsd_gate.py"
    $G policy add NAME --checkpoint ABS --role teacher|student      (refuses to rebind NAME to another sha256)
    $G policy list
    $G pin                                                          (results/launch.json, clean tree only)
    $G refarm [--dry]                       old T*: L000 / L090 / L135 nominal, x0.7 / x0.8 L090, es1+2 + 7 sel rows
    $G bridge [--policy TSTAR|C28000] [--relaunch] [--dry]          the code-path bridge to the sweep's stamps
    $G g1 --policy TDR [--bridge-class-flip] [--dry]   needs bridge_TSTAR.json PASS; TDR's 16 cell-seeds + its 7
                                            non-suite selection rows (x0.7 L090 first)
    $G tctl --policy TCTL [--bridge-class-flip] [--dry]   (bridge_TSTAR.json PASS) L090 nominal, x0.7 / x0.8 L090
    $G closing --policy NAME [--gate g1|g2] [--dry]                 L090 nominal es1 once more (launch Z)
    $G es3 CELL --policy NAME --gate g1|g2 [--dry]                  a near-line cell's es3 (<= 4 cell-seeds / gate)
    $G extend --policy TDR6 [--dry]         the post-EXTEND checkpoint's G1 rows (then: read g1 --extended)
    $G sel --student-run DIR --teacher TDR [--pick-name PICK] [--approve-extra-block] [--dry]
    $G g2ref [--dry]                        the in-pin c28000 reference arm (then: bridge --policy C28000)
    $G g2 --policy PICK [--bridge-class-flip] [--dry]   needs bridge_C28000.json PASS; the pick's rows, watch, closing
    $G g2relaunch CELL --policy PICK [--dry]                        G2's class-ambiguity relaunch (both sides)
    $G read g1 --tdr TDR [--old TSTAR] [--tctl TCTL] [--extended --prior-tdr TDR0] [--out JSON]
    $G read g2 --pick PICK [--ref C28000] [--tdr TDR] [--out JSON]
    $G replay [--no-dose] [--out JSON]      the CPU replay (step-8 precondition): sweep reads, Task 4 sel, F19 R
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rsd_lib as L  # noqa: E402

LOG = L.RESULTS / "rsd_gate.log"
NOTIFY = L.RESULTS / "notify.jsonl"
LOCK = L.RESULTS / "rsd_gate.lock"
STOP_FILE = L.CAMPAIGN / "STOP_QUEUE"
INVOCATION = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
STOP_REQUESTED: list[str] = []
#: CHARTER 3.6 budget table (GPU-h), for the selection's stop-and-ask above 35
BUDGET = {"main_total": 29.5, "extend_total": 34.3, "extra_block": 0.8, "cap": 35.0}


def _on_signal(signum, _frame) -> None:
    STOP_REQUESTED.append(signal.Signals(signum).name)


def say(msg: str) -> None:
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    line = f"{msg} {datetime.datetime.now().isoformat(timespec='seconds')}"
    with LOG.open("a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def notify(kind: str, text: str) -> None:
    L.RESULTS.mkdir(parents=True, exist_ok=True)
    with NOTIFY.open("a") as fh:
        fh.write(json.dumps({"at": L.now(), "kind": kind, "text": text}) + "\n")


# ============================================================================== the row runner
class Runner:
    def __init__(self, dry: bool, phase: str):
        self.dry, self.phase, self.n = dry, phase, 0
        self.stamps: dict = {}

    def preflight(self) -> None:
        if STOP_FILE.exists():
            raise L.Stop("user (STOP_QUEUE)", 4)
        if STOP_REQUESTED:
            raise L.Stop(f"signal {STOP_REQUESTED[0]}", 4)
        L.check_tree()
        bad = L.disk_problems()
        if bad:
            raise L.Stop("disk " + "; ".join(bad), 4)

    def launch(self, row: dict, attempt: int) -> int:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("KRABBY_", "TP_"))}
        env.update({"TP_OUT": str(L.TP_OUT), "TP_ROW_TIMEOUT": str(L.ROW_TIMEOUT_S)})
        if row["level_file"]:
            env["TP_LEVEL_FILE"] = row["level_file"]
        if row.get("strip"):
            env["TP_STRIP_KEYS"] = " ".join(row["strip"])
        cmd = ["bash", str(L.GATE_ROW), row["name"], row["ckpt"], row["env_file"], row["manifest"], row["scenario"],
               str(row["es"]), *row["args"]]
        logs = L.TP_OUT / "rowlogs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{row['name']}.{row['scenario']}.es{row['es']}.{INVOCATION}.try{attempt}.log"
        with log.open("w") as fh:
            return subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode

    def run(self, row: dict) -> Path | None:
        self.n += 1
        if row["kind"] == "cell":
            L.prepare_cell(row["cell"])
        if self.dry:
            print(f"RSD DRY {self.n:3d} {L.row_id(row)} role={row['role']} ckpt={Path(row['ckpt']).name} "
                  f"env={Path(row['env_file']).name} level={Path(row['level_file']).name if row['level_file'] else '-'}"
                  f" strip={len(row['strip'])} dose_s={row['rod_s']:g} args={' '.join(row['args'])}", flush=True)
            return None
        pin = L.launch_pin()
        st = L.row_stamp(row)
        probs = None if st is None else L.row_problems(st, row, pin)
        if probs == []:
            say(f"RSD SKIP {L.row_id(row)} (valid stamp {st.name})")
            self.stamps[row["key"]] = st
            return st
        for attempt in (1, 2):
            if probs:
                say(f"RSD RERUN {L.row_id(row)} (stamp fails the launch checks: {probs[:3]})")
            self.preflight()
            rc = self.launch(row, attempt)
            st = L.row_stamp(row)
            probs = ["no complete stamp"] if st is None else L.row_problems(st, row, L.launch_pin())
            say(f"RSD ROW {L.row_id(row)} try{attempt} rc={rc} stamp={st} ok={not probs}")
            if rc == 0 and not probs:
                self.stamps[row["key"]] = st
                return st
        raise L.Stop(f"row_failed_twice {L.row_id(row)}: {probs[:3]}", 4)

    def run_all(self, rows: list) -> list:
        return [self.run(r) for r in rows]


# ============================================================================== phase row lists
def cellseed_rows(pol: dict, cids: list, ess=(1, 2), suffix: str = "") -> list:
    out = []
    for cid in cids:
        for es in ess:
            out += L.cell_rows(pol, cid, es, suffix)
    return out


def refarm_rows(dry: bool = False) -> list:
    O = L.policy("TSTAR")
    return cellseed_rows(O, L.REF_CELLS) + [L.sel_row(O, k) for k in L.SEVEN]


def g1_rows(name: str, dry: bool = False) -> list:
    T = L.policy_for(name, "teacher", dry)
    if T["role"] != "teacher":
        raise L.Stop(f"{name} is a {T['role']} policy; G1 rows are teacher-role", 2)
    order = [L.X07, L.X08, L.X07_L000, L.X135_L000, L.X135_L090] + L.N_CELLS
    return cellseed_rows(T, order) + [L.sel_row(T, k) for k in L.SEVEN]


def tctl_rows(name: str, dry: bool = False) -> list:
    C = L.policy_for(name, "teacher", dry)
    return cellseed_rows(C, [L.X07, L.X08, L.NOM["L090"]])


def g2ref_rows() -> list:
    """CHARTER 3.4: the in-pin c28000 reference arm, run (and bridged) BEFORE the pick's rows."""
    return cellseed_rows(L.policy("C28000"), L.REF_CELLS)


def bridge_guard(a, ref: str, phase: str, how: str) -> None:
    """``phase`` runs its rows only after the ``ref`` bridge read PASS (or a STOP the user diagnosed as a class flip,
    recorded, passed as --bridge-class-flip): CHARTER 3.3's bridge rule (3.4 for c28000)."""
    b = L.read_json(L.RESULTS / f"bridge_{ref}.json", {}) or {}
    ok_sha = b.get("sha256") == L.policy(ref)["sha256"]
    v = b.get("verdict")
    if not ok_sha or not (v == "PASS" or (v == "STOP" and a.bridge_class_flip)):
        raise L.Stop(f"{phase}: results/bridge_{ref}.json verdict {v!r} (sha ok {ok_sha}); run {how} first (a STOP "
                     "needs the CPU diagnosis; a recorded class flip proceeds with --bridge-class-flip)", 3)


def g2_bridge_guard(a) -> None:
    """g2 runs the pick's rows only after the c28000 bridge (CHARTER 3.4)."""
    bridge_guard(a, "C28000", "g2", "`g2ref`, then `bridge --policy C28000 [--relaunch]`")


def g1_bridge_guard(a) -> None:
    """g1 / tctl / extend run a fine-tuned teacher's rows only after the old-T* bridge (CHARTER 3.3: the bridge runs
    before any training row is read against the in-pin references)."""
    bridge_guard(a, "TSTAR", a.cmd, "`refarm`, then `bridge [--relaunch]`")


def g2_rows(name: str, dry: bool = False) -> list:
    P, Rf = L.policy_for(name, "student", dry), L.policy("C28000")
    if P["role"] != "student":
        raise L.Stop(f"{name} is a {P['role']} policy; G2 rows are student-role", 2)
    return (cellseed_rows(Rf, L.REF_CELLS)
            + cellseed_rows(P, [L.X07, L.X08, L.X07_L000, L.X135_L000, L.X135_L090, L.NOM["L135"]])
            + cellseed_rows(P, L.G2_WATCH)
            + L.cell_rows(P, L.CLOSING_CELL, 1))


# ============================================================================== phases
def ph_bridge(a, R: Runner) -> int:
    pin = L.launch_pin()
    pol = L.policy(a.policy)
    if a.dry:
        plan = L.BRIDGE_TEACHER if pol["role"] == "teacher" else L.BRIDGE_STUDENT
        print(f"RSD DRY bridge {a.policy}: reads {len(plan)} cell-seeds against the sweep's "
              f"{'DRST' if pol['role'] == 'teacher' else 'DRS'} stamps; --relaunch runs their _RL rows:", flush=True)
        if a.relaunch:
            for cid, es in plan:
                R.run_all(L.cell_rows(pol, cid, es, L.RELAUNCH))
        return 0
    res = L.bridge_read(a.policy, pin=pin)
    if res["verdict"] == "RELAUNCH_DUE" and a.relaunch:
        for k in res["relaunch_due"]:
            cid, es = k.split("@es")
            R.run_all(L.cell_rows(pol, cid, int(es), L.RELAUNCH))
        res = L.bridge_read(a.policy, pin=pin)
    out = L.RESULTS / f"bridge_{a.policy}.json"
    L.write_json(out, res)
    say(f"RSD READ bridge {a.policy} verdict={res['verdict']} "
        f"{ {k: v['status'] for k, v in res['cell_seeds'].items()} }")
    if res["verdict"] == "STOP":
        raise L.Stop(f"bridge {a.policy}: {res['why']}", 3)
    return 0


def _gate_json(gate: str) -> dict:
    return L.read_json(L.RESULTS / f"{gate}.json", {}) or {}


def _read_is_for(g: dict, gate: str, key: str, P: dict, what: str) -> None:
    """A gate json authorises a launch only for the policy it was read for (a stale read of another policy, e.g. the
    pre-EXTEND TDR, authorises nothing)."""
    rec = ((g.get("policies") or {}).get(key) or {}).get("sha256")
    if rec != P["sha256"]:
        raise L.Stop(f"{what}: results/{gate}.json was read for {key} sha {str(rec)[:8]}, not {P['name']} "
                     f"({P['sha8']}); read {gate} again first", 3)


def _live_es3(g: dict, gate: str, P: dict) -> int:
    """The allowance spent NOW (es3 stamps on disk + the G2 relaunch charge), to refuse a stale read."""
    names = g.get("es3_policies") or [P["name"]]
    n = L.es3_used([P if nm == P["name"] else L.policy(nm) for nm in names], L.S_CELLS + L.N_CELLS,
                   pin=L.launch_pin())
    return n + (int(g.get("relaunch_charged") or 0) if gate == "g2" else 0)


def ph_es3(a, R: Runner) -> int:
    P = L.policy_for(a.policy, "student" if a.gate == "g2" else "teacher", a.dry)
    if a.cell not in L.S_CELLS + L.N_CELLS:
        raise L.Stop(f"es3: {a.cell} is not a gated cell", 2)
    if not a.dry:
        g = _gate_json(a.gate)
        if a.cell not in (g.get("es3_due") or []):
            raise L.Stop(f"es3: {a.cell} is not in results/{a.gate}.json es3_due ({g.get('es3_due')}); read first", 2)
        used = int(g.get("es3_used") or 0)
        if used >= L.ES3_CAP:
            raise L.Stop(f"es3 allowance spent ({used} / {L.ES3_CAP} cell-seeds): the verdict is UNRESOLVED", 3)
        _read_is_for(g, a.gate, "pick" if a.gate == "g2" else "tdr", P, "es3")
        if g.get("verdict") != "PENDING-ES3":
            raise L.Stop(f"es3: results/{a.gate}.json verdict is {g.get('verdict')!r}; es3 runs only on a PENDING-ES3 "
                         "read (an UNRESOLVED or decided gate needs no es3)", 3)
        live = _live_es3(g, a.gate, P)
        if live != used:
            raise L.Stop(f"es3: results/{a.gate}.json counts {used} spent cell-seeds, {live} on disk now: read "
                         f"{a.gate} again first", 3)
    R.run_all(L.cell_rows(P, a.cell, 3))
    return 0


def ph_g2relaunch(a, R: Runner) -> int:
    P, Rf = L.policy_for(a.policy, "student", a.dry), L.policy("C28000")
    if a.cell not in L.N_CELLS:
        raise L.Stop(f"g2relaunch: {a.cell} is not an N cell", 2)
    if not a.dry:
        g = _gate_json("g2")
        if a.cell not in (g.get("relaunch_due") or []):
            raise L.Stop(f"g2relaunch: {a.cell} is not class-ambiguous in results/g2.json", 2)
        if int(g.get("es3_only", g.get("es3_used")) or 0) + 4 > L.ES3_CAP:
            raise L.Stop("g2relaunch needs 4 cell-seeds of the es3 allowance: UNRESOLVED", 3)
        _read_is_for(g, "g2", "pick", P, "g2relaunch")
        if g.get("verdict") not in ("RELAUNCH-DUE", "PENDING-ES3"):
            raise L.Stop(f"g2relaunch: results/g2.json verdict is {g.get('verdict')!r} (RELAUNCH-DUE / PENDING-ES3 "
                         "only)", 3)
        if _live_es3(g, "g2", P) != int(g.get("es3_used") or 0):
            raise L.Stop("g2relaunch: es3 stamps changed since results/g2.json was read: read g2 again first", 3)
    R.run_all(cellseed_rows(P, [a.cell], suffix=L.RELAUNCH) + cellseed_rows(Rf, [a.cell], suffix=L.RELAUNCH))
    return 0


def _sel_rows_path() -> Path:
    return L.RESULTS / "sel.rows.json"


def _record_rows(rows: dict) -> None:
    rec = L.read_json(_sel_rows_path(), {}) or {}
    for key, (row, st) in rows.items():
        if st is not None:
            rec[key] = {"stamp": str(st), "job": row, "at": L.now()}
    L.write_json(_sel_rows_path(), rec)


def ph_sel(a, R: Runner) -> int:
    """pipeline.py ex_select (2244-2318) + full_block (2352-2410), line for line, with the 90-kg level explicit;
    the teacher reference rows are G1's TDR stamps (never re-run here)."""
    st = L.sel_spec()
    T = L.policy_for(a.teacher, "teacher", a.dry)
    if T["role"] != "teacher":
        raise L.Stop(f"--teacher {a.teacher} is not a teacher policy", 2)
    if not a.dry:
        g1 = _gate_json("g1")
        if g1.get("verdict") != "PASS" or ((g1.get("policies") or {}).get("tdr") or {}).get("sha256") != T["sha256"]:
            raise L.Stop(f"sel: the student stage runs on a G1 PASS only, for that read's TDR (results/g1.json "
                         f"verdict {g1.get('verdict')!r}, tdr {((g1.get('policies') or {}).get('tdr') or {}).get('name')!r}"
                         f" vs --teacher {a.teacher})", 3)
    run_dir = Path(a.student_run)
    if not run_dir.is_absolute() or not run_dir.is_dir():
        raise L.Stop(f"--student-run {run_dir}: not an absolute run dir", 2)
    start = T["iteration"] if T["iteration"] is not None else int(a.dry_teacher_iter)
    cands = L.ss().candidate_checkpoints(run_dir, start, st["offsets"])
    kg = int(st["prescreen_level"]["kg"])
    pres, pols = {}, {}
    for c in cands:
        if not c.get("ckpt"):
            continue
        cid = f"c{c['iter']}"
        P = L.ephemeral_policy(c["ckpt"], "student", cid)
        pols[cid] = P
        rows = {**{f"{cid}:{k}": r for k, r in L.student_sel_rows(P, "prescreen").items()},
                **{f"{cid}:{k}@L{kg:03d}": r for k, r in L.student_sel_rows(P, "prescreen_level").items()}}
        got = {k: (r, R.run(r)) for k, r in rows.items()}
        if R.dry:
            continue
        _record_rows(got)
        stamps = {k.split(":", 1)[1]: s for k, (r, s) in got.items() if "@L" not in k}
        lst = {k.split(":", 1)[1].split("@L")[0]: s for k, (r, s) in got.items() if "@L" in k}
        pres[cid] = L.prescreen_read({"iter": c["iter"], "ckpt": c["ckpt"]}, stamps, lst)
    trows = L.teacher_sel_rows(T)
    tst, tprob = {}, []
    for k, r in trows.items():
        s = L.row_stamp(r)
        if s is None:
            tprob.append(f"{L.row_id(r)}: missing")
        else:
            p = L.row_problems(s, r, L.launch_pin())
            tprob += p
            tst[k] = s
    if R.dry:
        print(f"RSD DRY sel teacher rows (reused from G1, never run here): "
              f"{[L.row_id(r) for r in trows.values()]}", flush=True)
        first = next(iter(pols.values()), None)
        if first is not None:
            pre = set(st["prescreen_rows"])
            prel = set(st["prescreen_level"]["rows"])
            for k, r in L.student_sel_rows(first, "full").items():
                if k not in pre:
                    R.run(r)
            for k, r in L.student_sel_rows(first, "full_level").items():
                if k not in prel:
                    R.run(r)
        return 0
    if tprob:
        raise L.Stop(f"sel: the teacher reference rows are missing / invalid (run g1 first): {tprob[:4]}", 4)
    ranked = [f"c{p['iter']}" for p in L.ss().rank(list(pres.values()))]
    g1 = _gate_json("g1")
    extended = bool(g1.get("extended"))
    blocks, bstate = [], {}
    doc = {"stage": "sel", "teacher": T, "student_run": str(run_dir), "candidates": cands, "prescreen": pres,
           "ranked": ranked, "level_kg": 90, "max_failed_blocks": int(st.get("max_failed_blocks", 2))}
    while True:
        act = L.ss().next_action({"ranked": ranked, "blocks": bstate}, int(st.get("max_failed_blocks", 2)))
        if act["do"] != "block":
            break
        if blocks:
            proj = (BUDGET["extend_total"] if extended else BUDGET["main_total"]) + BUDGET["extra_block"] * len(blocks)
            if proj > BUDGET["cap"] and not a.approve_extra_block:
                doc.update(verdict="BUDGET-ASK", blocks={b["cand"]: b for b in blocks}, gate=L.sel_gate_doc(blocks))
                L.write_json(L.RESULTS / "sel.json", doc)
                raise L.Stop(f"budget: a further selection block projects {proj:.1f} > {BUDGET['cap']:g} GPU-h "
                             "(CHARTER 3.6): stop and ask (--approve-extra-block)", 3)
        cid = act["cand"]
        P = pols[cid]
        rows = {**{f"{cid}:{k}": r for k, r in L.student_sel_rows(P, "full").items()},
                **{f"{cid}:{k}@L090": r for k, r in L.student_sel_rows(P, "full_level").items()}}
        got = {k: (r, R.run(r)) for k, r in rows.items()}
        _record_rows(got)
        stamps = {k.split(":", 1)[1]: s for k, (r, s) in got.items() if "@L" not in k}
        lst = {k.split(":", 1)[1].split("@L")[0]: s for k, (r, s) in got.items() if "@L" in k}
        blk = L.full_block_read(cid, stamps, lst, tst, [], lvl=90)
        blocks.append(blk)
        bstate[cid] = "PASS" if blk["pass"] else "FAIL"
        say(f"RSD READ sel block {cid} pass={blk['pass']} failed={[d['id'] for d in blk['decisive'] if not d.get('pass')]}")
    doc.update(blocks={b["cand"]: b for b in blocks}, gate=L.sel_gate_doc(blocks), action=act,
               failed_blocks=[c for c, v in bstate.items() if v == "FAIL"])
    if act["do"] == "selected":
        c = pres[act["cand"]]
        pick = L.policy_add(a.pick_name, c["ckpt"], "student", f"selected {act['cand']} (teacher {a.teacher})")
        doc.update(verdict="SELECTED", pick={"cand": act["cand"], "ckpt": c["ckpt"], "sha256": pick["sha256"],
                                             "iter": c["iter"], "teacher": a.teacher, "policy": a.pick_name})
        L.write_json(L.RESULTS / "sel.json", doc)
        say(f"RSD READ sel SELECTED {act['cand']} as policy {a.pick_name} (watch flags "
            f"{[f['id'] for f in doc['blocks'][act['cand']]['watch_flags']]})")
        return 0
    doc.update(verdict="HALT", why=act.get("why"))
    L.write_json(L.RESULTS / "sel.json", doc)
    raise L.Stop(f"sel HALT: {act.get('why')}", 3)


def ph_read(a) -> int:
    if a.which == "g1":
        if not a.tdr:
            raise L.Stop("read g1 needs --tdr", 2)
        if a.extended and not a.prior_tdr:
            raise L.Stop("read g1 --extended needs --prior-tdr (the pre-EXTEND TDR: the es3 allowance is per gate)", 2)
        res = L.g1_read(a.tdr, a.old, a.tctl, extended=a.extended, prior_tdr_name=a.prior_tdr)
        out = Path(a.out or L.RESULTS / "g1.json")
    else:
        if not a.pick:
            raise L.Stop("read g2 needs --pick", 2)
        res = L.g2_read(a.pick, a.ref, a.tdr)
        out = Path(a.out or L.RESULTS / "g2.json")
    L.write_json(out, res)
    out.with_suffix(".md").write_text(L.gate_md(res))
    say(f"RSD READ {a.which} verdict={res['verdict']} notes={res['notes'][:3]} out={out}")
    return 0


def ph_replay(a) -> int:
    dose = not a.no_dose
    res = {"read_at": L.now(), "dose_checked": dose,
           "a_sweep": L.replay_sweep(dose=dose), "b_task4_sel": L.replay_sel(dose=dose),
           "c_f19_zero_kg": L.replay_zero_kg(dose=dose)}
    res["exact"] = all(res[k]["exact"] for k in ("a_sweep", "b_task4_sel", "c_f19_zero_kg"))
    res["checks_clean"] = all(res[k]["checks_clean"] for k in ("a_sweep", "b_task4_sel", "c_f19_zero_kg"))
    out = Path(a.out or L.RESULTS / "replay.json")
    L.write_json(out, res)
    md = ["# CPU replay (step-8 precondition)", "", f"Read {res['read_at']}; dose checked: {dose}.", "",
          "| item | exact | launch checks clean | detail |", "|---|---|---|---|"]
    sa, sb, sc = res["a_sweep"], res["b_task4_sel"], res["c_f19_zero_kg"]
    md.append(f"| (a) sweep table.json + teacher.json | {sa['exact']} | {sa['checks_clean']} | {sa['cellseeds']} "
              f"cell-seeds; closing {sa['closing']['mine']} |")
    md.append(f"| (b) Task 4 F19 sel | {sb['exact']} | {sb['checks_clean']} | pick {(sb.get('pick') or {}).get('cand')}, "
              f"ranked {sb.get('ranked')}, failed blocks {sb.get('failed_blocks')} |")
    for sid, s in sc["stages"].items():
        md.append(f"| (c) F19 {sid} zero-kg (its {s['its_past_p2a']}) | {not s['diffs']} | {sc['checks_clean']} | "
                  f"{s['mine']['verdict']}, max margin used {s['mine']['max_margin_used']} |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    say(f"RSD READ replay exact={res['exact']} checks_clean={res['checks_clean']} out={out}")
    return 0 if res["exact"] and res["checks_clean"] else 3


def ph_policy(a) -> int:
    if a.pcmd == "add":
        p = L.policy_add(a.name, a.checkpoint, a.role)
        print(json.dumps(p, indent=1))
        return 0
    reg = L.load_registry()
    for name in ("TSTAR", "C28000"):
        if name not in reg["policies"]:
            L.policy(name)
    print(json.dumps(L.load_registry(), indent=1))
    return 0


ROW_PHASES = ("refarm", "bridge", "g1", "tctl", "closing", "es3", "extend", "sel", "g2ref", "g2", "g2relaunch")


def run_phase(a) -> int:
    R = Runner(a.dry, a.cmd)
    if not a.dry:
        pin = L.check_tree()
        bad = L.disk_problems()
        if bad:
            raise L.Stop("disk " + "; ".join(bad), 4)
        sp = L.strip_noop_problems()
        if sp:
            raise L.Stop(f"the shared L000 rows are not identical to Task 4's stripped rows: {sp}", 5)
        say(f"RSD START {a.cmd} {getattr(a, 'policy', '') or ''} code_sha={pin['code_sha'][:12]} "
            f"eval_sha={pin['eval_sha'][:12]} out={L.TP_OUT}")
    if a.cmd == "refarm":
        R.run_all(refarm_rows())
    elif a.cmd == "bridge":
        ph_bridge(a, R)
    elif a.cmd in ("g1", "extend"):
        if not a.dry:
            g1_bridge_guard(a)
        R.run_all(g1_rows(a.policy, a.dry))
    elif a.cmd == "tctl":
        if not a.dry:
            g1_bridge_guard(a)
        R.run_all(tctl_rows(a.policy, a.dry))
    elif a.cmd == "closing":
        R.run_all(L.cell_rows(L.policy_for(a.policy, "student" if a.gate == "g2" else "teacher", a.dry),
                              L.CLOSING_CELL, 1))
    elif a.cmd == "es3":
        ph_es3(a, R)
    elif a.cmd == "sel":
        ph_sel(a, R)
    elif a.cmd == "g2ref":
        R.run_all(g2ref_rows())
    elif a.cmd == "g2":
        if not a.dry:
            sel = L.read_json(L.RESULTS / "sel.json", {}) or {}
            if sel.get("verdict") != "SELECTED" or (sel.get("pick") or {}).get("sha256") != L.policy(a.policy)["sha256"]:
                raise L.Stop(f"g2: --policy {a.policy} is not the selection's pick (results/sel.json verdict "
                             f"{sel.get('verdict')!r})", 3)
            g2_bridge_guard(a)
        R.run_all(g2_rows(a.policy, a.dry))
    elif a.cmd == "g2relaunch":
        ph_g2relaunch(a, R)
    if a.dry:
        print(f"RSD DRY phase={a.cmd} rows={R.n}", flush=True)
    else:
        say(f"RSD END {a.cmd} rows={R.n}")
    return 0


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("policy")
    ps = p.add_subparsers(dest="pcmd", required=True)
    pa = ps.add_parser("add")
    pa.add_argument("name")
    pa.add_argument("--checkpoint", required=True)
    pa.add_argument("--role", required=True, choices=("teacher", "student"))
    ps.add_parser("list")
    sub.add_parser("pin")
    for name in ROW_PHASES:
        q = sub.add_parser(name)
        q.add_argument("--dry", action="store_true")
        if name in ("es3", "g2relaunch"):
            q.add_argument("cell")
        if name == "bridge":
            q.add_argument("--policy", default="TSTAR")
            q.add_argument("--relaunch", action="store_true")
        elif name == "sel":
            q.add_argument("--student-run", required=True)
            q.add_argument("--teacher", required=True)
            q.add_argument("--pick-name", default="PICK")
            q.add_argument("--approve-extra-block", action="store_true")
            q.add_argument("--dry-teacher-iter", default=None,
                           help="dry run with an unregistered --teacher: the teacher iteration to map candidates from")
        elif name not in ("refarm", "g2ref"):
            q.add_argument("--policy", required=True)
        if name in ("g1", "extend", "tctl", "g2"):
            q.add_argument("--bridge-class-flip", action="store_true",
                           help="the reference bridge STOP (T*: g1 / extend / tctl; c28000: g2) was diagnosed on CPU "
                                "as a class flip and recorded")
        if name in ("es3", "closing"):
            q.add_argument("--gate", choices=("g1", "g2"), default="g1")
    r = sub.add_parser("read")
    r.add_argument("which", choices=("g1", "g2"))
    r.add_argument("--tdr")
    r.add_argument("--old", default="TSTAR")
    r.add_argument("--tctl")
    r.add_argument("--extended", action="store_true")
    r.add_argument("--prior-tdr", default=None, help="with --extended: the pre-EXTEND TDR policy name")
    r.add_argument("--pick")
    r.add_argument("--ref", default="C28000")
    r.add_argument("--out")
    rp = sub.add_parser("replay")
    rp.add_argument("--no-dose", action="store_true")
    rp.add_argument("--out")
    return ap


def main(argv=None) -> int:
    a = parser().parse_args(argv)
    try:
        import numpy  # noqa: F401,PLC0415 -- fail fast: the readers need it (run with the Isaac venv python)
    except ImportError:
        print("RSD needs the Isaac venv python (numpy)", flush=True)
        return 2
    lock = None
    if a.cmd in ROW_PHASES and not a.dry:
        L.RESULTS.mkdir(parents=True, exist_ok=True)
        lock = open(LOCK, "w")  # noqa: SIM115 -- held for the process lifetime
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("RSD another driver holds results/rsd_gate.lock", flush=True)
            return 6
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, _on_signal)
    try:
        if a.cmd == "policy":
            return ph_policy(a)
        if a.cmd == "pin":
            pin = L.write_pin()
            say(f"RSD PIN code_sha={pin['code_sha']} eval_sha={pin['eval_sha']} head={pin['head']}")
            return 0
        if a.cmd == "read":
            return ph_read(a)
        if a.cmd == "replay":
            return ph_replay(a)
        return run_phase(a)
    except L.Stop as s:
        if a.cmd in ROW_PHASES and not getattr(a, "dry", False):
            say(f"RSD STOP {a.cmd} {s.why}")
            notify("STOP", f"{a.cmd}: {s.why}")
        else:
            print(f"RSD STOP {a.cmd} {s.why}", flush=True)
        return s.code
    except L.PolicyError as e:
        print(f"RSD POLICY {e}", flush=True)
        return 2
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 -- every exit leaves a sentinel
        if a.cmd in ROW_PHASES and not getattr(a, "dry", False):
            say(f"RSD STOP {a.cmd} crash {type(exc).__name__}: {exc}")
            notify("STOP", f"{a.cmd} crash {type(exc).__name__}: {exc}")
            (L.RESULTS / "crash.txt").write_text(traceback.format_exc())
        else:
            traceback.print_exc()
        return 7


if __name__ == "__main__":
    sys.exit(main())
