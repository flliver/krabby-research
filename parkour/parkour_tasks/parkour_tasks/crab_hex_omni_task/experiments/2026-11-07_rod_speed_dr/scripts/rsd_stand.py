"""Stand re-pairing to TDR (``STAND_REPAIR.md``): Task 4's stand stages ``sn`` and ``sh`` (``pipeline.yaml`` 766-827,
``pipeline.py`` ``ex_dagger_sn`` / ``ex_dagger_sh``) re-run with the walker switched from T* to TDR.

It runs this worktree's copies of the stand, phase-2 and pipeline scripts at the campaign pin, and writes every row,
run and table into the main checkout beside F19's, where the readers find the S1-era controls. Task 4's own
``stand_gate`` / ``sh_read_spec`` / ``clean_env`` / ``parse_row`` / ``Spec`` / ``dagger_last_it`` are loaded from
``pipeline.py`` as a library; nothing in ``tools/pipeline`` or the stand scripts is edited.

    rsd_stand.py preflight         STAND_REPAIR.md section 5 (CPU): P1 gate replay, P2 reader replay, P3 lazy-read
                                   replay, P4 static checks -> results/stand/preflight.json
    rsd_stand.py step STEP         one queue step (rsd_stand_queue.sh runs them in order):
                                   refs sn_train sn_rows sn_read sh_train sh_table sh_rows rh_rows sh_read
    rsd_stand.py walker            print the walker env the steps pass (after ``refs``)

A step prints ``RSDST <step> END rc=<rc> <why>`` last: rc 0 done; 4 a decide-record-notify stop (a gate HALT, a
premise STOP, a guard); 1 an infrastructure failure. Every step resumes: finished rows, runs and reads are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rsd_lib as L  # noqa: E402  (campaign paths, pin / tree / disk guards, json helpers)

# ------------------------------------------------------------------------------------------------ the walker swap
TDR_CKPT = L.W / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-07_19-49-46_RSD_TDR_s19/model_28991.pt"
TDR_SHA = "03c3e4728ac2175b4fbff66bc3944134c28fcbec0f6d8cc4aee4e5f3640c8567"
# T*'s Task 4 driver stamp (0-kg suite__omni es1, Walk-Play) was F19's REFS_ENV_STAMP. Here it is only the env input
# of the REFS(TDR) rows (run_arm reads its run_meta env_vars, nothing else); its env equals TDR's own 0-kg suite env.
TSTAR_DRIVER = L.MAIN / ("parkour/logs/rsl_rl/gait_eval/omni_phases/F19/seed19_X08_p3b_A15pB/suite__omni/seed001/"
                         "2026-10-22_16-32-02")
TSTAR_CKPT = L.MAIN / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-10-22_11-14-00_seed19_X08_p3b_A15pB/model_25992.pt"
TSTAR_SHA = "03b6009a6e952133f6c21c36c8d315b76f945d431615ef6d6798b7f4a5f59fc7"
TDR_L000 = L.TP_OUT / "lr/RSDT_03c3e472_T00_nominal_L000_A/suite__omni/seed001/2026-11-08_03-50-30"
WALK_PLAY = "Isaac-Crab-Hex-Omni-Walk-Play-v0"

REFS = "REFS_RSD_TDR"
SN = "SN_RSD_TDR_sn_s19"
SH = "SH_RSD_TDR_sh_s19"
SHZ = SH + "z"
RH = "SH_RH_RSD_TDR"
NAMES = (REFS, SN, SH, SHZ, RH)
RUN = "F19"             # the pipeline.yaml run whose recipe is reproduced (DAgger seed 19, chain F19_p2a)

# ------------------------------------------------------------------------------------- roots (as F19's: the main checkout)
P2_OUT = L.MAIN / "parkour/outputs/omni_plan/phase2"
LROUT = P2_OUT / "lr"
SN_OUT = L.MAIN / "parkour/outputs/omni_plan/stand_network"
SH_OUT = L.MAIN / "parkour/outputs/omni_plan/stand_height"
SH_TABLE_OUT = L.MAIN / "parkour/logs/videos/sh_table"
SN_SCRIPTS = L.EXP / "2026-10-11_stand_network/scripts"
SH_SCRIPTS = L.EXP / "2026-10-13_stand_height/scripts"
P2_SCRIPTS = L.EXP / "2026-10-07_omni_phase2/scripts"
RUN_ARM = P2_SCRIPTS / "run_arm.sh"
STAND = L.RESULTS / "stand"
LOGS = L.TP_OUT / "stand"
PYTHONPATH = f"{L.W}/parkour/parkour_tasks:{L.W}/parkour"
F19_STAND = L.T4 / "F19/stand"
F19_GATES = L.T4 / "F19/gates"
F19_KRABBY_ENV_SHA = "9f868e2c853ab471"   # params/krabby_env.json sha256 prefix of SN_T4F19_sn_s19 and SH_T4F19_sh_s19
F19_SN, F19_SH = "SN_T4F19_sn_s19", "SH_T4F19_sh_s19"

# ----------------------------------------------------------------------------- Task 4's live rules (pipeline.yaml live)
POLL_S = 120
WATCHDOG_S = 30 * 60
TERM_GRACE_S = 120
LAUNCHER_EXIT_S = 300

# gate_row.sh's one-Isaac-process patterns, verbatim
ISAAC_PAT = re.compile(r"^[^ ]*[i]saac_venv/bin/python3?( -[^ ]+)* [^ ]*(run_phases|run_candidates|eval_crab_hex_gait"
                       r"|train|play|play_stream_video|record_categories|crab_hex_phase_cfg_dump|sh_[A-Za-z0-9_]+"
                       r"|sn_[A-Za-z0-9_]+|bc_dump_cfgs|t4_dump_cfgs)\.py")
NONKIT_PAT = re.compile(r"(s[hn]_dagger\.py .*--(read|emit-env)([ =]|$))|(s[hn]_table_test\.py .*--(emit-facts|merge)"
                        r"([ =]|$))|(/s[hn]_(facts|kin|core|table_core|eval_common|read|bridge_read|observe_read|calib"
                        r"|common)\.py( |$))")


class Infra(Exception):
    """An infrastructure failure (rc != 0, a missing checkpoint or read): rc 1, a retry is the session's."""


_PP = None
_SPEC = None


def pp():
    """Task 4's pipeline.py, loaded by path: definitions only, its queue never runs here."""
    global _PP
    if _PP is None:
        _PP = L._load("rsd_tp_pipeline", L.PIPE / "pipeline.py")
    return _PP


def spec():
    global _SPEC
    if _SPEC is None:
        _SPEC = pp().Spec(L.SPEC_PATH)
    return _SPEC


def say(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


# ============================================================================================ guards and processes
def isaac_busy() -> list[str]:
    out = subprocess.run(["ps", "-ww", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return [a for a in out if ISAAC_PAT.search(a) and not NONKIT_PAT.search(a)]


def guard(what: str) -> None:
    """Before every Isaac launch: a clean tree at the campaign pin, >= 30 GB free, the DerivedDataCache <= 78 GB
    (CHARTER 3.5), then wait while any other Isaac process runs (gate_row.sh's patterns)."""
    try:
        L.check_tree()
    except L.Stop as e:
        raise L.Stop(f"tree / pin: {e.why}", 4) from None
    bad = L.disk_problems(L.MAIN / "parkour/outputs")
    if bad:
        raise L.Stop("; ".join(bad), 4)
    waited = False
    while True:
        busy = isaac_busy()
        if not busy:
            break
        if not waited:
            say(f"WAIT {what}: another Isaac process runs: {busy[0][:200]}")
            waited = True
        time.sleep(20)
    say(f"GUARD ok: {what}")


def child_env(extra: dict) -> dict:
    """pipeline.clean_env (no KRABBY_* / SN_WALKER_* / P2_* / P2R_* / SH_* / SN_STAND_* / TP_* from this process),
    the worktree on PYTHONPATH, then only what the step passes."""
    env = pp().clean_env()
    env["PYTHONPATH"] = PYTHONPATH
    env["OMNI_KIT_ACCEPT_EULA"] = "yes"
    env.update({k: str(v) for k, v in extra.items()})
    return env


def run(cmd: list, env: dict, log: Path, what: str) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"# {time.strftime('%F %T')} $ {' '.join(map(str, cmd))}\n")
        f.flush()
        p = subprocess.Popen([str(c) for c in cmd], env=env, stdout=f, stderr=subprocess.STDOUT,
                             cwd=str(L.W / "parkour"), start_new_session=True)
        rc = p.wait()
        f.write(f"# {time.strftime('%F %T')} rc={rc}\n")
    say(f"{what}: rc={rc} (log {log})")
    return rc


def complete_stamp(name: str, sc: str, es: int) -> Path | None:
    """The newest complete stamp of a row: run_arm's marker (complete.json / p2_complete.json) and its code-state
    sidecar, the newest by its UTC name, as p2_read picks it."""
    base = LROUT / name / sc / f"seed{int(es):03d}"
    for d in sorted(base.glob("*"), reverse=True):
        if d.is_dir() and ((d / "complete.json").is_file() or (d / "p2_complete.json").is_file()) \
                and (d / "code_state.json").is_file():
            return d
    return None


def scen_rows(keys: list) -> list[tuple]:
    """pipeline.yaml row keys (family@es) -> [(scenario, es, family)]."""
    out = []
    for k in keys:
        r = pp().parse_row(k)
        out.append((spec().row(r["family"])["scenario"], r["es"], r["family"]))
    return out


def fresh_json(p: Path, t0: float):
    if not p.is_file() or p.stat().st_mtime < t0 - 1:
        return None
    return L.read_json(p)


# ============================================================================================ the walker
def env_vars(stamp: Path) -> dict:
    return (L.read_json(Path(stamp) / "run_meta.json") or {}).get("env_vars") or {}


def walker_problems(stamp: Path, sc: str, es: int) -> list[str]:
    """A REFS(TDR) stamp: Walk-Play, TDR's checkpoint, 0 kg, no DR / stress / payload key, and the env its row should
    have: T*'s driver env plus the row's own manifest env (the field rows record their terrain keys on top, as F19's
    REFS_T4F19 field rows do), which must also equal REFS_T4F19's same row key for key."""
    m = L.read_json(Path(stamp) / "run_meta.json") or {}
    bad = []
    if m.get("task") != WALK_PLAY:
        bad.append(f"task {m.get('task')}")
    if m.get("checkpoint_sha256") != TDR_SHA:
        bad.append(f"checkpoint_sha256 {str(m.get('checkpoint_sha256'))[:12]}")
    if m.get("payload") is not None:
        bad.append(f"payload {m.get('payload')}")
    ev = m.get("env_vars") or {}
    if ev != {**env_vars(TSTAR_DRIVER), **(m.get("env_vars_manifest") or {})}:
        bad.append("env_vars != the driver env + this row's manifest env")
    f19 = complete_stamp("REFS_T4F19", sc, es)
    if f19 is None or ev != env_vars(f19):
        bad.append("env_vars != REFS_T4F19's same row")
    odd = [k for k in ev if k.startswith(("KRABBY_DR_", "KRABBY_STRESS_", "KRABBY_PAYLOAD_"))]
    if odd:
        bad.append(f"keys {odd}")
    return bad


F19_KE = SN_OUT / "runs" / F19_SN / "params" / "krabby_env.json"


def krabby_env_problems(ke: Path) -> list[str]:
    """The trainer's composed KRABBY_* env equals F19's key for key, except that KRABBY_RSI_BANK (an absolute path the
    omni preset derives from the checkout's own location) points into this worktree, at a file byte-identical to
    F19's (and KRABBY_RSI_FRAC is 0: the bank is never sampled)."""
    got, ref = L.read_json(ke) or {}, L.read_json(F19_KE) or {}
    if not ref:
        return [f"no F19 reference {F19_KE}"]
    want = dict(ref)
    if "KRABBY_RSI_BANK" in want:
        want["KRABBY_RSI_BANK"] = ref["KRABBY_RSI_BANK"].replace(str(L.MAIN), str(L.W), 1)
    bad = [f"{k}: {got.get(k)!r} != {want.get(k)!r}" for k in sorted(set(got) | set(want)) if got.get(k) != want.get(k)]
    if not bad and "KRABBY_RSI_BANK" in got:
        a, b = Path(got["KRABBY_RSI_BANK"]), Path(ref["KRABBY_RSI_BANK"])
        if not a.is_file() or not b.is_file() or L.sha256_file(a) != L.sha256_file(b):
            bad.append("KRABBY_RSI_BANK bytes differ from F19's")
    return bad


def walker_spec_doc(refs_env_stamp: Path) -> dict:
    """pipeline.Queue.walker_spec_env's SN_WALKER_SPEC with TDR in T*'s place. TDR has no Task 4 driver stamp: its
    REFS suite@1 stamp (Walk-Play, TDR, 0 kg, the driver env) stands in, as the SH table test requires."""
    s = spec()
    return {"S1_CKPT": str(TDR_CKPT), "S1_SHA256": TDR_SHA,
            "S1_AGENT_YAML": str(TDR_CKPT.parent / "params" / "agent.yaml"),
            "REFS_ENV_STAMP": str(refs_env_stamp), "CANDIDATES": str(L.T4 / "F19" / "candidates.yaml"),
            "OVERLAY_CHAIN": s.chain(RUN, "p2a"), "S1_OVERLAY": s.overlay("p2a"), "REFS_NAME": REFS}


def walker_env() -> dict:
    w = L.read_json(STAND / "walker_spec.json")
    if not w:
        raise Infra("no results/stand/walker_spec.json: run step refs first")
    sp = w["spec"]
    if L.sha256_file(Path(sp["S1_CKPT"])) != sp["S1_SHA256"]:
        raise L.Stop(f"walker checkpoint sha != {sp['S1_SHA256'][:12]}", 4)
    return {"SN_WALKER_SPEC": json.dumps(sp, sort_keys=True), "SN_WALKER_CK": sp["S1_CKPT"],
            "SN_WALKER_ST": sp["REFS_ENV_STAMP"], "P2_REANCHOR": "1"}


def need(path: Path, what: str) -> dict:
    d = L.read_json(path)
    if not d:
        raise Infra(f"{what} missing ({path})")
    return d


def gate_ok(sid: str) -> dict:
    g = need(STAND / "gates" / f"{sid}.json", f"the {sid} gate")
    if g.get("verdict") not in ("PASS", "WATCH") and not (
            L.read_json(STAND / "gates" / f"{sid}_reanchor.json") or {}).get("reanchored"):
        raise L.Stop(f"the {sid} gate is {g.get('verdict')}, not PASS / WATCH / re-anchored", 4)
    return g


# ============================================================================================ the gate (Task 4's code)
def thresholds():
    p = spec().thresholds_path()
    if p.resolve() != L.TH_PATH.resolve():
        raise Infra(f"pipeline.yaml thresholds {p} != {L.TH_PATH}")
    return pp().GR.Thresholds.load(p)


def gate(sid: str, which: str, rec: dict, out_name: str | None = None) -> dict:
    """pipeline.stand_gate (D1 = SN G1-G5 on sn_calib's head row; D2 = SH K1-K6 on sh_read + K5 on sn_calib of the
    head's z rows), called with a thresholds-only ctx; its write_gate is swapped for one writing results/stand/gates."""
    mod = pp()
    th = thresholds()
    path = STAND / "gates" / f"{out_name or sid}.json"

    def write_gate(_ctx, sid_, res):
        tp = spec().thresholds_path()
        L.write_json(path, {"run": "RSD_TDR (Task 4's sn / sh recipe, walker TDR)", "stage": sid_, "read_at": L.now(),
                            "thresholds": {"file": tp.name, "version": spec().d["thresholds"]["version"],
                                           "sha256": L.sha256_file(tp)}, **res})
        return path

    saved = mod.write_gate
    mod.write_gate = write_gate
    try:
        return mod.stand_gate(SimpleNamespace(thresholds=lambda: th), sid, which, rec, x=False)
    finally:
        mod.write_gate = saved


def clause_op(cl: str) -> str | None:
    m = re.search(r"(>=|<=)", cl)
    return m.group(1) if m else None


def rh_derived(d: dict, lines: dict) -> bool:
    """A D2 line that the lazy R_h control beside the walker can re-anchor: its line comes from the S1-era R_h value
    (the clause names R_h's), no thresholds override fixes it (K4 miss <= 3 / 800 is fixed: D9), and the control's
    value comes from the height rows the control re-runs. K3 does not qualify: its R_h value comes from the S1-era
    table test, which the lazy spec keeps (F19's spec: 'table stats stay the S1-era R_h'), so a K3 failure is a HALT.
    K5's lines are R / Z1 / HOLD / REFS-derived: a HALT."""
    ln = lines.get(d["id"]) or {}
    return ("R_h" in str(ln.get("clause") or "") and ln.get("override_line") is None
            and not ln["gate"].startswith("K5/") and ln["gate"] != "K3")


BH_RE = re.compile(r"^(suite|walkstop) es(\d+) \|body height - REFS\| <= ([0-9.]+)$")
BH_CONTROLS = ("R", "HOLD")


def body_height_reanchor(res: dict, calib: dict, head: str) -> dict:
    """The user's decision of 2026-11-09 11:53 (STAND_REPAIR.md section 4), in DESIGN 3.6's form: a failed body-height
    line (D1 G5 ``(suite|walkstop) es<k> |body height - REFS| <= 0.03``, or its K5/G5 copy in D2) re-anchors when the R
    and the HOLD control each fail the same line against REFS (walker-caused); the head then passes if it stands within
    the line's tolerance of each control. The heights are sn_calib's own G5 records, ``[row, REFS]`` per key, for the
    head and for the R / HOLD rows of the same read (whose REFS height must be the head's)."""
    lines = {ln["id"]: ln for ln in thresholds().lines}
    rows = (calib or {}).get("rows") or {}

    def heights(name: str) -> dict:
        return (((rows.get(name) or {}).get("gates") or {}).get("G5") or {}).get("heights") or {}

    out = []
    for d in res["decisive"]:
        ln = lines.get(d["id"]) or {}
        m = BH_RE.match(str(ln.get("clause") or ""))
        if d.get("pass") or ln.get("gate") not in ("G5", "K5/G5") or not m:
            continue
        key, tol = f"{m.group(1)} es{m.group(2)}", float(m.group(3))
        hh = heights(head).get(key) or [None, None]
        rec = {"id": d["id"], "key": key, "tol": tol, "head": hh[0], "refs": hh[1], "controls": {}}
        ok = head_ok = hh[0] is not None and hh[1] is not None
        for c in BH_CONTROLS:
            ch = heights(c).get(key) or [None, None]
            have = ch[0] is not None and ch[1] is not None and head_ok
            fails = have and abs(ch[0] - ch[1]) > tol + 1e-12
            same_refs = have and abs(ch[1] - hh[1]) < 1e-9
            within = have and abs(hh[0] - ch[0]) <= tol + 1e-12
            rec["controls"][c] = {"height": ch[0], "refs": ch[1], "control_fails_same_line": fails,
                                  "same_refs": same_refs, "head_minus_control": (hh[0] - ch[0]) if have else None,
                                  "head_within_tol": within}
            ok = ok and fails and same_refs and within
        rec["reanchored"] = bool(ok)
        out.append(rec)
    return {"rule": "the user's 2026-11-09 11:53 decision: a failed body-height line re-anchors on the R and HOLD "
                    "controls when each fails the same line against REFS; the head passes within the tolerance of "
                    "each control (STAND_REPAIR.md section 4)",
            "lines": out, "covered": [r["id"] for r in out if r["reanchored"]]}


def lazy_decide(res: dict, lazy: dict, head: str, ids: list | None = None) -> dict:
    """DESIGN 3.6 for R_h-derived lines: the control beside the walker fails the same (S1-era) line -> the line
    re-anchors on the control (walker-caused, decide-record-notify), read off the lazy read's head clause at the
    re-anchored line. Otherwise HALT. ``ids`` limits the read to those failed lines (the ones another rule left);
    every one of them must be R_h-derived and re-anchor for ``reanchored``."""
    lines = {ln["id"]: ln for ln in thresholds().lines}
    halts = [d for d in res["decisive"] if not d.get("pass") and (ids is None or d["id"] in ids)]
    other = [d["id"] for d in halts if not rh_derived(d, lines)]
    out = {"halt_lines": [d["id"] for d in halts], "not_rh_derived": other, "lines": []}
    for d in halts:
        if d["id"] in other:
            continue
        ln = lines[d["id"]]
        g, cl = ln["gate"], ln["clause"]
        ctrl = (((lazy.get("gates") or {}).get(g) or {}).get("R_h") or {}).get("clauses", {}).get(cl) or {}
        hc = (((lazy.get("gates") or {}).get(g) or {}).get(head) or {}).get("clauses", {}).get(cl) or {}
        op, v, orig = clause_op(cl), ctrl.get("value"), d.get("line")
        fails = (v is not None and orig is not None and op is not None
                 and not ((v >= orig - 1e-12) if op == ">=" else (v <= orig + 1e-12)))
        out["lines"].append({"id": d["id"], "clause": cl, "op": op, "s1_era_line": orig, "head": d.get("value"),
                             "control": v, "control_note": ctrl.get("note"), "control_fails_same_line": fails,
                             "reanchored_line": hc.get("line"), "head_value_lazy": hc.get("value"),
                             "head_passes_reanchored": bool(hc.get("ok"))})
    out["covered"] = [x["id"] for x in out["lines"] if x["control_fails_same_line"] and x["head_passes_reanchored"]]
    out["reanchored"] = bool(halts) and not other and len(out["covered"]) == len(out["lines"])
    out["rule"] = ("DESIGN 3.6: if the control fails the same line, the line re-anchors on the control "
                   "(walker-caused, decide-record-notify); otherwise HALT")
    return out


# ============================================================================================ DAgger training
def trainer_pids(run_dir: Path) -> list[int]:
    tok, pids = str(run_dir), []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            a = [x.decode(errors="ignore") for x in (d / "cmdline").read_bytes().split(b"\0") if x]
        except OSError:
            continue
        if a and "python" in Path(a[0]).name and any(a[i] == "--out" and a[i + 1] == tok for i in range(len(a) - 1)):
            pids.append(int(d.name))
    return pids


def stop_trainer(run_dir: Path, proc, why: str) -> None:
    """SIGTERM the trainer pid, the grace, SIGTERM, the grace: never SIGKILL (this campaign's rule; Task 4's
    stop_trainer escalated to SIGKILL, not reproduced)."""
    say(f"STOP trainer of {run_dir.name}: {why}: SIGTERM, {TERM_GRACE_S} s, SIGTERM, {TERM_GRACE_S} s (never SIGKILL)")
    if not trainer_pids(run_dir) and proc.poll() is None:
        # no trainer yet (the launcher still waits in --wait-idle): its process group holds bash / sleep, no Kit
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in (1, 2):
        pids = trainer_pids(run_dir)
        if not pids:
            break
        for p in pids:
            try:
                os.kill(p, signal.SIGTERM)
            except ProcessLookupError:
                pass
        t0 = time.time()
        while time.time() - t0 < TERM_GRACE_S and trainer_pids(run_dir):
            time.sleep(5)
    left = trainer_pids(run_dir)
    if left:
        say(f"STOP: trainer pid(s) {left} alive after 2 x SIGTERM; NOT SIGKILLed (the user's call)")
    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < LAUNCHER_EXIT_S:
        time.sleep(5)


def premise(trainer: Path, run_dir: Path, lo: int, hi: int, env: dict) -> dict:
    r = subprocess.run([L.PY, "-I", str(trainer), "--read", str(run_dir), "--premise", f"{lo}:{hi}"], env=env,
                       capture_output=True, text=True, cwd=str(L.W / "parkour"))
    prem = None
    for cand in [r.stdout] + r.stdout.strip().splitlines()[::-1]:
        try:
            doc = json.loads(cand)
        except ValueError:
            continue
        if isinstance(doc, dict) and "premise" in doc:
            prem = doc.get("premise") or {}
            break
    if prem is None:
        prem = {"stop": None, "unreadable": (r.stdout + r.stderr)[-400:]}
    return {"window": [lo, hi], "rc": r.returncode, "premise": prem, "at": L.now()}


def train(sid: str, name: str, launcher: Path, trainer: Path, out_key: str, out_root: Path, args: list) -> dict:
    """One DAgger run through its launcher (foreground; the launcher's rc is the trainer's), watched every 120 s:
    params/krabby_env.json must equal F19's (no DR / stray key reached the trainer); Task 4's progress watchdog
    (30 min); the trainer's own premise read once at it >= N/2 over [N//3, N//2] (a STOP stops the run: HALT)."""
    head_p = STAND / f"{sid}_head.json"
    run_dir = out_root / "runs" / name
    iters = int(args[args.index("--iterations") + 1])
    ck = run_dir / f"model_{iters}.pt"
    if ck.is_file() and L.read_json(head_p):
        say(f"SKIP {sid} train: {ck} recorded")
        return L.read_json(head_p)
    if run_dir.exists() and any(run_dir.iterdir()) and not ck.is_file():
        raise L.Stop(f"{run_dir} is not empty and has no {ck.name}: a DAgger run never resumes; a retry trains afresh "
                     f"under {name}_a2 (the user's call)", 4)
    env = child_env({**walker_env(), out_key: out_root})
    lo, hi = iters // 3, iters // 2
    if not ck.is_file():
        guard(f"DAgger {name}")
        log = LOGS / f"{sid}.train.log"
        LOGS.mkdir(parents=True, exist_ok=True)
        cmd = ["bash", str(launcher), "--wait-idle", name, "--", *map(str, args)]
        with open(log, "a") as f:
            f.write(f"# {time.strftime('%F %T')} $ {out_key}={out_root} {' '.join(cmd)}\n")
            f.flush()
            proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(L.W / "parkour"),
                                    start_new_session=True)
        say(f"LAUNCH {sid} DAgger {name} ({' '.join(map(str, args))})")
        launched, started, env_checked, prem_done = time.time(), None, False, False
        while proc.poll() is None:
            time.sleep(POLL_S)
            if proc.poll() is not None:
                break
            if started is None and trainer_pids(run_dir):
                started = time.time()            # the watchdog clock starts with the trainer, not the launcher
            ke, tl = run_dir / "params" / "krabby_env.json", run_dir / "train_log.jsonl"
            # checked once the trainer writes its train log: its own SIGTERM handler is installed by then
            if not env_checked and ke.is_file() and tl.is_file():
                env_checked = True
                bad = krabby_env_problems(ke)
                if bad:
                    stop_trainer(run_dir, proc, f"krabby_env.json differs from F19's: {bad[:3]}")
                    raise L.Stop(f"{sid}: the trainer's KRABBY_* env differs from F19's: {bad[:5]}", 4)
                say(f"CHECK {sid}: params/krabby_env.json == F19's except the worktree RSI bank path "
                    f"(sha {L.sha256_file(ke)[:16]})")
            if started is not None:
                newest = max([started] + [p.stat().st_mtime for p in pp().dagger_progress_files(run_dir)
                                          if p.exists()])
                if time.time() - newest > WATCHDOG_S:
                    stop_trainer(run_dir, proc, "progress watchdog (30 min)")
                    raise Infra(f"{sid}: no training progress for {(time.time() - newest) / 60:.0f} min")
            elif time.time() - launched > WATCHDOG_S:
                stop_trainer(run_dir, proc, "no trainer process 30 min after the launch")
                raise Infra(f"{sid}: the launcher started no trainer in 30 min (log {log})")
            if not prem_done and tl.is_file():
                last = pp().dagger_last_it(tl)
                if last is not None and last >= hi:
                    pr = premise(trainer, run_dir, lo, hi, env)
                    if pr["rc"] == 0 and "unreadable" not in pr["premise"]:
                        prem_done = True        # an unreadable read (a torn jsonl line) retries at the next poll
                    L.write_json(STAND / f"{sid}_premise.json", pr)
                    say(f"PREMISE {sid} {lo}:{hi} rc={pr['rc']} stop={pr['premise'].get('stop')} "
                        f"{json.dumps(pr['premise'], sort_keys=True)[:400]}")
                    if pr["premise"].get("stop"):
                        stop_trainer(run_dir, proc, f"premise STOP at {lo}:{hi}")
                        raise L.Stop(f"{sid}: DAgger premise STOP at {lo}:{hi} (Task 4's live rule; the user's call)", 4)
        rc = proc.wait()
        with open(log, "a") as f:
            f.write(f"# {time.strftime('%F %T')} rc={rc}\n")
        if rc != 0 or not ck.is_file():
            raise Infra(f"{launcher.name} rc={rc}; {ck.name} {'present' if ck.is_file() else 'missing'} (log {log})")
    bad = krabby_env_problems(run_dir / "params" / "krabby_env.json")
    if bad:
        raise L.Stop(f"{sid}: the trainer's KRABBY_* env differs from F19's: {bad[:5]}", 4)
    pr = L.read_json(STAND / f"{sid}_premise.json")
    if not pr or pr.get("rc") != 0 or "unreadable" in (pr.get("premise") or {}):
        pr = premise(trainer, run_dir, lo, hi, env)
        pr["post_run"] = True                    # not Task 4's live rule (it never read after the run): a fallback
        L.write_json(STAND / f"{sid}_premise.json", pr)
        say(f"PREMISE {sid} (post-run read) rc={pr['rc']} stop={pr['premise'].get('stop')}")
        if pr["rc"] != 0 or "unreadable" in pr["premise"]:
            raise L.Stop(f"{sid}: the premise read is unreadable ({pr['premise']})", 4)
    if (pr.get("premise") or {}).get("stop"):
        raise L.Stop(f"{sid}: DAgger premise STOP at {lo}:{hi} ({'post-run' if pr.get('post_run') else 'live'} read; "
                     "the user's call)", 4)
    head = {"stage": sid, "name": name, "ckpt": str(ck), "sha256": L.sha256_file(ck),
            "train_log": str(run_dir / "train_log.jsonl"),
            "krabby_env_sha256": L.sha256_file(run_dir / "params" / "krabby_env.json"),
            "krabby_env_vs_f19": "equal except the worktree RSI bank path (bytes identical)",
            "status": L.read_json(run_dir / "status.json"), "args": list(map(str, args)), "recorded_at": L.now()}
    L.write_json(head_p, head)
    say(f"HEAD {sid} {name} model sha {head['sha256'][:12]}")
    return head


# ============================================================================================ the steps
def step_refs():
    """REFS(TDR): the nine walker-alone rows, as pipeline's REFS(T*) rows (run_arm.sh --row-eval, P2_REANCHOR=1),
    then the walker spec with REFS(TDR)'s own suite@1 stamp as REFS_ENV_STAMP / SN_WALKER_ST."""
    STAND.mkdir(parents=True, exist_ok=True)
    rows = scen_rows(spec().stage("sn")["refs_rows"])
    for sc, es, fam in rows:
        if complete_stamp(REFS, sc, es):
            say(f"SKIP REFS {sc}@{es}: complete")
            continue
        guard(f"REFS(TDR) {sc}@{es}")
        rc = run(["bash", RUN_ARM, "--row-eval", REFS, TDR_CKPT, TSTAR_DRIVER, spec().manifest(fam), sc, es],
                 child_env({"P2_REANCHOR": "1", "P2_OUT": P2_OUT}), LOGS / "refs.log", f"REFS(TDR) {sc}@{es}")
        if rc != 0 or not complete_stamp(REFS, sc, es):
            raise Infra(f"REFS(TDR) {sc}@{es} rc={rc}")
    stamps = {f"{sc}@{es}": complete_stamp(REFS, sc, es) for sc, es, _ in rows}
    bad = {f"{sc}@{es}": walker_problems(stamps[f"{sc}@{es}"], sc, es) for sc, es, _ in rows}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise L.Stop(f"REFS(TDR) stamps: {bad}", 4)
    s1 = stamps["suite__omni@1"]
    sp = walker_spec_doc(s1)
    L.write_json(STAND / "walker_spec.json", {
        "spec": sp, "refs_stamps": {k: str(v) for k, v in stamps.items()},
        "code_state": {k: L.read_json(v / "code_state.json") for k, v in stamps.items()},
        "note": "TDR has no Task 4 driver stamp; REFS(TDR) suite@1 (Walk-Play, TDR, 0 kg, the driver env) stands in",
        "written_at": L.now()})
    return f"REFS(TDR) 9 rows; REFS_ENV_STAMP {s1}"


def step_sn_train():
    if not (L.read_json(STAND / "preflight.json") or {}).get("ok"):
        raise L.Stop("the pre-flight is not ok (results/stand/preflight.json; STAND_REPAIR.md section 5)", 4)
    st = spec().stage("sn")
    args = ["--mode", "dagger", "--arm", st["arm"], "--seed", spec().seed(RUN, "sn"), "--iterations",
            spec().iters(RUN, "sn")]
    h = train("sn", SN, SN_SCRIPTS / "sn_dagger.sh", SN_SCRIPTS / "sn_dagger.py", "SN_OUT", SN_OUT, args)
    return f"{SN} {h['sha256'][:12]}"


def step_sn_rows():
    h = need(STAND / "sn_head.json", "the SN head")
    rows = scen_rows(spec().stage("sn")["stand_rows"])
    todo = [f"{sc}@{es}" for sc, es, _ in rows if not complete_stamp(SN, sc, es)]
    if not todo:
        return "SN rows complete (skipped)"
    guard("sn_eval_rows.sh")
    rc = run(["bash", SN_SCRIPTS / "sn_eval_rows.sh", SN, h["ckpt"], "1", *todo],
             child_env({**walker_env(), "P2_OUT": P2_OUT}), LOGS / "sn.rows.log", f"SN stand rows {todo}")
    miss = [f"{sc}@{es}" for sc, es, _ in rows if not complete_stamp(SN, sc, es)]
    if rc != 0 or miss:
        raise Infra(f"sn_eval_rows.sh rc={rc}; missing {miss}")
    return "SN rows 9"


def step_sn_read():
    h = need(STAND / "sn_head.json", "the SN head")
    miss = [f"{sc}@{es}" for sc, es, _ in scen_rows(spec().stage("sn")["stand_rows"]) if not complete_stamp(SN, sc, es)]
    if miss:
        raise Infra(f"SN rows missing {miss}")
    env = child_env({**walker_env(), "P2_OUT": P2_OUT})
    t0 = time.time()
    rcb = run([L.PY, "-I", SN_SCRIPTS / "sn_bridge_read.py", "--head", SN, "--refs", REFS, "--live", "--json",
               STAND / "sn_bridge.json"], env, LOGS / "sn.read.log", "sn_bridge_read (recorded, not gated)")
    rcc = run([L.PY, "-I", SN_SCRIPTS / "sn_calib.py", "--head", SN, "--json", STAND / "sn_calib.json", "--md",
               STAND / "sn_calib.md"], env, LOGS / "sn.read.log", "sn_calib")
    if fresh_json(STAND / "sn_calib.json", t0) is None:
        raise Infra(f"sn_calib rc={rcc} wrote no fresh sn_calib.json")
    rec = {"stage": "sn", "name": SN, "ckpt": h["ckpt"], "sha256": h["sha256"], "refs": REFS,
           "calib": str(STAND / "sn_calib.json"), "bridge": str(STAND / "sn_bridge.json"), "train_log": h["train_log"],
           "reader_rc": {"sn_bridge_read": rcb, "sn_calib": rcc}}
    res = gate("sn", "D1", rec)
    say(f"GATE sn {res['verdict']}: {len(res['decisive'])} decisive, halt {res['halt_lines']}, "
        f"{len(res['watch_flags'])} watch flag(s)")
    if res["verdict"] != "HALT":
        return f"SN gate {res['verdict']}"
    bh = body_height_reanchor(res, L.read_json(STAND / "sn_calib.json"), SN)
    remaining = [i for i in res["halt_lines"] if i not in bh["covered"]]
    dec = {"stage": "sn", "halt_lines": res["halt_lines"], "body_height": bh, "remaining": remaining,
           "reanchored": not remaining, "decided_at": L.now()}
    L.write_json(STAND / "gates" / "sn_reanchor.json", dec)
    say(f"REANCHOR sn body height: covered {len(bh['covered'])} of {len(res['halt_lines'])} "
        f"{json.dumps(bh['lines'])[:600]}")
    if remaining:
        raise L.Stop(f"SN gate HALT; not re-anchored: {remaining} (R / HOLD lazy controls or the user's call: "
                     "STAND_REPAIR.md section 4)", 4)
    return "SN gate HALT -> re-anchored (the body-height lines on the R / HOLD controls: the user's 11:53 decision)"


def step_sh_train():
    gate_ok("sn")
    sn = need(STAND / "sn_head.json", "the SN head")
    st = spec().stage("sh")
    args = ["--mode", "dagger", "--arm", "mask", "--seed", spec().seed(RUN, "sh"), "--iterations",
            spec().iters(RUN, "sh"), *st["opts"], "--warm-start", sn["ckpt"], "--warm-sha", sn["sha256"]]
    h = train("sh", SH, SH_SCRIPTS / "sh_dagger.sh", SH_SCRIPTS / "sh_dagger.py", "SH_OUT", SH_OUT, args)
    return f"{SH} {h['sha256'][:12]}"


def step_sh_table():
    h = need(STAND / "sh_head.json", "the SH head")
    rec_p = STAND / "sh_table.json"
    r = L.read_json(rec_p)
    if r and Path(r["dir"]).is_dir():
        return f"table recorded (skipped): {r['dir']}"
    db = str(spec().stage("sh").get("deadband", "0.25"))
    guard("sh_table_test.sh")
    t0 = time.time()
    rc = run(["bash", SH_SCRIPTS / "sh_table_test.sh", "--wait-idle", "--foreground", "--controllers", "stand_net",
              "--videos", "stand_net", "--stand-ckpt", h["ckpt"], "--label", SH, "--deadband", db],
             child_env({**walker_env(), "SH_TABLE_OUT": SH_TABLE_OUT}), LOGS / "sh.table.log", "SH table test")
    latest = SH_TABLE_OUT / "LATEST"
    tdir = (SH_TABLE_OUT / latest.read_text().strip()) if latest.is_file() else None
    if rc != 0 or tdir is None or not tdir.is_dir() or tdir.stat().st_mtime < t0 - 1:
        raise Infra(f"sh_table_test.sh rc={rc}; LATEST -> {tdir}")
    txt = " ".join((tdir / f).read_text(errors="ignore") for f in ("jobs.txt", "queue.log") if (tdir / f).is_file())
    if SH not in txt and h["ckpt"] not in txt:
        raise Infra(f"table dir {tdir} does not name {SH} or its checkpoint (another table test moved LATEST?)")
    for s in (1, 2):
        if not list(tdir.glob(f"stats_stand_net_s{s}__*.json")):
            raise Infra(f"table dir {tdir}: no stats_stand_net_s{s}__*.json")
    L.write_json(rec_p, {"dir": str(tdir), "launched": t0, "recorded_at": L.now()})
    return f"table {tdir.name}"


def step_sh_rows():
    h = need(STAND / "sh_head.json", "the SH head")
    st = spec().stage("sh")
    db = str(st.get("deadband", "0.25"))
    done = []
    for which, key, kind, nm in (("hrows", "height_rows", "manifest", SH), ("k5rows", "k5_rows", "zero", SHZ)):
        rows = scen_rows(st[key])
        todo = [f"{sc}@{es}" for sc, es, _ in rows if not complete_stamp(nm, sc, es)]
        if not todo:
            say(f"SKIP SH {which}: complete")
            continue
        guard(f"sh_eval_rows.sh ({which})")
        rc = run(["bash", SH_SCRIPTS / "sh_eval_rows.sh", "--deadband", db, nm, "stand_net", kind, "armed", h["ckpt"],
                  "1", *todo], child_env({**walker_env(), "P2_OUT": P2_OUT}), LOGS / f"sh.{which}.log",
                 f"SH {which} {todo}")
        miss = [f"{sc}@{es}" for sc, es, _ in rows if not complete_stamp(nm, sc, es)]
        if rc != 0 or miss:
            raise Infra(f"sh_eval_rows.sh ({which}) rc={rc}; missing {miss}")
        done.append(which)
    return f"SH rows {done or 'complete (skipped)'}"


def step_rh_rows():
    """The lazy R_h control beside TDR (DESIGN 3.6), run up front as reversible work; read only if an R_h-derived
    line fails."""
    rows = scen_rows(spec().stage("sh")["height_rows"])
    todo = [f"{sc}@{es}" for sc, es, _ in rows if not complete_stamp(RH, sc, es)]
    if not todo:
        return "R_h control rows complete (skipped)"
    guard("R_h beside TDR")
    rc = run(["bash", SH_SCRIPTS / "sh_eval_rows.sh", RH, "r_h", "manifest", "armed", "-", "-", *todo],
             child_env({**walker_env(), "P2_OUT": P2_OUT}), LOGS / "sh.rh_control.log",
             f"lazy R_h control beside TDR {todo}")
    miss = [f"{sc}@{es}" for sc, es, _ in rows if not complete_stamp(RH, sc, es)]
    if rc != 0 or miss:
        raise Infra(f"R_h control rc={rc}; missing {miss}")
    return "R_h control rows 4"


def step_sh_read():
    h = need(STAND / "sh_head.json", "the SH head")
    tab = need(STAND / "sh_table.json", "the SH table test")["dir"]
    st = spec().stage("sh")
    miss = [f"{nm}:{sc}@{es}" for key, nm in (("height_rows", SH), ("k5_rows", SHZ))
            for sc, es, _ in scen_rows(st[key]) if not complete_stamp(nm, sc, es)]
    if miss:
        raise Infra(f"SH rows missing {miss}")
    rspec = pp().sh_read_spec(spec(), st, SH, Path(tab), REFS)
    L.write_json(STAND / "sh_read_spec.json", rspec)
    env = child_env({**walker_env(), "P2_OUT": P2_OUT})
    t0 = time.time()
    rcr = run([L.PY, "-I", SH_SCRIPTS / "sh_read.py", "--spec", STAND / "sh_read_spec.json", "--json",
               STAND / "sh_read.json", "--md", STAND / "sh_read.md"], env, LOGS / "sh.read.log", "sh_read")
    rck = run([L.PY, "-I", SN_SCRIPTS / "sn_calib.py", "--head", SHZ, "--json", STAND / "sh_k5_calib.json", "--md",
               STAND / "sh_k5_calib.md"], env, LOGS / "sh.read.log", "sn_calib (K5)")
    for p in ("sh_read.json", "sh_k5_calib.json"):
        if fresh_json(STAND / p, t0) is None:
            raise Infra(f"{p} not written (sh_read rc={rcr}, sn_calib rc={rck})")
    rec = {"stage": "sh", "name": SH, "ckpt": h["ckpt"], "sha256": h["sha256"], "refs": REFS,
           "read": str(STAND / "sh_read.json"), "k5_calib": str(STAND / "sh_k5_calib.json"), "table": tab,
           "train_log": h["train_log"], "reader_rc": {"sh_read": rcr, "sn_calib_k5": rck}}
    res = gate("sh", "D2", rec)
    say(f"GATE sh {res['verdict']}: {len(res['decisive'])} decisive, halt {res['halt_lines']}")
    if res["verdict"] != "HALT":
        return f"SH gate {res['verdict']}"
    # the K5/G5 body-height copies re-anchor on the R / HOLD rows of sh_k5_calib (the user's 11:53 decision), then
    # the R_h-derived lines on the lazy R_h control beside TDR (DESIGN 3.6); every failed line must be covered
    bh = body_height_reanchor(res, L.read_json(STAND / "sh_k5_calib.json"), SHZ)
    remaining = [i for i in res["halt_lines"] if i not in bh["covered"]]
    lazy_dec = None
    lines = {ln["id"]: ln for ln in thresholds().lines}
    if remaining and any(rh_derived({"id": i}, lines) for i in remaining):
        rows = scen_rows(st["height_rows"])
        if not all(complete_stamp(RH, sc, es) for sc, es, _ in rows):
            raise L.Stop(f"SH gate HALT on {remaining}; the lazy R_h control rows are not complete", 4)
        lspec = json.loads(json.dumps(rspec))
        lspec["roles"]["R_h"]["name"] = RH
        L.write_json(STAND / "sh_lazy_rh_read_spec.json", lspec)
        t1 = time.time()
        rcl = run([L.PY, "-I", SH_SCRIPTS / "sh_read.py", "--spec", STAND / "sh_lazy_rh_read_spec.json", "--json",
                   STAND / "sh_lazy_rh_read.json", "--md", STAND / "sh_lazy_rh_read.md"], env, LOGS / "sh.read.log",
                  "sh_read (lazy R_h beside TDR)")
        lazy = fresh_json(STAND / "sh_lazy_rh_read.json", t1)
        if lazy is None:
            raise Infra(f"sh_lazy_rh_read.json not written (rc={rcl})")
        lazy_dec = lazy_decide(res, lazy, SH, ids=remaining)
        lazy_dec.update({"control": RH, "lazy_read": str(STAND / "sh_lazy_rh_read.json")})
        remaining = [i for i in remaining if i not in lazy_dec["covered"]]
    dec = {"stage": "sh", "halt_lines": res["halt_lines"], "body_height": bh, "lazy_rh": lazy_dec,
           "remaining": remaining, "reanchored": not remaining, "decided_at": L.now()}
    L.write_json(STAND / "gates" / "sh_reanchor.json", dec)
    say(f"REANCHOR sh: body height covered {len(bh['covered'])}, lazy R_h covered "
        f"{len((lazy_dec or {}).get('covered') or [])}, remaining {remaining}")
    if remaining:
        raise L.Stop(f"SH gate HALT; not re-anchored: {remaining} (the user's call: STAND_REPAIR.md section 4)", 4)
    return "SH gate HALT -> re-anchored (body height on R / HOLD; R_h-derived lines on the lazy R_h control)"


STEPS = {"refs": step_refs, "sn_train": step_sn_train, "sn_rows": step_sn_rows, "sn_read": step_sn_read,
         "sh_train": step_sh_train, "sh_table": step_sh_table, "sh_rows": step_sh_rows, "rh_rows": step_rh_rows,
         "sh_read": step_sh_read}


# ============================================================================================ pre-flight (CPU)
def same_gate(res: dict, ref: dict) -> list[str]:
    """Verdict and every decisive / watch line (id, value, line, pass) equal to a recorded gate."""
    bad = []
    if res.get("verdict") != ref.get("verdict"):
        bad.append(f"verdict {res.get('verdict')} != {ref.get('verdict')}")
    for key in ("decisive", "watch_lines"):
        a = {d["id"]: d for d in res.get(key) or []}
        b = {d["id"]: d for d in ref.get(key) or []}
        if set(a) != set(b):
            bad.append(f"{key} ids differ: {sorted(set(a) ^ set(b))[:6]}")
        for i in sorted(set(a) & set(b)):
            for f in ("value", "line", "pass"):
                if a[i].get(f) != b[i].get(f):
                    bad.append(f"{key} {i} {f} {a[i].get(f)} != {b[i].get(f)}")
    return bad


def clause_map(doc: dict, path: tuple) -> dict:
    """{(gate, role, clause): clause} under ``path``: sn_calib's rows.<head>.gates.<G>.clauses, or sh_read's
    gates.<K>.<role>.clauses."""
    d = doc or {}
    for k in path:
        d = (d or {}).get(k) or {}
    out = {}
    for g, gv in d.items():
        if not isinstance(gv, dict):
            continue
        if isinstance(gv.get("clauses"), dict):
            out.update({(g, "", k): c for k, c in gv["clauses"].items()})
            continue
        for r, rv in gv.items():
            if isinstance(rv, dict) and isinstance(rv.get("clauses"), dict):
                out.update({(g, r, k): c for k, c in rv["clauses"].items()})
    return out


def same_clauses(new: dict, ref: dict, path: tuple, skip_gates: tuple = ()) -> list[str]:
    """Every clause's value / line / ok equal to a recorded read (gates in ``skip_gates`` left out)."""
    a = {k: v for k, v in clause_map(new, path).items() if k[0] not in skip_gates}
    b = {k: v for k, v in clause_map(ref, path).items() if k[0] not in skip_gates}
    bad = [f"clause set differs: {sorted(set(a) ^ set(b))[:6]}"] if set(a) != set(b) else []
    if not b:
        bad.append("the recorded read has no clauses under this path")
    for k in sorted(set(a) & set(b)):
        for f in ("value", "line", "ok"):
            if (a[k] or {}).get(f) != (b[k] or {}).get(f):
                bad.append(f"{k} {f} {(a[k] or {}).get(f)} != {(b[k] or {}).get(f)}")
    return bad


def f19_walker_env() -> dict:
    sp = {"S1_CKPT": str(TSTAR_CKPT), "S1_SHA256": TSTAR_SHA, "S1_AGENT_YAML": str(TSTAR_CKPT.parent / "params/agent.yaml"),
          "REFS_ENV_STAMP": str(TSTAR_DRIVER), "CANDIDATES": str(L.T4_MAIN / "F19/candidates.yaml"),
          "OVERLAY_CHAIN": spec().chain(RUN, "p2a"), "S1_OVERLAY": spec().overlay("p2a"), "REFS_NAME": "REFS_T4F19"}
    return {"SN_WALKER_SPEC": json.dumps(sp, sort_keys=True), "SN_WALKER_CK": str(TSTAR_CKPT),
            "SN_WALKER_ST": str(TSTAR_DRIVER), "P2_REANCHOR": "1"}


def preflight():
    out = STAND / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    rep = {"at": L.now()}
    # P4: static checks
    dm = L.read_json(TSTAR_DRIVER / "run_meta.json") or {}
    names = {n: [str(p) for p in (LROUT / n, SN_OUT / "runs" / n, SH_OUT / "runs" / n) if p.exists()] for n in NAMES}
    # the refs step may already have run (the pre-flight runs beside it): REFS_RSD_TDR may exist, holding TDR's only
    refs_meta = [L.read_json(p) or {} for p in (LROUT / REFS).glob("*/seed*/*/run_meta.json")]
    if names[REFS] and refs_meta and all(m.get("checkpoint_sha256") == TDR_SHA for m in refs_meta):
        names[REFS] = []
    same_tree = {}
    for d in (SN_SCRIPTS, SH_SCRIPTS, P2_SCRIPTS, L.PIPE):
        rel = d.relative_to(L.W)
        diff = [str(p.relative_to(d)) for p in sorted(d.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
                and (not (L.MAIN / rel / p.relative_to(d)).is_file()
                     or (L.MAIN / rel / p.relative_to(d)).read_bytes() != p.read_bytes())]
        same_tree[str(rel)] = diff
    p4 = {"tdr_sha": L.sha256_file(TDR_CKPT) == TDR_SHA,
          "tdr_agent_yaml": (TDR_CKPT.parent / "params/agent.yaml").is_file(),
          "tstar_ckpt_sha": L.sha256_file(TSTAR_CKPT) == TSTAR_SHA,
          "driver_is_tstar_walkplay_0kg": (dm.get("task") == WALK_PLAY and dm.get("checkpoint_sha256") == TSTAR_SHA
                                           and dm.get("payload") is None),
          "driver_env_eq_tdr_l000": env_vars(TSTAR_DRIVER) == env_vars(TDR_L000) and bool(env_vars(TSTAR_DRIVER)),
          "driver_env_keys": len(env_vars(TSTAR_DRIVER)),
          "no_dr_stress_payload_keys": not any(k.startswith(("KRABBY_DR_", "KRABBY_STRESS_", "KRABBY_PAYLOAD_"))
                                               for k in env_vars(TSTAR_DRIVER)),
          "names_absent": names,
          "candidates_identical_to_main": (L.T4 / "F19/candidates.yaml").read_bytes()
          == (L.T4_MAIN / "F19/candidates.yaml").read_bytes(),
          "thresholds_path": str(spec().thresholds_path()), "thresholds_is_v1_2": spec().thresholds_path().resolve()
          == L.TH_PATH.resolve(),
          "scripts_differ_from_main": same_tree, "code_state": L.code_state(), "pin": L.launch_pin(),
          "disk": L.disk_problems(L.MAIN / "parkour/outputs") or "ok",
          "walker_spec_template": walker_spec_doc(Path("<REFS(TDR) suite@1 stamp>"))}
    rep["P4"] = p4
    # P1: Task 4's stand_gate on F19's recorded inputs reproduces F19's gates
    sn_rec = {"stage": "sn", "name": F19_SN, "calib": str(F19_STAND / "sn_calib.json")}
    sh_rec = {"stage": "sh", "name": F19_SH, "read": str(F19_STAND / "sh_read.json"),
              "k5_calib": str(F19_STAND / "sh_k5_calib.json")}
    r1 = gate("sn", "D1", sn_rec, out_name="replay_f19_sn")
    r2 = gate("sh", "D2", sh_rec, out_name="replay_f19_sh")
    rep["P1"] = {"sn": same_gate(r1, L.read_json(F19_GATES / "sn.json")) or "equal",
                 "sh": same_gate(r2, L.read_json(F19_GATES / "sh.json")) or "equal",
                 "sn_verdict": r1["verdict"], "sh_verdict": r2["verdict"], "sh_halt": r2["halt_lines"]}
    # P2: the worktree's readers on F19's stamps (P2_OUT = main, F19's walker spec) reproduce F19's reads
    env = child_env({**f19_walker_env(), "P2_OUT": P2_OUT})
    # sh_read's own K5 roll-up (gates.K5) is left out of the lazy-read comparison: neither the gate (K5 comes from
    # sn_calib on the head's z rows) nor the re-anchor (K6) reads it, and F19's HAND-RUN lazy read scored it against
    # S1's REFS (its S1-era controls PASS, the head INVALID) while F19's own pipeline read and this replay score it
    # against REFS_T4F19 (controls INVALID, head PASS); found 2026-11-09 10:35, see lazy_k5_vs_f19_original
    jobs = [("sn_calib", [SN_SCRIPTS / "sn_calib.py", "--head", F19_SN], "sn_calib.json", ("rows", F19_SN, "gates"),
             ()),
            ("sh_read", [SH_SCRIPTS / "sh_read.py", "--spec", F19_STAND / "sh_read_spec.json"], "sh_read.json",
             ("gates",), ()),
            ("sh_k5_calib", [SN_SCRIPTS / "sn_calib.py", "--head", F19_SH + "z"], "sh_k5_calib.json",
             ("rows", F19_SH + "z", "gates"), ()),
            ("sh_lazy_rh_read", [SH_SCRIPTS / "sh_read.py", "--spec", F19_STAND / "sh_lazy_rh_read_spec.json"],
             "sh_lazy_rh_read.json", ("gates",), ("K5",))]
    p2 = {}
    for tag, cmd, ref, path, skip in jobs:
        t0 = time.time()
        rc = run([L.PY, "-I", *cmd, "--json", out / f"{tag}.json", "--md", out / f"{tag}.md"], env,
                 LOGS / "preflight.log", f"P2 {tag}")
        new = fresh_json(out / f"{tag}.json", t0)
        p2[tag] = {"rc": rc, "skipped_gates": list(skip),
                   "diff": (same_clauses(new, L.read_json(F19_STAND / ref), path, skip) or "equal")
                   if new is not None else "NOT WRITTEN"}
    lz = L.read_json(out / "sh_lazy_rh_read.json")
    if lz is not None:
        a = {k: v for k, v in clause_map(lz, ("gates",)).items() if k[0] == "K5"}
        o = {k: v for k, v in clause_map(L.read_json(F19_STAND / "sh_read.json"), ("gates",)).items() if k[0] == "K5"}
        p2["lazy_k5_vs_f19_original"] = {
            "rc": 0, "diff": [f"{k} {(a[k] or {}).get('value')} != {(o.get(k) or {}).get('value')}" for k in sorted(a)
                              if (a[k] or {}).get("value") != (o.get(k) or {}).get("value")] or "equal"}
    rep["P2"] = p2
    # P1 on the replayed reads, and P3: the lazy decision on F19's HALT
    if all(p2[t]["diff"] == "equal" for t in ("sh_read", "sh_k5_calib")):
        r3 = gate("sh", "D2", {**sh_rec, "read": str(out / "sh_read.json"), "k5_calib": str(out / "sh_k5_calib.json")},
                  out_name="replay_f19_sh_rereads")
        rep["P1_on_P2"] = same_gate(r3, L.read_json(F19_GATES / "sh.json")) or "equal"
    dec = lazy_decide(L.read_json(F19_GATES / "sh.json"), L.read_json(F19_STAND / "sh_lazy_rh_read.json"), F19_SH)
    rep["P3"] = {"reanchored": dec["reanchored"], "lines": dec["lines"], "not_rh_derived": dec["not_rh_derived"],
                 "expect": "reanchored True: R_h beside T* bin 4 es1 0.89 < 0.945; head 0.935 passes the re-anchored line"}
    ok = (p4["tdr_sha"] and p4["tstar_ckpt_sha"] and p4["driver_is_tstar_walkplay_0kg"] and p4["driver_env_eq_tdr_l000"]
          and p4["no_dr_stress_payload_keys"] and not any(names.values()) and p4["candidates_identical_to_main"]
          and p4["thresholds_is_v1_2"] and not any(same_tree.values()) and not p4["code_state"]["dirty"]
          and p4["disk"] == "ok" and rep["P1"]["sn"] == "equal" and rep["P1"]["sh"] == "equal"
          and all(v["diff"] == "equal" for v in p2.values())
          and rep.get("P1_on_P2") == "equal" and dec["reanchored"] is True)
    rep["ok"] = bool(ok)
    L.write_json(STAND / "preflight.json", rep)
    say(f"PREFLIGHT ok={rep['ok']} P1 sn={rep['P1']['sn']} sh={rep['P1']['sh']} P2={ {k: v['diff'] if isinstance(v['diff'], str) else len(v['diff']) for k, v in p2.items()} } P1_on_P2={rep.get('P1_on_P2')} P3={dec['reanchored']}")
    if not ok:
        raise L.Stop("pre-flight failed (results/stand/preflight.json)", 4)
    return "pre-flight ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    s = sub.add_parser("step")
    s.add_argument("step", choices=list(STEPS))
    sub.add_parser("walker")
    a = ap.parse_args()
    if a.cmd == "walker":
        print(json.dumps(walker_env(), indent=1, sort_keys=True))
        return 0
    tag = a.step if a.cmd == "step" else "preflight"
    try:
        note = (STEPS[a.step] if a.cmd == "step" else preflight)() or ""
        rc = 0
    except L.Stop as e:
        rc, note = 4, f"STOP {e.why}"
    except Infra as e:
        rc, note = 1, f"INFRA {e}"
    except Exception as e:  # noqa: BLE001 -- recorded as an infra failure with its traceback
        traceback.print_exc()
        rc, note = 1, f"INFRA {type(e).__name__}: {e}"
    print(f"RSDST {tag} END rc={rc} {note}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
