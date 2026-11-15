#!/usr/bin/env python3
"""P2 step 9 (p2/P2.md): the stand re-pair -- RSD's STAND_REPAIR.md recipe with ONLY the walker swapped, TDR -> c1
(RT_s19 ``model_29500``, sha256 6d1f29be...). Task 4's two stand stages re-run beside c1:

  * REFS(c1)  the walker alone, 9 rows (suite@1,2 walkstop@1,2 field_hard@1-4 field_turn@1; pipeline.yaml refs_rows)
  * SN        ``sn_dagger.sh`` DAgger of R beside c1, 4000 its, judged ``model_4000.pt``; its 9 stand rows; D1
  * SH        ``sh_dagger.sh`` DAgger of R_h warm-started from SN, 5000 its, judged ``model_5000.pt``; the table test;
              the height rows + the K5 zero-command rows; the lazy R_h control beside c1 (DESIGN 3.6, 4 rows, run up
              front as reversible work); D2

    p2_stand.py preflight [--jobs N] [--no-readers]   STAND_REPAIR.md section 5 P1-P4 (CPU) re-pointed for P2, plus
                                                     the RSD-record replays and the body-height report line (P5);
                                                     -> results/p2/stand/preflight.json (``ok``); sn_train refuses
                                                     without a FRESH ok pre-flight (same pin, same p2_stand.py bytes)
    p2_stand.py step STEP                            one queue step (p2_stand_queue.sh runs them in order):
                                                     preflight refs sn_train sn_rows sn_read sh_train sh_table
                                                     sh_rows rh_rows sh_read  (+ bh: the body-height report, CPU)
    p2_stand.py bh                                   the body-height report line (CPU): measured on REFS(c1) once
                                                     its four suite / walkstop rows exist, predicted before
    p2_stand.py walker                               print the walker env the steps pass (after ``refs``)
    p2_stand.py dry                                  the plan and every input / name / pin / disk / stop-file check;
                                                     reads only (no lock probe, no write, no launch); exit 0 / 1
    p2_stand.py selftest --out DIR                   CPU checks into DIR only: the K4 test on RSD's recorded
                                                     numbers, the RSD gate / re-anchor / lazy / K4 decision replays,
                                                     F19's gate and lazy replays (P1 / P3), synthetic K4 / body-height
                                                     cases; exit 0 iff every check passes

A step prints ``P2ST <step> END rc=<rc> <why>`` last: rc 0 done; 4 a decide-record-notify stop (a gate HALT, a premise
STOP, a guard, a STOP_QUEUE file); 1 an infrastructure failure. Every step resumes: finished rows, runs and reads are
skipped. ``P2ST NOTIFY ...`` lines are for the watcher (push-notify).

REUSE (nothing copied that can be imported). ``2026-11-07_rod_speed_dr/scripts/rsd_stand.py`` (= R) and its
``rsd_lib`` (= L) are imported and RE-POINTED by attribute (``configure``); their files stay byte-unchanged. Kept from
R as is: Task 4's ``pipeline.py`` library (stand_gate, sh_read_spec, clean_env, Spec), ``train`` (the DAgger runner:
krabby_env-vs-F19 check, Task 4's 30-min watchdog, the premise read at N/2, SIGTERM-only stops), ``walker_problems``,
``walker_spec_doc`` / ``walker_env``, ``body_height_reanchor`` (the user's 2026-11-09 11:53 rule), ``lazy_decide``
(DESIGN 3.6), ``rh_derived``, ``complete_stamp``, ``scen_rows``, ``same_gate`` / ``same_clauses``, the steps
``sn_rows`` / ``sh_train`` / ``sh_table`` / ``sh_rows``. ``p2_step3`` (= S3) gives the P2 pin check
(``p2_pin_problems``) and ``raw_problems`` (Left 11); ``scripts/rt_machinery`` (= M) the build check against
results/p2/pin.json (Left 7/24: rsd_lib.check_tree read RESULTS/launch.json) and the Isaac-busy patterns.

WHAT IS RE-POINTED (Left 7):
  * the walker: R.TDR_CKPT / R.TDR_SHA -> c1 (the worktree run dir, the path the distill and D3 use); names
    REFS_RT_6d1f29be, SN_RT_6d1f29be_sn_s19, SH_RT_6d1f29be_sh_s19 (+ ``z``), SH_RH_RT_6d1f29be (each checked absent);
  * the guard: results/p2/pin.json (code 89c1916b / eval 9fe964d7, the frozen campaign files, a clean tree) instead
    of rsd_lib.check_tree's results/launch.json; >= 30 GB free; DerivedDataCache <= 78 GB; the STOP_QUEUE files;
    then wait while any Isaac process runs (gate_row.sh's patterns, plus rt_machinery's rt_kit_* entry points);
  * the records: results/p2/stand/ (walker spec, heads, premise reads, CPU reads, gates, decisions, pre-flight);
  * the logs: <MAIN>/parkour/outputs/omni_plan/rough_terrain_p2/stand/ (step logs, the queue log, STOP_QUEUE); the
    SH table test dir: .../rough_terrain_p2/stand/sh_table (its own LATEST, no shared-LATEST race);
  * the gate labels ("RT c1 ...", never "RSD_TDR").
  NOT re-pointed, on purpose: the rows / run dirs stay in the main checkout's phase2/lr, stand_network/runs and
  stand_height/runs (R.P2_OUT / SN_OUT / SH_OUT), exactly as F19's and RSD's: that is where sn_calib / sh_read find
  the S1-era controls (REFERENCES "$P2_OUT/lr/<NAME>"), and STAND_REPAIR.md P2 replays F19's reads with P2_OUT = main.
  The new names never collide (checked absent before anything launches).

c1 UNDER MAIN logs/rsl_rl (the plan's deferred hardlink): NOT needed. Every stand launcher takes the walker as an
absolute path (SN_WALKER_SPEC S1_CKPT / SN_WALKER_CK, run_arm --row-eval <checkpoint>, sh_record_video --checkpoint);
the only logs/rsl_rl-relative paths are their historical S1 defaults, never used under the override. RSD ran TDR the
same way from a worktree run dir. So the walker stays at the worktree path; ``dry`` / the pre-flight report whether a
MAIN copy exists (report only).

DECISION RULES (p2/P2.md "Decision rules", Task 4's otherwise; STAND_REPAIR.md section 4):
  * SN D1 (35 decisive + 8 WATCH) PASS / WATCH -> SH. HALT: the body-height lines re-anchor on R and HOLD (the
    11:53 precedent, settled 6e: ``body_height_reanchor``); any other failed line -> STOP for the R / HOLD lazy
    controls or the user (their command is worked out then, as in RSD).
  * SH D2 (101 decisive) PASS / WATCH -> done. HALT: (1) the K5/G5 body-height copies re-anchor on R / HOLD; (2) the
    R_h-derived lines on the lazy R_h control beside c1 (``lazy_decide``); (3) the K4 restore-miss lines (D9's fixed
    <= 3): ACCEPT AND DISCLOSE when the head's restore misses are at chance against the lazy R_h beside c1, pooled over
    the env seeds, by K4_TEST below at p >= 0.05; otherwise HALT (settled 6f, the 18:10 terms). Anything left -> STOP,
    the user's call. A K4 acceptance writes results/p2/stand/gates/sh_decision.json (both counts and p) and a NOTIFY.
  * The body-height report line (Left 15): c1 stands lower than TDR (G5 estimator ~1.107 vs 1.1205 m); if REFS(c1)
    sits within 0.03 m of a control (REFS <= ~1.095), the controls PASS the line and a head that fails it HALTs
    (no re-anchor). Reported by the pre-flight (predicted from c1's own suite rows) and after ``refs`` (measured);
    report only, never a stop (the gate decides).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent                    # 2026-11-10_rough_terrain/p2
CAMP = HERE.parent
EXPS = CAMP.parent
RSD_CAMP = EXPS / "2026-11-07_rod_speed_dr"
RSD_SCRIPTS = RSD_CAMP / "scripts"
MAIN = Path("/home/nickmagus/krabby/krabby-research")
P2_OUT = MAIN / "parkour/outputs/omni_plan/rough_terrain_p2"
RT_OUT = MAIN / "parkour/outputs/omni_plan/rough_terrain"
RESULTS_P2 = CAMP / "results" / "p2"

# rsd_lib reads these two at import: point them at P2 first (a caller's different value is refused in main())
_CALLER = {k: os.environ.get(k) for k in ("RSD_RESULTS", "RSD_TP_OUT")}
os.environ["RSD_RESULTS"] = str(RESULTS_P2)
os.environ["RSD_TP_OUT"] = str(P2_OUT)

sys.path.insert(0, str(HERE))
import p2_step3 as S3  # noqa: E402  (the P2 pin check, raw_problems; sets RT_TP_OUT = rough_terrain_p2)

M = S3.M                                                 # scripts/rt_machinery (build check, busy patterns)
sys.path.insert(0, str(RSD_SCRIPTS))
import rsd_lib as L  # noqa: E402
import rsd_stand as R  # noqa: E402

assert Path(L.__file__).resolve().parent == RSD_SCRIPTS and Path(R.__file__).resolve().parent == RSD_SCRIPTS
assert L.W == M.W, f"rsd_lib.W {L.W} != rt_machinery.W {M.W}"
assert L.RESULTS == RESULTS_P2 and L.TP_OUT == P2_OUT, (L.RESULTS, L.TP_OUT)

# ============================================================================================ the walker swap
C1_REL = Path("parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19/model_29500.pt")
C1_CK = L.W / C1_REL                                     # the path the distill loads and D3 compares (p2_queue.sh)
C1_MAIN_CK = MAIN / C1_REL                               # the deferred hardlink: report only, never used
C1_SHA = "6d1f29be8bf4504fd53fd20018dbab38ff60c6ccb32d7bd885f3ab15eaaed3c1"
assert C1_SHA == S3.C1_SHA
C1_SUITE_SLOTS = {1: RT_OUT / "lr/RTSW_6d1f29be_R_suite/suite__omni/seed001",   # c1's own 0-kg suite rows (RT, at
                  2: RT_OUT / "lr/RTA_6d1f29be_R_suite/suite__omni/seed002"}    # the RT pin): env + height inputs
TAG = f"RT_{C1_SHA[:8]}"
REFS = f"REFS_{TAG}"
SN = f"SN_{TAG}_sn_s19"
SH = f"SH_{TAG}_sh_s19"
SHZ = SH + "z"
RH = f"SH_RH_{TAG}"
NAMES = (REFS, SN, SH, SHZ, RH)
GATE_RUN_LABEL = f"RT c1 {C1_SHA[:8]} (Task 4's sn / sh recipe, walker c1 = RT_s19 model_29500; P2 step 9)"

# ============================================================================================ P2 roots
STAND = RESULTS_P2 / "stand"                              # records
LOGS = P2_OUT / "stand"                                   # step logs, queue log, STOP_QUEUE
SH_TABLE_OUT = LOGS / "sh_table"
QUEUE_STOP = LOGS / "STOP_QUEUE"
STOP_FILES = (QUEUE_STOP, RESULTS_P2 / "STOP_QUEUE", CAMP / "results" / "STOP_QUEUE")
#: the launchers' own STOP files (run_arm $P2_OUT = main phase2; sn/sh_eval_rows check THIS worktree's phase2 /
#: stand_height / stand_network; the trainers their SN_OUT / SH_OUT): any of them present refuses a row -> checked
LAUNCHER_STOP_FILES = tuple(p / "STOP_QUEUE" for p in (
    R.P2_OUT, R.SN_OUT, R.SH_OUT, L.W / "parkour/outputs/omni_plan/phase2",
    L.W / "parkour/outputs/omni_plan/stand_network", L.W / "parkour/outputs/omni_plan/stand_height"))
PREFLIGHT_VERSION = 1

# ============================================================================================ RSD's records (replays)
RSD_STAND = RSD_CAMP / "results" / "stand"
RSD_SN, RSD_SH, RSD_RH, RSD_REFS = "SN_RSD_TDR_sn_s19", "SH_RSD_TDR_sh_s19", "SH_RH_RSD_TDR", "REFS_RSD_TDR"

# ============================================================================================ K4 (settled 6f)
K4_ALPHA = 0.05
K4_TEST = ("exact conditional binomial test, two-sided: the head's restore misses H among the pooled misses H + C "
           "(env seeds 1 + 2) against Bin(H + C, n_H / (n_H + n_C)), n = the restore-required hand-backs per role; "
           "two-sided p = the sum of P(k) <= P(H) * (1 + 1e-7) (scipy.stats.binomtest's rule). It is the statistic "
           "of RSD's stand record (STAND_REPAIR.md 17:34, the sh-k4-halt-verify check; results/stand/gates/"
           "sh_decision.json 'pooled 11 vs 9 of 1600, two-sided p = 0.82'): it reproduces all three recorded p "
           "values, 11 vs 9 -> 0.82, 11 vs 5 -> 0.21, R18's 23 vs 10 -> 0.035 (Fisher's exact test gives 0.034 on "
           "the last, so it is not the recorded test)")
#: RSD's recorded K4 statistics (STAND_REPAIR.md 17:34): (head misses, control misses, n per role, recorded p)
K4_RECORDED = [(11, 9, 1600, 0.82, 2), (11, 5, 1600, 0.21, 2), (23, 10, 1600, 0.035, 3)]
K4_MISS_RE = re.compile(r"^miss es(\d+) <= ")
K4_N_RE = re.compile(r"^(\d+) restore-required exits")

# ============================================================================================ body height (Left 15)
BH_TOL = 0.03
BH_PLAN_LINE = 1.095        # p2/P2.md Left 15: "if REFS(c1) reads <= ~1.095 the controls pass and the head fails"
BH_KNIFE = 0.005            # a spare under this is a knife edge: NOTIFY
BH_KEYS = ("suite es1", "walkstop es1", "suite es2", "walkstop es2")
BH_CONTROLS = R.BH_CONTROLS  # ("R", "HOLD")


def say(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def notify(msg: str) -> None:
    print(f"P2ST NOTIFY {msg}", flush=True)


def sha256_file(p) -> str:
    return L.sha256_file(Path(p))


# ============================================================================================ configure (re-point R)
def configure(stand: Path = STAND, logs: Path = LOGS) -> None:
    """Re-point rsd_stand's module globals (its functions read them at call time). ``stand`` / ``logs`` move only for
    the self-test (into its --out dir); a real run uses the P2 roots."""
    global STAND, LOGS, SH_TABLE_OUT
    STAND, LOGS = Path(stand), Path(logs)
    SH_TABLE_OUT = LOGS / "sh_table"
    R.STAND, R.LOGS, R.SH_TABLE_OUT = STAND, LOGS, SH_TABLE_OUT
    R.TDR_CKPT, R.TDR_SHA = C1_CK, C1_SHA
    R.REFS, R.SN, R.SH, R.SHZ, R.RH, R.NAMES = REFS, SN, SH, SHZ, RH, NAMES
    R.TDR_L000 = None                                     # RSD's driver-env reference: not used here (P4 below)
    R.guard = guard
    R.gate = gate
    R.preflight = _no_rsd_preflight
    R.STEPS = STEPS
    L.launch_pin = p2_pin                                 # any rsd_lib.check_tree call reads the P2 pin


def _no_rsd_preflight(*_a, **_k):
    raise RuntimeError("rsd_stand.preflight is RSD's (TDR constants); use p2_stand.py preflight")


# ============================================================================================ pin, stops, guard
def p2_pin() -> dict | None:
    return M.read_pin(S3.P2_PIN)


def build_problems() -> list[str]:
    """build_check.py check --pin-path results/p2/pin.json, in process: the P2 pin exists, is rt_pin_v1 and carries
    results/pin.json's code / eval (S3.p2_pin_problems); the live code_sha / eval_sha equal it, the code + eval tree
    is clean, the frozen campaign files are unchanged (M.pin_problems); eval_sha is 9fe964d7."""
    bad, _warn = S3.p2_pin_problems()
    pin = p2_pin()
    b2, _w2 = M.pin_problems(pin, M.live_build(), M.campaign_files(M.FROZEN_GLOBS),
                             M.campaign_files(M.REPORTED_GLOBS))
    bad += b2
    if pin and pin.get("eval_sha") != L.EVAL_SHA:
        bad.append(f"P2 pin eval_sha {str(pin.get('eval_sha'))[:12]} != {L.EVAL_SHA[:12]}")
    return bad


def stop_file() -> Path | None:
    for p in STOP_FILES:
        if p.exists():
            return p
    return None


def launcher_stops() -> list[str]:
    return [str(p) for p in LAUNCHER_STOP_FILES if p.exists()]


def isaac_busy() -> list[str]:
    """gate_row.sh's patterns (RSD's guard) plus rt_machinery's (distill.sh's + rt_kit_*): ``ps -ww``, never pgrep."""
    out = list(R.isaac_busy())
    try:
        out += [a for _pid, a in M.busy(exclude=(os.getpid(), os.getppid()))]
    except RuntimeError as e:                             # distill.sh's pattern unreadable: fail closed
        out.append(f"(busy pattern unreadable: {e})")
    return sorted(set(out))


def guard(what: str) -> None:
    """Before every Isaac launch (replaces rsd_stand.guard): no STOP_QUEUE file; the P2 pin build check; >= 30 GB free
    and the DerivedDataCache <= 78 GB (moving it aside is the parent's job: a stop, never a prune); no launcher STOP
    file; then wait while any other Isaac process runs (a STOP file appearing while waiting stops)."""
    sf = stop_file()
    if sf:
        raise L.Stop(f"{sf} exists (stopped before {what})", 4)
    bad = build_problems()
    if bad:
        raise L.Stop(f"build / pin (results/p2/pin.json): {bad[:4]}", 4)
    disk = L.disk_problems(MAIN / "parkour/outputs")
    if disk:
        raise L.Stop("; ".join(disk) + " -- the DerivedDataCache move is the parent's job (renamed, never deleted)", 4)
    ls = launcher_stops()
    if ls:
        raise L.Stop(f"launcher STOP_QUEUE file(s) present, a row would refuse: {ls}", 4)
    waited = False
    while True:
        busy = isaac_busy()
        if not busy:
            break
        if not waited:
            say(f"WAIT {what}: another Isaac process runs: {busy[0][:200]}")
            waited = True
        sf = stop_file()
        if sf:
            raise L.Stop(f"{sf} appeared while waiting before {what}", 4)
        time.sleep(20)
    say(f"GUARD ok: {what}")


def stamp_pin_problems(stamp: Path) -> list[str]:
    return M.build_mismatch(M.stamp_build(stamp), p2_pin())


def name_stamps(name: str) -> list[Path]:
    return sorted(p.parent for p in (R.LROUT / name).glob("*/seed*/*/run_meta.json"))


def rows_pin_problems(name: str, rows: list) -> list[str]:
    """Every complete stamp a read uses (the newest per row, R.complete_stamp) was recorded at the P2 pin."""
    bad = []
    for sc, es, _ in rows:
        st = R.complete_stamp(name, sc, es)
        if st is None:
            bad.append(f"{name} {sc}@{es}: no complete stamp")
            continue
        bad += [f"{name} {sc}@{es}: {x}" for x in stamp_pin_problems(st)]
    return bad


# ============================================================================================ the gate (Task 4's code)
def gate(sid: str, which: str, rec: dict, out_name: str | None = None) -> dict:
    """pipeline.stand_gate (D1 = SN G1-G5 on sn_calib's head row; D2 = SH K1-K6 on sh_read + K5 on sn_calib of the
    head's z rows) with a thresholds-only ctx; its write_gate swapped for one writing <STAND>/gates (R.gate's body
    with the P2 label)."""
    mod = R.pp()
    th = R.thresholds()
    path = STAND / "gates" / f"{out_name or sid}.json"

    def write_gate(_ctx, sid_, res):
        tp = R.spec().thresholds_path()
        L.write_json(path, {"run": GATE_RUN_LABEL, "stage": sid_, "read_at": L.now(),
                            "thresholds": {"file": tp.name, "version": R.spec().d["thresholds"]["version"],
                                           "sha256": L.sha256_file(tp)}, **res})
        return path

    saved = mod.write_gate
    mod.write_gate = write_gate
    try:
        from types import SimpleNamespace  # noqa: PLC0415
        return mod.stand_gate(SimpleNamespace(thresholds=lambda: th), sid, which, rec, x=False)
    finally:
        mod.write_gate = saved


def th_lines() -> dict:
    return {ln["id"]: ln for ln in R.thresholds().lines}


def is_k4_miss(i: str, lines: dict) -> bool:
    ln = lines.get(i) or {}
    return ln.get("gate") == "K4" and bool(K4_MISS_RE.match(str(ln.get("clause") or "")))


# ============================================================================================ K4: the chance test
def binom_two_sided(k: int, n: int, p: float) -> float:
    """scipy.stats.binomtest(k, n, p).pvalue (alternative 'two-sided'), in log space (no overflow)."""
    if n <= 0:
        return 1.0
    if p <= 0.0 or p >= 1.0:
        return 1.0 if (k == 0 and p <= 0.0) or (k == n and p >= 1.0) else 0.0

    def lp(i):
        return math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * math.log(p) + \
            (n - i) * math.log1p(-p)
    pk = lp(k)
    thr = pk + math.log1p(1e-7)
    return min(1.0, sum(math.exp(lp(i)) for i in range(n + 1) if lp(i) <= thr))


def k4_counts(lazy: dict, role: str) -> dict:
    """{seed: {clause, misses, n}} from a sh_read's K4 clauses of ``role`` (n parsed from the reader's own note,
    '<n> restore-required exits: ...')."""
    cl = ((((lazy or {}).get("gates") or {}).get("K4") or {}).get(role) or {}).get("clauses") or {}
    out = {}
    for c, v in cl.items():
        m = K4_MISS_RE.match(c)
        if not m:
            continue
        nm = K4_N_RE.match(str((v or {}).get("note") or ""))
        out[int(m.group(1))] = {"clause": c, "misses": (v or {}).get("value"), "n": int(nm.group(1)) if nm else None}
    return out


def k4_chance(res: dict, lazy: dict, head: str, ids: list, lines: dict | None = None) -> dict:
    """The pre-registered K4 rule (settled 6f): the head's restore misses against the lazy R_h beside the walker,
    pooled over the env seeds, by K4_TEST; accept-and-disclose at p >= K4_ALPHA, else HALT. Reads the LAZY read
    (its R_h role is the control beside c1; its head role is the head's own rows) and checks the head's values
    there equal the gate's decisive values."""
    lines = lines if lines is not None else th_lines()
    h, c = k4_counts(lazy, head), k4_counts(lazy, "R_h")
    dec = {d["id"]: d for d in res.get("decisive") or []}
    problems = []
    seeds = sorted(set(h) | set(c))
    if not seeds or set(h) != set(c):
        problems.append(f"K4 seeds head {sorted(h)} != control {sorted(c)}")
    for s in seeds:
        for role, d in (("head", h), ("control", c)):
            v = d.get(s) or {}
            if not isinstance(v.get("misses"), (int, float)) or not v.get("n"):
                problems.append(f"{role} es{s}: misses {v.get('misses')!r} / n {v.get('n')!r} unreadable")
    for i in ids:
        ln = lines.get(i) or {}
        m = K4_MISS_RE.match(str(ln.get("clause") or ""))
        if not m:
            problems.append(f"{i}: not a K4 miss line")
            continue
        s = int(m.group(1))
        gv = (dec.get(i) or {}).get("value")
        lv = (h.get(s) or {}).get("misses")
        if gv != lv:
            problems.append(f"{i}: the gate's value {gv!r} != the lazy read's head value {lv!r}")
    H = sum(int((h.get(s) or {}).get("misses") or 0) for s in seeds)
    C = sum(int((c.get(s) or {}).get("misses") or 0) for s in seeds)
    NH = sum(int((h.get(s) or {}).get("n") or 0) for s in seeds)
    NC = sum(int((c.get(s) or {}).get("n") or 0) for s in seeds)
    p = binom_two_sided(H, H + C, NH / (NH + NC)) if (NH + NC) and not problems else None
    accept = p is not None and p >= K4_ALPHA
    return {"rule": "p2/P2.md Decision rules 'K4 fails' (settled 6f, the 18:10 terms): accept and disclose if the "
                    "head's restore misses are at chance against the lazy R_h beside c1, pooled over seeds; "
                    "otherwise HALT",
            "test": K4_TEST, "alpha": K4_ALPHA, "ids": list(ids), "per_seed": {"head": h, "control": c},
            "pooled": {"head_misses": H, "head_n": NH, "control_misses": C, "control_n": NC},
            "p_two_sided": p, "accept": bool(accept), "problems": problems,
            "covered": list(ids) if accept else [],
            "note": "two-sided, as recorded: a head significantly BETTER than the control also reads 'not at chance' "
                    "and HALTs for the user (the literal rule)"}


# ============================================================================================ decisions (pure)
def decide_sn(res: dict, calib: dict, head: str) -> dict:
    bh = R.body_height_reanchor(res, calib, head)
    remaining = [i for i in res["halt_lines"] if i not in bh["covered"]]
    return {"stage": "sn", "halt_lines": res["halt_lines"], "body_height": bh, "remaining": remaining,
            "reanchored": not remaining}


def sh_needs_lazy(res: dict, k5calib: dict, headz: str) -> bool:
    lines = th_lines()
    bh = R.body_height_reanchor(res, k5calib, headz)
    rem = [i for i in res["halt_lines"] if i not in bh["covered"]]
    return any(R.rh_derived({"id": i}, lines) or is_k4_miss(i, lines) for i in rem if i in lines)


def decide_sh(res: dict, k5calib: dict, lazy: dict | None, *, head: str, headz: str, control: str,
              lazy_path: str | None = None) -> dict:
    """RSD's step_sh_read order, then the K4 rule: (1) the K5/G5 body-height copies on R / HOLD; (2) the R_h-derived
    lines on the lazy R_h control (lazy_decide, called exactly as RSD called it, ids = what (1) left); (3) the K4
    restore-miss lines by k4_chance. ``remaining`` empty -> the stage passes (re-anchored / accepted)."""
    lines = th_lines()
    bh = R.body_height_reanchor(res, k5calib, headz)
    remaining = [i for i in res["halt_lines"] if i not in bh["covered"]]
    lazy_dec = None
    if remaining and any(R.rh_derived({"id": i}, lines) for i in remaining if i in lines):
        if lazy is None:
            return {"stage": "sh", "needs_lazy": True, "halt_lines": res["halt_lines"], "body_height": bh,
                    "remaining": remaining, "reanchored": False}
        lazy_dec = R.lazy_decide(res, lazy, head, ids=remaining)
        lazy_dec.update({"control": control, "lazy_read": lazy_path})
        remaining = [i for i in remaining if i not in lazy_dec["covered"]]
    k4 = None
    k4_ids = [i for i in remaining if is_k4_miss(i, lines)]
    if k4_ids:
        if lazy is None:
            return {"stage": "sh", "needs_lazy": True, "halt_lines": res["halt_lines"], "body_height": bh,
                    "lazy_rh": lazy_dec, "remaining": remaining, "reanchored": False}
        k4 = k4_chance(res, lazy, head, k4_ids, lines)
        remaining = [i for i in remaining if i not in k4["covered"]]
    return {"stage": "sh", "halt_lines": res["halt_lines"], "body_height": bh, "lazy_rh": lazy_dec, "k4": k4,
            "remaining": remaining, "reanchored": not remaining,
            "k4_accepted": bool(k4 and k4["accept"]) and not remaining}


def near_line_passes(res: dict, width: float = 0.02) -> list[str]:
    """Disclosure helper: decisive lines that PASS within ``width`` of their line (value and line numeric)."""
    out = []
    for d in res.get("decisive") or []:
        v, ln = d.get("value"), d.get("line")
        if d.get("pass") and isinstance(v, (int, float)) and isinstance(ln, (int, float)) and abs(v - ln) <= width:
            out.append(f"{d['id']}: {v:.4g} against {ln:.4g}")
    return out


def sh_decision_doc(res: dict, dec: dict, head: dict) -> dict:
    k4 = dec["k4"]
    po = k4["pooled"]
    per = k4["per_seed"]
    seeds = sorted(per["head"])
    hs = " / ".join(str(per["head"][s]["misses"]) for s in seeds)
    cs = " / ".join(str(per["control"][s]["misses"]) for s in seeds)
    ns = " / ".join(str(per["head"][s]["n"]) for s in seeds)
    lines = th_lines()
    acc = {i: {"value": d.get("value"), "line": d.get("line")} for d in res["decisive"] for i in k4["ids"]
           if d["id"] == i}
    re_anch = (dec["body_height"]["covered"] + ((dec.get("lazy_rh") or {}).get("covered") or []))
    return {"stage": "sh", "decided_at": L.now(), "decision": "accept, disclose",
            "by": "the pre-registered rule (p2/P2.md Decision rules 'K4 fails', settled 6f) -- applied by "
                  "p2_stand.py, not a user decision; the bake stays the user's call",
            "rule": k4["rule"], "test": K4_TEST, "alpha": K4_ALPHA,
            "accepted_lines": acc,
            "k4": {"head_misses_per_seed": hs, "control_misses_per_seed": cs, "n_per_seed": ns, "pooled": po,
                   "p_two_sided": k4["p_two_sided"]},
            "reason": (f"Chance-level relative to the scripted teacher R_h beside c1 ({RH} {cs}; head {hs}; pooled "
                       f"{po['head_misses']} vs {po['control_misses']} of {po['head_n']} / {po['control_n']}, "
                       f"two-sided p = {k4['p_two_sided']:.3f} >= {K4_ALPHA})"),
            "disclosed": [f"K4 restore misses {hs} per {ns} against D9's fixed {next(iter(acc.values()))['line']} "
                          f"(R_h beside c1 {cs})",
                          f"re-anchored lines ({len(re_anch)}): {re_anch}",
                          *(f"near-line pass (auto, within 0.02): {x}" for x in near_line_passes(res))],
            "head": {"name": head.get("name"), "ckpt": head.get("ckpt"), "sha256": head.get("sha256")},
            "walker": {"name": "c1 (RT_s19 model_29500)", "ckpt": str(C1_CK), "sha256": C1_SHA},
            "gate": str(STAND / "gates" / "sh.json"), "reanchor": str(STAND / "gates" / "sh_reanchor.json"),
            "k4_line_ids_checked_fixed": [i for i in k4["ids"] if (lines.get(i) or {}).get("override_line")
                                          is not None]}


# ============================================================================================ body height (Left 15)
_P2READ: dict = {}


def p2read():
    """The phase-2 reader set (p2_read, as sn_calib imports it), for the G5 estimators p2s_summary (suite) and
    z2_summary (walkstop) -- exactly the values sn_calib's G5 compares."""
    if "P" not in _P2READ:
        os.environ.setdefault("P2_OUT", str(R.P2_OUT))
        sys.path.insert(0, str(R.P2_SCRIPTS))
        import p2_read as P  # noqa: PLC0415
        _P2READ["P"] = P
    return _P2READ["P"]


def stamp_heights(suite: Path | None, walkstop: Path | None) -> dict:
    P = p2read()
    s = P.p2s_summary(str(suite)) if suite else None
    w = P.z2_summary(str(walkstop)) if walkstop else None
    return {"suite": (s or {}).get("body_height_m"), "walkstop": (w or {}).get("body_height_m")}


def recorded_heights(calib: dict, role: str) -> dict:
    """sn_calib's G5 ``heights`` {key: [row, REFS]} of a role / head row."""
    return ((((calib or {}).get("rows") or {}).get(role) or {}).get("gates") or {}).get("G5", {}).get("heights") or {}


def c1_suite_stamp(es: int) -> Path | None:
    slot = C1_SUITE_SLOTS.get(es)
    if slot is None or not slot.is_dir():
        return None
    for d in sorted(slot.iterdir(), reverse=True):
        m = L.read_json(d / "run_meta.json") or {}
        if (d / "complete.json").is_file() and m.get("checkpoint_sha256") == C1_SHA:
            return d
    return None


def bh_classify(refs: dict, controls: dict) -> dict:
    """Per key: the controls' spare = min over R / HOLD of |REFS - control| - 0.03. spare > 0: both controls FAIL the
    line, so a head standing with the controls re-anchors (the 11:53 rule); spare <= 0: a control PASSES it, and a
    head that fails the line HALTs (Left 15)."""
    out = {}
    for k in BH_KEYS:
        r = refs.get(k)
        cs = {c: (controls.get(c) or {}).get(k) for c in BH_CONTROLS}
        if r is None or any(v is None for v in cs.values()):
            out[k] = {"refs": r, "controls": cs, "class": "UNREADABLE"}
            continue
        spare = min(abs(r - v) for v in cs.values()) - BH_TOL
        cmean = sum(cs.values()) / len(cs)
        out[k] = {"refs": r, "controls": cs, "controls_spare_m": round(spare, 5),
                  "head_like_controls_fails_strict_line": abs(r - cmean) > BH_TOL,
                  "refs_at_or_below_plan_line": r <= BH_PLAN_LINE,
                  "class": "REANCHOR_AVAILABLE" if spare > 0 else "CONTROLS_PASS_HEAD_FAIL_IS_HALT",
                  "knife_edge": abs(spare) < BH_KNIFE}
    return out


def body_height_report(write: bool = True) -> dict:
    """The Left-15 report line. REFS(c1): measured on its suite@1,2 / walkstop@1,2 rows when all four are complete,
    else PREDICTED from c1's own 0-kg suite rows (RTSW es1 / RTA es2, same env as the driver) with RSD's
    suite - walkstop offset of REFS(TDR). The controls: sn_calib's R / HOLD heights (this stage's read when it exists,
    else RSD's results/stand/sn_calib.json: the same S1-era stamps, walker-independent)."""
    calib_p = STAND / "sn_calib.json"
    calib = L.read_json(calib_p) if calib_p.is_file() else None
    src = str(calib_p) if calib else str(RSD_STAND / "sn_calib.json")
    calib = calib or L.read_json(RSD_STAND / "sn_calib.json") or {}
    controls = {c: {k: (v or [None])[0] for k, v in recorded_heights(calib, c).items()} for c in BH_CONTROLS}
    st = {f"{lab} es{s}": R.complete_stamp(REFS, sc, s) for lab, sc in (("suite", "suite__omni"),
                                                                       ("walkstop", "walkstop__omni"))
          for s in (1, 2)}
    rep = {"at": L.now(), "controls_source": src, "tol": BH_TOL, "plan_line": BH_PLAN_LINE}
    if all(st.values()):
        refs = {}
        for s in (1, 2):
            h = stamp_heights(st[f"suite es{s}"], st[f"walkstop es{s}"])
            refs[f"suite es{s}"], refs[f"walkstop es{s}"] = h["suite"], h["walkstop"]
        rep.update(mode="measured", refs_stamps={k: str(v) for k, v in st.items()}, refs=refs)
    else:
        rsd = L.read_json(RSD_STAND / "sn_calib.json") or {}
        tdr = {k: (v or [None, None])[1] for k, v in recorded_heights(rsd, RSD_SN).items()}
        refs, used = {}, {}
        for s in (1, 2):
            c1s = c1_suite_stamp(s)
            used[s] = str(c1s) if c1s else None
            h = stamp_heights(c1s, None)["suite"] if c1s else None
            refs[f"suite es{s}"] = h
            off = (tdr.get(f"suite es{s}") or 0) - (tdr.get(f"walkstop es{s}") or 0) \
                if tdr.get(f"suite es{s}") and tdr.get(f"walkstop es{s}") else None
            refs[f"walkstop es{s}"] = (h - off) if (h is not None and off is not None) else None
        rep.update(mode="predicted", c1_suite_stamps=used, refs=refs,
                   note="walkstop = c1 suite - REFS(TDR)'s (suite - walkstop) per seed (RSD sn_calib.json); "
                        "suite = p2s_summary on c1's own 0-kg suite rows")
    rep["lines"] = bh_classify(rep["refs"], controls)
    for k, v in rep["lines"].items():
        if v["class"] == "UNREADABLE":
            print(f"P2ST BODY_HEIGHT {rep['mode']} {k}: UNREADABLE refs {v['refs']} controls {v['controls']}",
                  flush=True)
            continue
        print(f"P2ST BODY_HEIGHT {rep['mode']} {k}: REFS(c1) {v['refs']:.4f} R {v['controls']['R']:.4f} "
              f"HOLD {v['controls']['HOLD']:.4f} spare {v['controls_spare_m']:+.4f} -> {v['class']}"
              f"{' (REFS <= 1.095)' if v['refs_at_or_below_plan_line'] else ''}"
              f"{' KNIFE-EDGE' if v['knife_edge'] else ''}", flush=True)
    bad = [k for k, v in rep["lines"].items() if v["class"] != "REANCHOR_AVAILABLE" or v.get("knife_edge")]
    rep["notify"] = bad
    if bad:
        notify(f"body height ({rep['mode']}): {bad} -- a control passes or sits within {BH_KNIFE} m of the 0.03 "
               "line against REFS(c1); a head failing such a line HALTs (Left 15)")
    if write:
        L.write_json(STAND / f"body_height_{rep['mode']}.json", rep)
    return rep


# ============================================================================================ c1 / driver checks
def c1_problems() -> list[str]:
    bad = []
    if not C1_CK.is_file():
        return [f"no c1 checkpoint {C1_CK}"]
    if sha256_file(C1_CK) != C1_SHA:
        bad.append(f"{C1_CK} sha256 != {C1_SHA[:12]}")
    if not (C1_CK.parent / "params" / "agent.yaml").is_file():
        bad.append(f"no {C1_CK.parent}/params/agent.yaml (the walker spec's S1_AGENT_YAML)")
    return bad


def c1_main_copy() -> dict:
    """The deferred hardlink (P2.md step 2): report only."""
    if not C1_MAIN_CK.exists():
        return {"exists": False, "path": str(C1_MAIN_CK), "needed": False}
    st = C1_MAIN_CK.stat()
    return {"exists": True, "path": str(C1_MAIN_CK), "sha_ok": sha256_file(C1_MAIN_CK) == C1_SHA,
            "nlink": st.st_nlink, "same_inode_as_worktree": C1_CK.exists() and os.path.samestat(st, C1_CK.stat()),
            "needed": False}


def names_present() -> dict:
    out = {}
    for n in NAMES:
        cands = [R.LROUT / n, R.SN_OUT / "runs" / n, R.SH_OUT / "runs" / n]
        cands += [R.SN_OUT / "runs" / f"{n}.{x}" for x in ("log", "unit")]
        cands += [R.SH_OUT / "runs" / f"{n}.{x}" for x in ("log", "unit")]
        out[n] = [str(p) for p in cands if p.exists()]
    # the pre-flight may run beside `refs`: REFS may exist when every stamp under it is c1's at the P2 pin
    metas = [L.read_json(p) or {} for p in (R.LROUT / REFS).glob("*/seed*/*/run_meta.json")]
    if out[REFS] == [str(R.LROUT / REFS)] and metas and all(m.get("checkpoint_sha256") == C1_SHA for m in metas) \
            and not any(stamp_pin_problems(s) for s in name_stamps(REFS)):
        out[REFS] = []
    return out


def scripts_vs_main() -> dict:
    out = {}
    for d in (R.SN_SCRIPTS, R.SH_SCRIPTS, R.P2_SCRIPTS, L.PIPE):
        rel = d.relative_to(L.W)
        out[str(rel)] = [str(p.relative_to(d)) for p in sorted(d.rglob("*")) if p.is_file()
                         and "__pycache__" not in p.parts and (not (MAIN / rel / p.relative_to(d)).is_file()
                         or (MAIN / rel / p.relative_to(d)).read_bytes() != p.read_bytes())]
    return out


def driver_checks() -> dict:
    dm = L.read_json(R.TSTAR_DRIVER / "run_meta.json") or {}
    dev = R.env_vars(R.TSTAR_DRIVER)
    c1s = {s: c1_suite_stamp(s) for s in (1, 2)}
    return {"tstar_ckpt_sha": R.TSTAR_CKPT.is_file() and sha256_file(R.TSTAR_CKPT) == R.TSTAR_SHA,
            "driver_is_tstar_walkplay_0kg": (dm.get("task") == R.WALK_PLAY and dm.get("checkpoint_sha256")
                                             == R.TSTAR_SHA and dm.get("payload") is None),
            "driver_env_keys": len(dev),
            "c1_suite_stamps": {s: str(v) if v else None for s, v in c1s.items()},
            "driver_env_eq_c1_0kg_suite": bool(dev) and all(v is not None and R.env_vars(v) == dev
                                                            for v in c1s.values()),
            "c1_suite_payload_none": all(v is not None and (L.read_json(v / "run_meta.json") or {}).get("payload")
                                         is None for v in c1s.values()),
            "no_dr_stress_payload_keys": not any(k.startswith(("KRABBY_DR_", "KRABBY_STRESS_", "KRABBY_PAYLOAD_"))
                                                 for k in dev)}


# ============================================================================================ the steps
def need(path: Path, what: str) -> dict:
    return R.need(path, what)


def step_preflight():
    """The queue's pre-flight step: reuse a FRESH ok results/p2/stand/preflight.json (same pin, same p2_stand.py
    bytes, version), else run it (CPU, ~10 min)."""
    if preflight_fresh():
        return "pre-flight ok (fresh; skipped)"
    return preflight()


def preflight_fresh(pre: dict | None = None) -> bool:
    pre = pre if pre is not None else (L.read_json(STAND / "preflight.json") or {})
    pin = p2_pin() or {}
    return bool(pre.get("ok")) and pre.get("version") == PREFLIGHT_VERSION \
        and (pre.get("pin") or {}).get("code_sha") == pin.get("code_sha") \
        and (pre.get("pin") or {}).get("eval_sha") == pin.get("eval_sha") \
        and pre.get("p2_stand_sha256") == sha256_file(Path(__file__))


def step_refs():
    """REFS(c1): the nine walker-alone rows (run_arm.sh --row-eval, P2_REANCHOR=1, T*'s driver stamp as the env
    input, as RSD's), every stamp checked (walker_problems: Walk-Play, c1's sha, 0 kg, the driver env + the row's
    manifest env == REFS_T4F19's same row; raw npz == episodes; recorded at the P2 pin); then the walker spec with
    REFS(c1)'s own suite@1 stamp as REFS_ENV_STAMP, and the measured body-height report line."""
    STAND.mkdir(parents=True, exist_ok=True)
    bad = c1_problems()
    if bad:
        raise L.Stop(f"c1: {bad}", 4)
    rows = R.scen_rows(R.spec().stage("sn")["refs_rows"])
    if len(rows) != 9:
        raise L.Stop(f"pipeline.yaml sn refs_rows has {len(rows)} rows, not 9", 4)
    for sc, es, fam in rows:
        if R.complete_stamp(REFS, sc, es):
            say(f"SKIP REFS(c1) {sc}@{es}: complete")
            continue
        guard(f"REFS(c1) {sc}@{es}")
        rc = R.run(["bash", R.RUN_ARM, "--row-eval", REFS, C1_CK, R.TSTAR_DRIVER, R.spec().manifest(fam), sc, es],
                   R.child_env({"P2_REANCHOR": "1", "P2_OUT": R.P2_OUT}), LOGS / "refs.log", f"REFS(c1) {sc}@{es}")
        if rc != 0 or not R.complete_stamp(REFS, sc, es):
            raise R.Infra(f"REFS(c1) {sc}@{es} rc={rc}")
    stamps = {f"{sc}@{es}": R.complete_stamp(REFS, sc, es) for sc, es, _ in rows}
    probs = {}
    for sc, es, _ in rows:
        k = f"{sc}@{es}"
        p = R.walker_problems(stamps[k], sc, es) + S3.raw_problems(stamps[k]) + stamp_pin_problems(stamps[k])
        if p:
            probs[k] = p
    if probs:
        raise L.Stop(f"REFS(c1) stamps: {probs}", 4)
    s1 = stamps["suite__omni@1"]
    L.write_json(STAND / "walker_spec.json", {
        "spec": R.walker_spec_doc(s1), "refs_stamps": {k: str(v) for k, v in stamps.items()},
        "code_state": {k: L.read_json(v / "code_state.json") for k, v in stamps.items()},
        "walker": {"name": "c1 (RT_s19 model_29500)", "ckpt": str(C1_CK), "sha256": C1_SHA,
                   "main_copy": c1_main_copy()},
        "note": "c1 has no Task 4 driver stamp; REFS(c1) suite@1 (Walk-Play, c1, 0 kg, the driver env) stands in, as "
                "REFS(TDR) did for TDR (STAND_REPAIR.md section 3)", "written_at": L.now()})
    try:
        body_height_report(write=True)
    except Exception as e:  # noqa: BLE001 -- a report line; never a stop
        notify(f"body-height report failed after refs (report only): {type(e).__name__}: {e}")
    return f"REFS(c1) 9 rows; REFS_ENV_STAMP {s1}"


def step_sn_train():
    if not preflight_fresh():
        raise L.Stop("no FRESH ok pre-flight (results/p2/stand/preflight.json: ok, this pin, this p2_stand.py): run "
                     "`p2_stand.py preflight` (STAND_REPAIR.md section 5)", 4)
    st = R.spec().stage("sn")
    args = ["--mode", "dagger", "--arm", st["arm"], "--seed", R.spec().seed(R.RUN, "sn"), "--iterations",
            R.spec().iters(R.RUN, "sn")]
    if str(args[args.index("--iterations") + 1]) != "4000" or str(args[args.index("--seed") + 1]) != "19":
        raise L.Stop(f"SN recipe drift: {args} (STAND_REPAIR.md: seed 19, 4000 its)", 4)
    h = R.train("sn", SN, R.SN_SCRIPTS / "sn_dagger.sh", R.SN_SCRIPTS / "sn_dagger.py", "SN_OUT", R.SN_OUT, args)
    return f"{SN} {h['sha256'][:12]}"


def step_sn_rows():
    note = R.step_sn_rows()
    bad = rows_pin_problems(SN, R.scen_rows(R.spec().stage("sn")["stand_rows"]))
    if bad:
        raise L.Stop(f"SN rows not at the P2 pin: {bad[:4]}", 4)
    return note


def step_sn_read():
    h = need(STAND / "sn_head.json", "the SN head")
    rows = R.scen_rows(R.spec().stage("sn")["stand_rows"])
    miss = [f"{sc}@{es}" for sc, es, _ in rows if not R.complete_stamp(SN, sc, es)]
    if miss:
        raise R.Infra(f"SN rows missing {miss}")
    bad = rows_pin_problems(SN, rows)
    if bad:
        raise L.Stop(f"SN rows not at the P2 pin: {bad[:4]}", 4)
    env = R.child_env({**R.walker_env(), "P2_OUT": R.P2_OUT})
    t0 = time.time()
    rcb = R.run([L.PY, "-I", R.SN_SCRIPTS / "sn_bridge_read.py", "--head", SN, "--refs", REFS, "--live", "--json",
                 STAND / "sn_bridge.json"], env, LOGS / "sn.read.log", "sn_bridge_read (recorded, not gated)")
    rcc = R.run([L.PY, "-I", R.SN_SCRIPTS / "sn_calib.py", "--head", SN, "--json", STAND / "sn_calib.json", "--md",
                 STAND / "sn_calib.md"], env, LOGS / "sn.read.log", "sn_calib")
    if R.fresh_json(STAND / "sn_calib.json", t0) is None:
        raise R.Infra(f"sn_calib rc={rcc} wrote no fresh sn_calib.json")
    rec = {"stage": "sn", "name": SN, "ckpt": h["ckpt"], "sha256": h["sha256"], "refs": REFS,
           "calib": str(STAND / "sn_calib.json"), "bridge": str(STAND / "sn_bridge.json"), "train_log": h["train_log"],
           "reader_rc": {"sn_bridge_read": rcb, "sn_calib": rcc}, "walker": {"ckpt": str(C1_CK), "sha256": C1_SHA}}
    res = gate("sn", "D1", rec)
    say(f"GATE sn {res['verdict']}: {len(res['decisive'])} decisive, halt {res['halt_lines']}, "
        f"{len(res['watch_flags'])} watch flag(s)")
    if len(res["decisive"]) != 35 or len(res["watch_lines"]) != 8:
        notify(f"SN gate line count {len(res['decisive'])} decisive / {len(res['watch_lines'])} WATCH, not 35 / 8")
    try:
        body_height_report(write=True)
    except Exception as e:  # noqa: BLE001
        notify(f"body-height report failed at sn_read (report only): {type(e).__name__}: {e}")
    if res["verdict"] != "HALT":
        return f"SN gate {res['verdict']}"
    dec = decide_sn(res, L.read_json(STAND / "sn_calib.json"), SN)
    dec["decided_at"] = L.now()
    L.write_json(STAND / "gates" / "sn_reanchor.json", dec)
    say(f"REANCHOR sn body height: covered {len(dec['body_height']['covered'])} of {len(res['halt_lines'])} "
        f"{json.dumps(dec['body_height']['lines'])[:600]}")
    if dec["remaining"]:
        raise L.Stop(f"SN gate HALT; not re-anchored: {dec['remaining']} (R / HOLD lazy controls or the user's call: "
                     "STAND_REPAIR.md section 4; P2.md 'Other stand HALT')", 4)
    notify(f"SN gate HALT -> re-anchored: {dec['body_height']['covered']} (body height on R / HOLD, the 11:53 rule)")
    return "SN gate HALT -> re-anchored (the body-height lines on the R / HOLD controls: the 11:53 rule)"


def step_sh_train():
    sn = need(STAND / "sn_head.json", "the SN head")
    if not sn.get("sha256") or sha256_file(Path(sn["ckpt"])) != sn["sha256"]:
        raise L.Stop(f"SN head {sn.get('ckpt')} sha != its record", 4)
    st = R.spec().stage("sh")
    if str(R.spec().iters(R.RUN, "sh")) != "5000":
        raise L.Stop(f"SH recipe drift: {R.spec().iters(R.RUN, 'sh')} its (STAND_REPAIR.md: 5000)", 4)
    if "--deadband" not in st["opts"] or "--trim-branch" not in st["opts"]:
        raise L.Stop(f"SH recipe drift: opts {st['opts']}", 4)
    return R.step_sh_train()


def step_sh_table():
    return R.step_sh_table()


def step_sh_rows():
    note = R.step_sh_rows()
    st = R.spec().stage("sh")
    bad = rows_pin_problems(SH, R.scen_rows(st["height_rows"])) + rows_pin_problems(SHZ, R.scen_rows(st["k5_rows"]))
    if bad:
        raise L.Stop(f"SH rows not at the P2 pin: {bad[:4]}", 4)
    return note


def step_rh_rows():
    """The lazy R_h control beside c1 (DESIGN 3.6), run up front as reversible work; read when an R_h-derived line
    fails, and always for a K4 failure (the chance test reads it)."""
    rows = R.scen_rows(R.spec().stage("sh")["height_rows"])
    todo = [f"{sc}@{es}" for sc, es, _ in rows if not R.complete_stamp(RH, sc, es)]
    if todo:
        guard("R_h beside c1")
        rc = R.run(["bash", R.SH_SCRIPTS / "sh_eval_rows.sh", RH, "r_h", "manifest", "armed", "-", "-", *todo],
                   R.child_env({**R.walker_env(), "P2_OUT": R.P2_OUT}), LOGS / "sh.rh_control.log",
                   f"lazy R_h control beside c1 {todo}")
        miss = [f"{sc}@{es}" for sc, es, _ in rows if not R.complete_stamp(RH, sc, es)]
        if rc != 0 or miss:
            raise R.Infra(f"R_h control rc={rc}; missing {miss}")
    bad = rows_pin_problems(RH, rows)
    if bad:
        raise L.Stop(f"R_h control rows not at the P2 pin: {bad[:4]}", 4)
    return f"R_h control rows {len(rows)}" + ("" if todo else " (complete, skipped)")


def step_sh_read():
    h = need(STAND / "sh_head.json", "the SH head")
    tab = need(STAND / "sh_table.json", "the SH table test")["dir"]
    st = R.spec().stage("sh")
    miss = [f"{nm}:{sc}@{es}" for key, nm in (("height_rows", SH), ("k5_rows", SHZ))
            for sc, es, _ in R.scen_rows(st[key]) if not R.complete_stamp(nm, sc, es)]
    if miss:
        raise R.Infra(f"SH rows missing {miss}")
    bad = rows_pin_problems(SH, R.scen_rows(st["height_rows"])) + rows_pin_problems(SHZ, R.scen_rows(st["k5_rows"]))
    if bad:
        raise L.Stop(f"SH rows not at the P2 pin: {bad[:4]}", 4)
    rspec = R.pp().sh_read_spec(R.spec(), st, SH, Path(tab), REFS)
    L.write_json(STAND / "sh_read_spec.json", rspec)
    env = R.child_env({**R.walker_env(), "P2_OUT": R.P2_OUT})
    t0 = time.time()
    rcr = R.run([L.PY, "-I", R.SH_SCRIPTS / "sh_read.py", "--spec", STAND / "sh_read_spec.json", "--json",
                 STAND / "sh_read.json", "--md", STAND / "sh_read.md"], env, LOGS / "sh.read.log", "sh_read")
    rck = R.run([L.PY, "-I", R.SN_SCRIPTS / "sn_calib.py", "--head", SHZ, "--json", STAND / "sh_k5_calib.json",
                 "--md", STAND / "sh_k5_calib.md"], env, LOGS / "sh.read.log", "sn_calib (K5)")
    for p in ("sh_read.json", "sh_k5_calib.json"):
        if R.fresh_json(STAND / p, t0) is None:
            raise R.Infra(f"{p} not written (sh_read rc={rcr}, sn_calib rc={rck})")
    rec = {"stage": "sh", "name": SH, "ckpt": h["ckpt"], "sha256": h["sha256"], "refs": REFS,
           "read": str(STAND / "sh_read.json"), "k5_calib": str(STAND / "sh_k5_calib.json"), "table": tab,
           "train_log": h["train_log"], "reader_rc": {"sh_read": rcr, "sn_calib_k5": rck},
           "walker": {"ckpt": str(C1_CK), "sha256": C1_SHA}}
    res = gate("sh", "D2", rec)
    say(f"GATE sh {res['verdict']}: {len(res['decisive'])} decisive, halt {res['halt_lines']}")
    if len(res["decisive"]) != 101:
        notify(f"SH gate line count {len(res['decisive'])} decisive, not 101")
    if res["verdict"] != "HALT":
        return f"SH gate {res['verdict']}"
    k5 = L.read_json(STAND / "sh_k5_calib.json")
    lazy, lazy_path = None, None
    if sh_needs_lazy(res, k5, SHZ):
        rows = R.scen_rows(st["height_rows"])
        if not all(R.complete_stamp(RH, sc, es) for sc, es, _ in rows):
            raise L.Stop(f"SH gate HALT on {res['halt_lines']}; the lazy R_h control rows are not complete "
                         "(run step rh_rows)", 4)
        lspec = json.loads(json.dumps(rspec))
        lspec["roles"]["R_h"]["name"] = RH
        L.write_json(STAND / "sh_lazy_rh_read_spec.json", lspec)
        t1 = time.time()
        rcl = R.run([L.PY, "-I", R.SH_SCRIPTS / "sh_read.py", "--spec", STAND / "sh_lazy_rh_read_spec.json", "--json",
                     STAND / "sh_lazy_rh_read.json", "--md", STAND / "sh_lazy_rh_read.md"], env, LOGS / "sh.read.log",
                    "sh_read (lazy R_h beside c1)")
        lazy = R.fresh_json(STAND / "sh_lazy_rh_read.json", t1)
        if lazy is None:
            raise R.Infra(f"sh_lazy_rh_read.json not written (rc={rcl})")
        lazy_path = str(STAND / "sh_lazy_rh_read.json")
    dec = decide_sh(res, k5, lazy, head=SH, headz=SHZ, control=RH, lazy_path=lazy_path)
    dec["decided_at"] = L.now()
    L.write_json(STAND / "gates" / "sh_reanchor.json", dec)
    k4 = dec.get("k4") or {}
    say(f"REANCHOR sh: body height covered {len(dec['body_height']['covered'])}, lazy R_h covered "
        f"{len((dec.get('lazy_rh') or {}).get('covered') or [])}, K4 "
        f"{'none' if not k4 else ('ACCEPT' if k4.get('accept') else 'HALT')} p={k4.get('p_two_sided')} "
        f"{json.dumps(k4.get('pooled'))}, remaining {dec['remaining']}")
    if dec.get("k4_accepted"):
        doc = sh_decision_doc(res, dec, h)
        L.write_json(STAND / "gates" / "sh_decision.json", doc)
        notify(f"SH K4 ACCEPTED AND DISCLOSED by the pre-registered rule (settled 6f): {doc['reason']}")
    elif k4:
        notify(f"SH K4 not at chance (p={k4.get('p_two_sided')}, problems {k4.get('problems')}): HALT, the user's call")
    if dec["remaining"]:
        raise L.Stop(f"SH gate HALT; not covered: {dec['remaining']} (lazy controls exhausted: the user's call; "
                     "STAND_REPAIR.md section 4, P2.md Decision rules)", 4)
    if not dec.get("k4_accepted"):
        notify(f"SH gate HALT -> re-anchored ({len(dec['halt_lines'])} lines)")
    return "SH gate HALT -> covered (body height on R / HOLD; R_h-derived on the lazy R_h; K4 by the chance rule)"


def step_bh():
    body_height_report(write=True)
    return "body-height report written"


STEPS = {"preflight": step_preflight, "refs": step_refs, "sn_train": step_sn_train, "sn_rows": step_sn_rows,
         "sn_read": step_sn_read, "sh_train": step_sh_train, "sh_table": step_sh_table, "sh_rows": step_sh_rows,
         "rh_rows": step_rh_rows, "sh_read": step_sh_read, "bh": step_bh}
QUEUE_ORDER = ("preflight", "refs", "sn_train", "sn_rows", "sn_read", "sh_train", "sh_table", "sh_rows", "rh_rows",
               "sh_read")


# ============================================================================================ pre-flight (CPU)
def rsd_gate_replay(rep: dict) -> None:
    """P1r: Task 4's stand_gate on RSD's recorded inputs reproduces RSD's gates (verdict + every decisive / WATCH line);
    and RSD's thresholds sha is this file's."""
    sn_rec = {"stage": "sn", "name": RSD_SN, "calib": str(RSD_STAND / "sn_calib.json")}
    sh_rec = {"stage": "sh", "name": RSD_SH, "read": str(RSD_STAND / "sh_read.json"),
              "k5_calib": str(RSD_STAND / "sh_k5_calib.json")}
    g_sn, g_sh = L.read_json(RSD_STAND / "gates/sn.json"), L.read_json(RSD_STAND / "gates/sh.json")
    r1 = gate("sn", "D1", sn_rec, out_name="replay_rsd_sn")
    r2 = gate("sh", "D2", sh_rec, out_name="replay_rsd_sh")
    tp = R.spec().thresholds_path()
    rep["P1r"] = {"sn": R.same_gate(r1, g_sn) or "equal", "sh": R.same_gate(r2, g_sh) or "equal",
                  "sn_verdict": r1["verdict"], "sh_verdict": r2["verdict"], "sh_halt": r2["halt_lines"],
                  "n": {"sn_decisive": len(r1["decisive"]), "sn_watch": len(r1["watch_lines"]),
                        "sh_decisive": len(r2["decisive"])},
                  "thresholds_sha_eq_rsd": (g_sh.get("thresholds") or {}).get("sha256") == L.sha256_file(tp)
                  == (g_sn.get("thresholds") or {}).get("sha256")}


def _strip(d: dict, drop=("decided_at", "control", "lazy_read")) -> dict:
    return {k: v for k, v in (d or {}).items() if k not in drop}


def rsd_decision_replay(rep: dict) -> None:
    """P3r: the P2 decision code on RSD's records reproduces RSD's re-anchor records (sn_reanchor.json; sh_reanchor
    .json body height + lazy R_h) and, on RSD's K4, the 18:10 outcome under the P2 rule: 11 vs 9 of 1600,
    p 0.82 -> accept and disclose."""
    g_sn, g_sh = L.read_json(RSD_STAND / "gates/sn.json"), L.read_json(RSD_STAND / "gates/sh.json")
    sn_dec = decide_sn(g_sn, L.read_json(RSD_STAND / "sn_calib.json"), RSD_SN)
    rec_sn = L.read_json(RSD_STAND / "gates/sn_reanchor.json")
    lazy = L.read_json(RSD_STAND / "sh_lazy_rh_read.json")
    sh_dec = decide_sh(g_sh, L.read_json(RSD_STAND / "sh_k5_calib.json"), lazy, head=RSD_SH, headz=RSD_SH + "z",
                       control=RSD_RH, lazy_path=str(RSD_STAND / "sh_lazy_rh_read.json"))
    rec_sh = L.read_json(RSD_STAND / "gates/sh_reanchor.json")
    d_sn = L._diff(L._norm(_strip(sn_dec)), _strip(rec_sn))
    d_bh = L._diff(L._norm(sh_dec["body_height"]), rec_sh["body_height"])
    d_lz = L._diff(L._norm(_strip(sh_dec["lazy_rh"])), _strip(rec_sh["lazy_rh"]))
    k4 = sh_dec.get("k4") or {}
    po = k4.get("pooled") or {}
    dec_rec = L.read_json(RSD_STAND / "gates/sh_decision.json") or {}
    rep["P3r"] = {
        "sn_reanchor": d_sn or "equal",
        "sh_body_height": d_bh or "equal",
        "sh_lazy_rh": d_lz or "equal",
        "sh_remaining_before_k4_eq_rsd": sorted(rec_sh.get("remaining") or []) == sorted(k4.get("ids") or []),
        "k4": {"pooled": po, "p_two_sided": k4.get("p_two_sided"), "accept": k4.get("accept"),
               "problems": k4.get("problems")},
        "k4_reproduces_record": (po.get("head_misses"), po.get("control_misses"), po.get("head_n"),
                                 po.get("control_n")) == (11, 9, 1600, 1600)
        and k4.get("p_two_sided") is not None and round(k4["p_two_sided"], 2) == 0.82
        and "p = 0.82" in str(dec_rec.get("reason")),
        "p2_rule_outcome": "accept and disclose" if sh_dec.get("k4_accepted") else "HALT",
        "expect": "every diff 'equal'; K4 11 vs 9 of 1600, p 0.82 -> accept and disclose (RSD's 18:10 outcome, here "
                  "by the pre-registered rule)"}


def k4_recorded_check() -> dict:
    out = []
    for hm, cm, n, want, nd in K4_RECORDED:
        p = binom_two_sided(hm, hm + cm, n / (n + n))      # equal exposure (800 per env seed per role)
        out.append({"head": hm, "control": cm, "n_each": n, "p": p, "recorded": want, "ok": round(p, nd) == want})
    return {"test": K4_TEST, "cases": out, "ok": all(c["ok"] for c in out)}


def f19_replays(rep: dict, out: Path, jobs: int | None, readers: bool) -> None:
    """RSD's own P1 / P2 / P1-on-P2 / P3 on F19's records (STAND_REPAIR.md section 5), unchanged in substance."""
    sn_rec = {"stage": "sn", "name": R.F19_SN, "calib": str(R.F19_STAND / "sn_calib.json")}
    sh_rec = {"stage": "sh", "name": R.F19_SH, "read": str(R.F19_STAND / "sh_read.json"),
              "k5_calib": str(R.F19_STAND / "sh_k5_calib.json")}
    r1 = gate("sn", "D1", sn_rec, out_name="replay_f19_sn")
    r2 = gate("sh", "D2", sh_rec, out_name="replay_f19_sh")
    rsd_pre = L.read_json(RSD_STAND / "preflight.json") or {}
    rep["P1"] = {"sn": R.same_gate(r1, L.read_json(R.F19_GATES / "sn.json")) or "equal",
                 "sh": R.same_gate(r2, L.read_json(R.F19_GATES / "sh.json")) or "equal",
                 "sn_verdict": r1["verdict"], "sh_verdict": r2["verdict"], "sh_halt": r2["halt_lines"]}
    rep["P1_vs_rsd_preflight"] = "equal" if all(rep["P1"].get(k) == (rsd_pre.get("P1") or {}).get(k)
                                                for k in ("sn", "sh", "sn_verdict", "sh_verdict", "sh_halt")) \
        else {"mine": rep["P1"], "rsd": rsd_pre.get("P1")}
    dec = R.lazy_decide(L.read_json(R.F19_GATES / "sh.json"), L.read_json(R.F19_STAND / "sh_lazy_rh_read.json"),
                        R.F19_SH)
    rep["P3"] = {"reanchored": dec["reanchored"], "lines": dec["lines"], "not_rh_derived": dec["not_rh_derived"],
                 "expect": "reanchored True: R_h beside T* bin 4 es1 0.89 < 0.945; head 0.935 passes the re-anchored "
                           "line", "eq_rsd_preflight": dec["lines"] == (rsd_pre.get("P3") or {}).get("lines")
                 and dec["reanchored"] == (rsd_pre.get("P3") or {}).get("reanchored")}
    if not readers:
        rep["P2"] = "SKIPPED (--no-readers): not a valid pre-flight"
        return
    env = R.child_env({**R.f19_walker_env(), "P2_OUT": R.P2_OUT})
    jx = ["--jobs", str(jobs)] if jobs else []
    jobs_ = [("sn_calib", [R.SN_SCRIPTS / "sn_calib.py", "--head", R.F19_SN, *jx], "sn_calib.json",
              ("rows", R.F19_SN, "gates"), ()),
             ("sh_read", [R.SH_SCRIPTS / "sh_read.py", "--spec", R.F19_STAND / "sh_read_spec.json"], "sh_read.json",
              ("gates",), ()),
             ("sh_k5_calib", [R.SN_SCRIPTS / "sn_calib.py", "--head", R.F19_SH + "z", *jx], "sh_k5_calib.json",
              ("rows", R.F19_SH + "z", "gates"), ()),
             ("sh_lazy_rh_read", [R.SH_SCRIPTS / "sh_read.py", "--spec", R.F19_STAND / "sh_lazy_rh_read_spec.json"],
              "sh_lazy_rh_read.json", ("gates",), ("K5",))]
    p2 = {}
    for tag, cmd, ref, path, skip in jobs_:
        t0 = time.time()
        rc = R.run([L.PY, "-I", *cmd, "--json", out / f"{tag}.json", "--md", out / f"{tag}.md"], env,
                   LOGS / "preflight.log", f"P2 {tag}")
        new = R.fresh_json(out / f"{tag}.json", t0)
        p2[tag] = {"rc": rc, "skipped_gates": list(skip), "s": round(time.time() - t0, 1),
                   "diff": (R.same_clauses(new, L.read_json(R.F19_STAND / ref), path, skip) or "equal")
                   if new is not None else "NOT WRITTEN"}
    lz = L.read_json(out / "sh_lazy_rh_read.json")
    if lz is not None:
        a = {k: v for k, v in R.clause_map(lz, ("gates",)).items() if k[0] == "K5"}
        o = {k: v for k, v in R.clause_map(L.read_json(R.F19_STAND / "sh_read.json"), ("gates",)).items()
             if k[0] == "K5"}
        p2["lazy_k5_vs_f19_original"] = {
            "rc": 0, "diff": [f"{k} {(a[k] or {}).get('value')} != {(o.get(k) or {}).get('value')}" for k in sorted(a)
                              if (a[k] or {}).get("value") != (o.get(k) or {}).get("value")] or "equal"}
    rep["P2"] = p2
    if all(p2[t]["diff"] == "equal" for t in ("sh_read", "sh_k5_calib")):
        r3 = gate("sh", "D2", {**sh_rec, "read": str(out / "sh_read.json"), "k5_calib": str(out / "sh_k5_calib.json")},
                  out_name="replay_f19_sh_rereads")
        rep["P1_on_P2"] = R.same_gate(r3, L.read_json(R.F19_GATES / "sh.json")) or "equal"


def p4_static() -> dict:
    pin = p2_pin() or {}
    tp = R.spec().thresholds_path()
    names = names_present()
    sv = scripts_vs_main()
    ke = L.read_json(R.F19_KE) or {}
    bank = ke.get("KRABBY_RSI_BANK")
    wbank = Path(bank.replace(str(L.MAIN), str(L.W), 1)) if bank else None
    p4 = {"c1_problems": c1_problems(), "c1_ckpt": str(C1_CK), "c1_main_copy": c1_main_copy(),
          **driver_checks(),
          "names_absent": names,
          "candidates_identical_to_main": (L.T4 / "F19/candidates.yaml").read_bytes()
          == (L.T4_MAIN / "F19/candidates.yaml").read_bytes(),
          "thresholds_path": str(tp), "thresholds_is_v1_2": tp.resolve() == L.TH_PATH.resolve(),
          "scripts_differ_from_main": sv,
          "build_problems": build_problems(), "pin": {k: pin.get(k) for k in ("code_sha", "eval_sha", "head")},
          "code_state": L.code_state(),
          "disk": L.disk_problems(MAIN / "parkour/outputs") or "ok",
          "ddc_gb": L.ddc_gb(), "free_gb": round(L.free_gb(MAIN / "parkour/outputs"), 1),
          "stop_files": [str(p) for p in STOP_FILES if p.exists()], "launcher_stop_files": launcher_stops(),
          "f19_krabby_env": bool(ke), "rsi_bank_worktree_eq_f19": bool(wbank and wbank.is_file() and Path(bank).is_file()
                                                                       and L.sha256_file(wbank)
                                                                       == L.sha256_file(Path(bank))),
          "rsd_records": {p: (RSD_STAND / p).is_file() for p in ("sn_calib.json", "sh_read.json", "sh_k5_calib.json",
                                                                 "sh_lazy_rh_read.json", "gates/sn.json",
                                                                 "gates/sh.json", "gates/sn_reanchor.json",
                                                                 "gates/sh_reanchor.json", "gates/sh_decision.json",
                                                                 "preflight.json")},
          "walker_spec_template": R.walker_spec_doc(Path("<REFS(c1) suite@1 stamp>"))}
    p4["ok"] = (not p4["c1_problems"] and p4["tstar_ckpt_sha"] and p4["driver_is_tstar_walkplay_0kg"]
                and p4["driver_env_eq_c1_0kg_suite"] and p4["c1_suite_payload_none"]
                and p4["no_dr_stress_payload_keys"] and p4["driver_env_keys"] == 27
                and not any(names.values()) and p4["candidates_identical_to_main"] and p4["thresholds_is_v1_2"]
                and not any(sv.values()) and not p4["build_problems"] and not p4["code_state"]["dirty"]
                and p4["disk"] == "ok" and not p4["stop_files"] and not p4["launcher_stop_files"]
                and p4["f19_krabby_env"] and p4["rsi_bank_worktree_eq_f19"] and all(p4["rsd_records"].values()))
    return p4


def p5_body_height() -> dict:
    """P5 (report line, Left 15): the G5 estimators reproduce RSD's recorded REFS(TDR) heights on REFS(TDR)'s own
    stamps (a check), then the predicted (or measured) REFS(c1) lines (report only)."""
    rsd = L.read_json(RSD_STAND / "sn_calib.json") or {}
    tdr = {k: (v or [None, None])[1] for k, v in recorded_heights(rsd, RSD_SN).items()}
    got, diffs = {}, {}
    for s in (1, 2):
        h = stamp_heights(R.complete_stamp(RSD_REFS, "suite__omni", s), R.complete_stamp(RSD_REFS, "walkstop__omni", s))
        got[f"suite es{s}"], got[f"walkstop es{s}"] = h["suite"], h["walkstop"]
    for k, v in got.items():
        diffs[k] = None if v is None or tdr.get(k) is None else round(v - tdr[k], 6)
    est_ok = all(d is not None and abs(d) <= 1e-4 for d in diffs.values())
    rep = body_height_report(write=True)
    return {"estimator_on_refs_tdr": got, "rsd_recorded_refs_tdr": tdr, "diff": diffs, "estimator_ok": est_ok,
            "report": rep}


def preflight(jobs: int | None = None, readers: bool = True) -> str:
    out = STAND / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    rep = {"at": L.now(), "version": PREFLIGHT_VERSION, "p2_stand_sha256": sha256_file(Path(__file__)),
           "pin": {k: (p2_pin() or {}).get(k) for k in ("code_sha", "eval_sha", "head")},
           "walker": {"name": "c1 (RT_s19 model_29500)", "ckpt": str(C1_CK), "sha256": C1_SHA},
           "names": list(NAMES), "roots": {"records": str(STAND), "logs": str(LOGS), "rows": str(R.LROUT),
                                           "sn_runs": str(R.SN_OUT / "runs"), "sh_runs": str(R.SH_OUT / "runs"),
                                           "sh_table": str(SH_TABLE_OUT)}}
    rep["P4"] = p4_static()
    f19_replays(rep, out, jobs, readers)
    rsd_gate_replay(rep)
    rsd_decision_replay(rep)
    rep["K4_recorded"] = k4_recorded_check()
    try:
        rep["P5"] = p5_body_height()
    except Exception as e:  # noqa: BLE001 -- recorded; the estimator check then fails the pre-flight
        rep["P5"] = {"estimator_ok": False, "error": f"{type(e).__name__}: {e}"}
    p2 = rep.get("P2")
    ok = (rep["P4"]["ok"] and rep["P1"]["sn"] == "equal" and rep["P1"]["sh"] == "equal"
          and isinstance(p2, dict) and all(v["diff"] == "equal" for v in p2.values())
          and rep.get("P1_on_P2") == "equal" and rep["P3"]["reanchored"] is True
          and rep["P1r"]["sn"] == "equal" and rep["P1r"]["sh"] == "equal" and rep["P1r"]["thresholds_sha_eq_rsd"]
          and all(rep["P3r"][k] == "equal" for k in ("sn_reanchor", "sh_body_height", "sh_lazy_rh"))
          and rep["P3r"]["k4_reproduces_record"] and rep["P3r"]["p2_rule_outcome"] == "accept and disclose"
          and rep["K4_recorded"]["ok"] and rep["P5"].get("estimator_ok"))
    rep["ok"] = bool(ok)
    L.write_json(STAND / "preflight.json", rep)
    summ = {"P4": rep["P4"]["ok"], "P1": (rep["P1"]["sn"], rep["P1"]["sh"]),
            "P2": {k: v["diff"] if isinstance(v["diff"], str) else len(v["diff"]) for k, v in p2.items()}
            if isinstance(p2, dict) else p2, "P1_on_P2": rep.get("P1_on_P2"), "P3": rep["P3"]["reanchored"],
            "P1r": (rep["P1r"]["sn"], rep["P1r"]["sh"]), "P3r": {k: rep["P3r"][k] for k in
                                                                ("sn_reanchor", "sh_body_height", "sh_lazy_rh",
                                                                 "p2_rule_outcome")},
            "K4": rep["K4_recorded"]["ok"], "P5": rep["P5"].get("estimator_ok")}
    print(f"P2ST PREFLIGHT ok={rep['ok']} {json.dumps(summ, default=str)}", flush=True)
    if not ok:
        bad4 = {k: v for k, v in rep["P4"].items() if k in ("c1_problems", "names_absent", "build_problems",
                                                             "disk", "stop_files", "launcher_stop_files")}
        print(f"P2ST PREFLIGHT P4 {json.dumps(bad4, default=str)[:1500]}", flush=True)
        raise L.Stop(f"pre-flight failed ({STAND / 'preflight.json'})", 4)
    return "pre-flight ok"


# ============================================================================================ dry (read-only)
def dry() -> int:
    """The plan and the read-only checks: nothing written, no lock probed, nothing launched."""
    bad, info = [], []
    pin = p2_pin()
    bp = build_problems()
    (bad if bp else info).append(f"build vs results/p2/pin.json: {bp or 'OK'} (pin code {str((pin or {}).get('code_sha'))[:8]} / "
                                 f"eval {str((pin or {}).get('eval_sha'))[:8]})")
    cp = c1_problems()
    (bad if cp else info).append(f"c1 {C1_CK}: {cp or 'sha ' + C1_SHA[:8] + ' OK, params/agent.yaml OK'}")
    info.append(f"c1 MAIN copy (deferred hardlink; NOT needed by the stand): {c1_main_copy()}")
    dc = driver_checks()
    ok_d = dc["tstar_ckpt_sha"] and dc["driver_is_tstar_walkplay_0kg"] and dc["driver_env_eq_c1_0kg_suite"] \
        and dc["no_dr_stress_payload_keys"] and dc["driver_env_keys"] == 27
    (info if ok_d else bad).append(f"driver / c1 0-kg env: {dc}")
    names = names_present()
    (bad if any(names.values()) else info).append(f"names absent: {names}")
    sv = scripts_vs_main()
    (bad if any(sv.values()) else info).append(f"stand / phase-2 / pipeline scripts == main checkout: "
                                               f"{'yes' if not any(sv.values()) else sv}")
    disk = L.disk_problems(MAIN / "parkour/outputs")
    (bad if disk else info).append(f"disk: free {L.free_gb(MAIN / 'parkour/outputs'):.0f} GB, DerivedDataCache "
                                   f"{L.ddc_gb():.0f} GB (guard 78; the move before the stand is the parent's job): "
                                   f"{disk or 'OK'}")
    sf = [str(p) for p in STOP_FILES if p.exists()] + launcher_stops()
    (bad if sf else info).append(f"STOP files: {sf or 'none'}")
    pre = L.read_json(STAND / "preflight.json")
    info.append(f"pre-flight: {'none yet' if pre is None else ('FRESH ok' if preflight_fresh(pre) else 'present, not fresh/ok')}"
                f" ({STAND / 'preflight.json'})")
    th = R.spec().thresholds_path()
    (info if th.resolve() == L.TH_PATH.resolve() else bad).append(f"thresholds {th.name} sha "
                                                                 f"{L.sha256_file(th)[:8]}")
    miss = [p for p in ("sn_calib.json", "sh_read.json", "sh_k5_calib.json", "sh_lazy_rh_read.json", "gates/sn.json",
                        "gates/sh.json", "gates/sn_reanchor.json", "gates/sh_reanchor.json") if not
            (RSD_STAND / p).is_file()]
    (bad if miss else info).append(f"RSD stand records for the replays: {miss or 'all present'}")
    st, sh = R.spec().stage("sn"), R.spec().stage("sh")
    plan = [
        ("preflight", "CPU ~10 min: P1-P5 + RSD replays -> results/p2/stand/preflight.json (skipped when FRESH ok)"),
        ("refs", f"GPU ~0.6 h: run_arm.sh --row-eval {REFS} <c1> <T* driver stamp> <manifest> <row> <es> x "
                 f"{len(st['refs_rows'])}: {st['refs_rows']}"),
        ("sn_train", f"GPU ~1.7 h: sn_dagger.sh --wait-idle {SN} -- --mode dagger --arm {st['arm']} --seed "
                     f"{R.spec().seed(R.RUN, 'sn')} --iterations {R.spec().iters(R.RUN, 'sn')} (SN_OUT={R.SN_OUT})"),
        ("sn_rows", f"GPU ~0.6 h: sn_eval_rows.sh {SN} <model_4000> 1 {st['stand_rows']}"),
        ("sn_read", "CPU: sn_bridge_read + sn_calib -> stand_gate D1 (35 + 8 WATCH); HALT -> body height on R/HOLD"),
        ("sh_train", f"GPU ~2.3 h: sh_dagger.sh --wait-idle {SH} -- --mode dagger --arm mask --seed "
                     f"{R.spec().seed(R.RUN, 'sh')} --iterations {R.spec().iters(R.RUN, 'sh')} {' '.join(sh['opts'])} "
                     f"--warm-start <SN model_4000> --warm-sha <its sha>"),
        ("sh_table", f"GPU ~0.1 h: sh_table_test.sh --wait-idle --foreground --controllers stand_net --videos stand_net "
                     f"--stand-ckpt <SH> --label {SH} --deadband {sh.get('deadband', 0.25)} (SH_TABLE_OUT={SH_TABLE_OUT})"),
        ("sh_rows", f"GPU ~1.0 h: sh_eval_rows.sh --deadband {sh.get('deadband', 0.25)} {SH} stand_net manifest armed "
                    f"<SH> 1 {sh['height_rows']}; {SHZ} zero rows {sh['k5_rows']}"),
        ("rh_rows", f"GPU ~0.4 h: sh_eval_rows.sh {RH} r_h manifest armed - - {sh['height_rows']}"),
        ("sh_read", "CPU: sh_read + sn_calib (K5) -> stand_gate D2 (101); HALT -> body height, lazy R_h, K4 chance "
                    "test (accept-and-disclose at p >= 0.05)"),
    ]
    for s, what in plan:
        print(f"P2ST DRY PLAN {s}: {what}")
    for i in info:
        print(f"P2ST DRY OK   {i}")
    for b in bad:
        print(f"P2ST DRY BAD  {b}")
    print(f"P2ST DRY {'OK' if not bad else 'BAD'} ({len(bad)} problem(s)); records {STAND}; logs {LOGS}; rows "
          f"{R.LROUT}; queue lock {LOGS / 'p2_stand_queue.lock'} + {M.LOCK_PATH} (not probed in dry)")
    return 0 if not bad else 1


# ============================================================================================ self-test (CPU)
def selftest(out: Path) -> int:
    """CPU checks, writing only under ``out``."""
    configure(out / "stand", out / "logs")
    checks = []

    def chk(name, ok, detail=""):
        checks.append((name, bool(ok), detail))
        print(f"P2ST SELFTEST {'PASS' if ok else 'FAIL'} {name}{': ' + str(detail)[:300] if detail else ''}",
              flush=True)

    # K4 test vectors
    k = k4_recorded_check()
    chk("K4 test reproduces RSD's recorded p (0.82 / 0.21 / 0.035)", k["ok"],
        [(c["head"], c["control"], round(c["p"], 4)) for c in k["cases"]])
    fish = 0.0342  # Fisher's exact on 23 vs 10 of 1600 (computed independently; not the recorded test)
    chk("K4 test is not Fisher's (R18: binomial 0.0351 rounds to the recorded 0.035, Fisher 0.034)",
        round(binom_two_sided(23, 33, 0.5), 3) == 0.035 and round(fish, 3) != 0.035)
    chk("K4 binomial symmetric two-sided = 2 x tail at p = 0.5",
        abs(binom_two_sided(11, 20, 0.5) - 2 * sum(math.comb(20, i) for i in range(11, 21)) / 2 ** 20) < 1e-12)
    chk("K4 binomial no overflow at n = 1600", 0.0 <= binom_two_sided(800, 1600, 0.5) <= 1.0)
    # RSD gate replays + decision replays
    rep: dict = {}
    rsd_gate_replay(rep)
    chk("P1r RSD SN gate replay equal", rep["P1r"]["sn"] == "equal", rep["P1r"]["sn"])
    chk("P1r RSD SH gate replay equal", rep["P1r"]["sh"] == "equal", rep["P1r"]["sh"])
    chk("P1r line counts 35 + 8 WATCH / 101", rep["P1r"]["n"] == {"sn_decisive": 35, "sn_watch": 8,
                                                                  "sh_decisive": 101}, rep["P1r"]["n"])
    chk("P1r thresholds sha == RSD's gates'", rep["P1r"]["thresholds_sha_eq_rsd"])
    rsd_decision_replay(rep)
    for kk in ("sn_reanchor", "sh_body_height", "sh_lazy_rh"):
        chk(f"P3r RSD {kk} reproduced", rep["P3r"][kk] == "equal", rep["P3r"][kk])
    chk("P3r RSD K4: 11 vs 9 of 1600, p 0.82 (record), P2 rule -> accept and disclose",
        rep["P3r"]["k4_reproduces_record"] and rep["P3r"]["p2_rule_outcome"] == "accept and disclose",
        rep["P3r"]["k4"])
    chk("P3r RSD remaining-before-K4 == RSD's remaining", rep["P3r"]["sh_remaining_before_k4_eq_rsd"])
    # synthetic K4: a head far worse than its control HALTs; a head/lazy mismatch HALTs
    g_sh = L.read_json(RSD_STAND / "gates/sh.json")
    lazy = L.read_json(RSD_STAND / "sh_lazy_rh_read.json")
    ids = [d["id"] for d in g_sh["decisive"] if is_k4_miss(d["id"], th_lines()) and not d["pass"]]
    lz2 = copy.deepcopy(lazy)
    for s, v in ((1, 1), (2, 0)):
        c = lz2["gates"]["K4"]["R_h"]["clauses"][f"miss es{s} <= max(1, R_h's)"]
        c["value"] = v
    k_bad = k4_chance(g_sh, lz2, RSD_SH, ids)
    chk("K4 synthetic: head 5/6 vs control 1/0 -> not at chance -> HALT", not k_bad["accept"]
        and k_bad["p_two_sided"] < 0.05, round(k_bad["p_two_sided"], 4))
    g2 = copy.deepcopy(g_sh)
    for d in g2["decisive"]:
        if d["id"] == ids[0]:
            d["value"] = 99
    k_mm = k4_chance(g2, lazy, RSD_SH, ids)
    chk("K4 synthetic: gate value != lazy head value -> problem -> HALT", not k_mm["accept"] and k_mm["problems"],
        k_mm["problems"])
    dec_no_lazy = decide_sh(g_sh, L.read_json(RSD_STAND / "sh_k5_calib.json"), None, head=RSD_SH,
                            headz=RSD_SH + "z", control=RSD_RH)
    chk("decide_sh without the lazy read asks for it", dec_no_lazy.get("needs_lazy") is True)
    # synthetic: a K4-only HALT (every other failed line passing) still reads the lazy read
    g3 = copy.deepcopy(g_sh)
    for d in g3["decisive"]:
        if d["id"] not in ids:
            d["pass"] = True
    g3["halt_lines"] = list(ids)
    chk("sh_needs_lazy on a K4-only HALT", sh_needs_lazy(g3, L.read_json(RSD_STAND / "sh_k5_calib.json"),
                                                         RSD_SH + "z"))
    d3 = decide_sh(g3, L.read_json(RSD_STAND / "sh_k5_calib.json"), lazy, head=RSD_SH, headz=RSD_SH + "z",
                   control=RSD_RH)
    chk("decide_sh K4-only HALT at chance -> accepted, nothing remaining", d3["k4_accepted"] and not d3["remaining"])
    doc = sh_decision_doc(g3, d3, {"name": RSD_SH, "ckpt": "x", "sha256": "y"})
    chk("sh_decision doc carries both counts and p", "11 vs 9" in doc["reason"] and "p = 0.82" in doc["reason"],
        doc["reason"])
    # body-height classification
    ctrl = {"R": {k: 1.0648 for k in BH_KEYS}, "HOLD": {k: 1.0648 for k in BH_KEYS}}
    c_ok = bh_classify({k: 1.107 for k in BH_KEYS}, ctrl)
    c_bad = bh_classify({k: 1.090 for k in BH_KEYS}, ctrl)
    chk("body height 1.107 vs controls 1.0648: re-anchor available, spare +0.0122",
        all(v["class"] == "REANCHOR_AVAILABLE" and abs(v["controls_spare_m"] - 0.0122) < 1e-9 for v in c_ok.values()))
    chk("body height 1.090: controls pass -> a head failure is a HALT",
        all(v["class"] == "CONTROLS_PASS_HEAD_FAIL_IS_HALT" and v["refs_at_or_below_plan_line"] for v in c_bad.values()))
    # F19 gate + lazy replays (P1 / P3; the reader replay P2 is the real pre-flight's)
    rep2: dict = {}
    f19_replays(rep2, out / "stand" / "preflight", None, readers=False)
    chk("P1 F19 SN gate replay equal", rep2["P1"]["sn"] == "equal", rep2["P1"]["sn"])
    chk("P1 F19 SH gate replay equal (HALT on K6 bin 4 es1)", rep2["P1"]["sh"] == "equal", rep2["P1"]["sh_halt"])
    chk("P1 == RSD's recorded pre-flight P1", rep2["P1_vs_rsd_preflight"] == "equal", rep2["P1_vs_rsd_preflight"])
    chk("P3 F19 lazy decision re-anchored, == RSD's pre-flight P3", rep2["P3"]["reanchored"] is True
        and rep2["P3"]["eq_rsd_preflight"])
    # static
    p4 = p4_static()
    for kk in ("c1_problems", "build_problems", "stop_files", "launcher_stop_files"):
        chk(f"P4 {kk} empty", not p4[kk], p4[kk])
    chk("P4 names absent", not any(p4["names_absent"].values()), p4["names_absent"])
    chk("P4 driver env == c1's own 0-kg suite env (27 keys)", p4["driver_env_eq_c1_0kg_suite"]
        and p4["driver_env_keys"] == 27)
    chk("P4 scripts == main checkout", not any(p4["scripts_differ_from_main"].values()))
    chk("P4 RSI bank in this worktree == F19's", p4["rsi_bank_worktree_eq_f19"])
    chk("P4 RSD records present", all(p4["rsd_records"].values()), p4["rsd_records"])
    chk("P4 disk ok", p4["disk"] == "ok", f"{p4['disk']} ddc {p4['ddc_gb']} GB free {p4['free_gb']} GB")
    chk("P4 ok", p4["ok"])
    L.write_json(out / "selftest_p4.json", p4)
    L.write_json(out / "selftest_rep.json", {**rep, **rep2})
    fails = [c for c in checks if not c[1]]
    print(f"P2ST SELFTEST {'OK' if not fails else 'FAIL'} {len(checks) - len(fails)}/{len(checks)}"
          + (f" -- failed: {[c[0] for c in fails]}" if fails else ""), flush=True)
    return 0 if not fails else 1


# ============================================================================================ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--jobs", type=int, default=None, help="sn_calib --jobs (its default 4)")
    p.add_argument("--no-readers", action="store_true", help="skip the reader replay (P2): never a valid pre-flight")
    p.add_argument("--out", type=Path, default=None, help="a scratch records + logs dir (a dry run of the "
                                                         "pre-flight: its preflight.json never counts for sn_train)")
    s = sub.add_parser("step")
    s.add_argument("step", choices=list(STEPS))
    sub.add_parser("walker")
    sub.add_parser("bh")
    sub.add_parser("dry")
    t = sub.add_parser("selftest")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--with-p5", action="store_true", help="also run P5 (reads REFS(TDR)'s and c1's suite / walkstop "
                                                         "raw npz: ~2-4 GB of IO)")
    a = ap.parse_args(argv)
    for k, v in _CALLER.items():
        if v and Path(v) != Path(os.environ[k]):
            print(f"P2ST REFUSED caller {k}={v} != {os.environ[k]}", flush=True)
            return 2
    if a.cmd == "selftest":
        out = a.out.resolve()
        if out == STAND.resolve() or str(out).startswith(str(CAMP.resolve())):
            print("P2ST REFUSED selftest --out must be outside the campaign tree (a scratch dir)", flush=True)
            return 2
        rc = selftest(out)
        if a.with_p5:
            try:
                r5 = p5_body_height()
                print(f"P2ST SELFTEST {'PASS' if r5['estimator_ok'] else 'FAIL'} P5 estimator reproduces REFS(TDR): "
                      f"{r5['diff']}", flush=True)
                rc = rc or (0 if r5["estimator_ok"] else 1)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                print(f"P2ST SELFTEST FAIL P5 {type(e).__name__}: {e}", flush=True)
                rc = 1
        return rc
    if a.cmd == "preflight" and a.out is not None:
        out = a.out.resolve()
        if str(out).startswith(str(CAMP.resolve())):
            print("P2ST REFUSED preflight --out must be outside the campaign tree (a scratch dir)", flush=True)
            return 2
        configure(out / "stand", out / "logs")
    else:
        configure()
    if a.cmd == "walker":
        print(json.dumps(R.walker_env(), indent=1, sort_keys=True))
        return 0
    if a.cmd == "dry":
        return dry()
    tag = {"step": getattr(a, "step", None), "preflight": "preflight", "bh": "bh"}[a.cmd]
    try:
        if a.cmd == "preflight":
            note = preflight(a.jobs, not a.no_readers)
        elif a.cmd == "bh":
            note = step_bh()
        else:
            note = STEPS[a.step]() or ""
        rc = 0
    except L.Stop as e:
        rc, note = 4, f"STOP {e.why}"
    except R.Infra as e:
        rc, note = 1, f"INFRA {e}"
    except Exception as e:  # noqa: BLE001 -- recorded as an infra failure with its traceback
        traceback.print_exc()
        rc, note = 1, f"INFRA {type(e).__name__}: {e}"
    print(f"P2ST {tag} END rc={rc} {note}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
