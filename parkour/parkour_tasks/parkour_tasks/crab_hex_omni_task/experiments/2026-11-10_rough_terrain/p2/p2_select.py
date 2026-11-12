#!/usr/bin/env python3
"""P2 steps 5-6 (p2/P2.md Design 'Selection', step 6): Task 4 ``sel`` against the c1 teacher, unchanged in rule --
prescreen, rank, first full-block PASS, HALT after 2 failed full blocks (``select_student.next_action``, max_failed 2),
so at most 2 blocks -- run through the rough-terrain campaign's OWN row machinery with the results/p2/pin.json check.

    p2_select.py --block teacher|prescreen|full [--dry | --list] [--from L] [--only L ...]   one block (GPU rows)
    p2_select.py --dry                     every block's dry pass (teacher, prescreen, full): slots, counts, commands
    p2_select.py --inputs-check --block B  (CPU) the block's pre-flight; also the first step of every block
    p2_select.py --rank                    (CPU) the prescreen read + rank: P2_SEL_RANK ... (exit 0; 12 / 13 else)
    p2_select.py --decide [--dry]          (CPU) results/p2/sel.json + ``P2_SEL PICK|HALT|NEXT|INCOMPLETE|STOP ...``
                                           (exit 0 / 11 / 10 / 12 / 13); --dry prints and writes nothing
    p2_select.py --fork-check [--out J]    (CPU) RSD's constants restored (TDR teacher rows, RSD stamps): the selection
                                           code reproduces RSD results/sel.json to 1e-9 on every number + the same pick
    p2_select.py --self-test               (CPU) the save-schedule projection, the WATCH and field-vs-c31000 reads on
                                           recorded stamps, the c1 suite reuse proof, the names, the dry slot counts

REUSE (nothing copied): the selection is rod_speed_dr's ``rsd_lib`` read code -- ``prescreen_read``, ``rank`` /
``next_action`` / ``candidate_checkpoints`` (Task 4 ``select_student``), ``full_block_read`` (floors via
``gate_read.walker_decisive``, ``completion_pairs`` vs the teacher, ``tracking_watch``), ``sel_gate_doc`` -- in
``rsd_gate.ph_sel``'s order, with ONLY these constants overridden: the teacher (c1 6d1f29be, not TDR), the student run
(RT_ST_s19), the output root (RSD_TP_OUT -> omni_plan/rough_terrain_p2), the results dir (RSD_RESULTS -> results/p2),
the row-name prefixes (P2SEL_ / P2SELT_ / P2SW_), the candidate ids (named by sha: ``<sha8>_it<iter>``, PLAN item 9;
a P2 save ``model_31000`` must never read as 'c31000') and the build pin (results/p2/pin.json, item 24). Rows are
composed by ``rsd_lib.sel_row`` / ``cell_rows`` (Task 4's ``row_job``: the p3b row env, TP_STRIP_KEYS on the sel
rows, the 90-kg level file) and launched as ``rt_machinery.gate_step`` steps by ``rt_queue.main`` (the p2_step3.py
pattern: flock on the campaign's .isaac.lock, the anchored ps -ww Isaac-busy wait, DDC <= 78 GB / free >= 30 GB before
every row, ``RTQ <label> rc=<n>`` sentinels, skip-on-complete at the pinned build, STOP_QUEUE between steps).

TEACHER ROWS (c1 = RT_s19 model_29500, role teacher on Student-Play; ``rsd_lib.teacher_sel_rows``: suite@1/2 +
field_light@1/2 = the L000 nominal cell rows, the other seven the sel rows): field_hard@1/2, field_light@1/2,
ring_obst@1/2, field_turn@1/2 run (8 rows, ``P2SELT_6d1f29be_*``); suite@1 / suite@2 REUSE c1's RTSW_6d1f29be_R_suite
(es1) / RTA_6d1f29be_R_suite (es2) stamps ONLY while ``rsd_lib.row_problems`` (every launch check: tp_row args /
env_file / level_file / manifest / scenario / env seed / stripped keys / rc, run_meta env_vars == the composed env,
carry, sha256, trained categories, task, payload, role, seed, scenario_id, no KRABBY_DR_*, the rod-speed dose) is
empty against the very row the fork would launch (its name aside) at results/p2/pin.json -- else they run too;
side22@1 is NOT run: ``full_block_read`` never reads the teacher's side22 (side22 FAST/FAST is student-absolute).

CANDIDATES: ``select_student.candidate_checkpoints(run_dir, start, offsets 1000..5000)`` on RT_ST_s19's real save
points, start = the run's first iteration (29500 = c1's iter, as RSD's 28991 was TDR's), the same rule RSD applied
relative to its run's first / last iteration (the last offset lands on the final save, start + 4999). The runner's
save schedule (on_policy_runner_with_extractor.py 654-658: every 100 to +2500, every 200 to +5000, + the final) is
projected for dry passes and cross-checked against the real saves.

WATCH TERRAIN (P2.md settled 6c / 7 Q1; REPORT ONLY, never fails a block or moves the pick): each full block adds the
student on rows_rt's rt_slopes_b2 / rt_ledge_down_b2 at es1-2 (role student: p2_step3.dprime_row), success = complete
AND crossed at the readout cross radius, paired on (es, env_index) with c1's existing RTSW es1-2 rows
(readout.pair_tables), line = c1 - max(0.05, 2 SE) with ``readout.retention_line`` semantics. c1's es1-2 values are the
seeds it was picked on (inflated by ~0.015-0.07, P2.md). At a 2-block HALT the candidates' field rows are also shown
against c31000's recorded RSDSEL rows (completion_pairs, cross-pin; P2.md Left item 25).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent                    # 2026-11-10_rough_terrain/p2
CAMP = HERE.parent
MAIN = Path("/home/nickmagus/krabby/krabby-research")
P2_OUT = MAIN / "parkour" / "outputs" / "omni_plan" / "rough_terrain_p2"
RSD_OUT = MAIN / "parkour" / "outputs" / "omni_plan" / "rod_speed_dr"
RESULTS_P2 = CAMP / "results" / "p2"

# ---- rsd_lib reads these at import (P2.md Design 'Reuse': RSD_RESULTS -> results/p2, RSD_TP_OUT -> rough_terrain_p2)
_CALLER = {k: os.environ.get(k) for k in ("RSD_TP_OUT", "RSD_RESULTS")}
os.environ["RSD_TP_OUT"] = str(P2_OUT)
os.environ["RSD_RESULTS"] = str(RESULTS_P2)

sys.path.insert(0, str(HERE))
import p2_step3 as S3  # noqa: E402  (sets RT_TP_OUT = P2_OUT before rt_machinery loads; its install() is never called)

M, Q, readout, RS, rt_short = S3.M, S3.Q, S3.readout, S3.S, S3.rt_short
L = S3.rsd_lib()
assert M.OUT == P2_OUT and Path(L.TP_OUT) == P2_OUT and Path(L.RESULTS) == RESULTS_P2, (M.OUT, L.TP_OUT, L.RESULTS)

# ============================================================================== pre-registered constants
W = M.W
C1_CK = W / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19/model_29500.pt"
C1_SHA = S3.C1_SHA
C1_ITER = 29500
C1_GUARD = ("c1 sha256 6d1f29be + the user's 2026-11-12 17:25 decision 'A' (c1 -> P2 by a disclosed overrule of R "
            "tracking; P2.md settled 1 / Design 'Distill' guard) -- RSD's g1.json PASS guard does not apply")
C31000_SHA, C31000_CK = S3.C31000_SHA, S3.C31000_CK
RUN_NAME = "RT_ST_s19"
STUDENT_RUNS = W / "parkour" / "logs" / "rsl_rl" / "crab_hex_omni_student"
DISTILL_LOG = P2_OUT / "distill" / "distill.log"
RUN_LOG = P2_OUT / "distill" / f"{RUN_NAME}.log"
RUN_ITS, RUN_SAVE_INTERVAL = 5000, 100
RT_OUT = S3.RT_OUT
REUSE_SUITE = {"suite@1": "RTSW_6d1f29be_R_suite", "suite@2": "RTA_6d1f29be_R_suite"}
NOT_CONSUMED = ("side22@1",)                              # full_block_read never reads the teacher's side22 row
WATCH_KINDS = dict(S3.KINDS)                              # slopes / ledge_down -> rows_rt rt_*_b2 ids
WATCH_ES = (1, 2)
WATCH_CFG = dict(readout.R_COMPLETION_CFG)                # floor 0.05, se_mult 2, split 0 -> max(0.05, 2 SE)
ROW_TIMEOUT_S = 5400                                      # RSD's row timeout (SIGTERM only)
PREFIX = {"student": "P2SEL_", "teacher": "P2SELT_", "watch": "P2SW"}
BLOCKS = {"teacher": "p2sel_teacher", "prescreen": "p2sel_prescreen", "full": "p2sel_full"}
#: rows per block: teacher 8 (+ suite@1/2 if the reuse proof fails); prescreen 5 per candidate (5 candidates unless
#: two offsets map to one save); full 11 + 4 at 90 kg + 4 WATCH (5 of them shared with the prescreen: skip-on-complete)
EXPECT_SLOTS = {"p2sel_teacher": (8, 10), "p2sel_prescreen": (5, 25), "p2sel_full": (19, 19)}
SEL_JSON = RESULTS_P2 / "sel.json"
PIN_P2 = S3.P2_PIN
RSD_RESULTS = L.CAMPAIGN / "results"
RSD_SEL_JSON = RSD_RESULTS / "sel.json"
RSD_SEL_ROWS = RSD_RESULTS / "sel.rows.json"
RSD_POLICIES = RSD_RESULTS / "policies.json"
RSD_STUDENT_RUN = MAIN / ".claude/worktrees/dr-stress-sweep/parkour/logs/rsl_rl/crab_hex_omni_student/" \
                         "2026-11-08_02-57-24_RSD_ST_s19"   # rsd_queue.sh one_student_dir (W = dr-stress-sweep then)
DECIDE_RC = {"SELECTED": 0, "NEXT": 10, "HALT": 11, "INCOMPLETE": 12, "STOP": 13}
WORD = {"SELECTED": "PICK", "NEXT": "NEXT", "HALT": "HALT", "INCOMPLETE": "INCOMPLETE", "STOP": "STOP"}
REFUSED_RC, HALT_RC = S3.REFUSED_RC, S3.HALT_RC
FORK_TOL = 1e-9
#: on_policy_runner_with_extractor.py 654-658 (learn_vision): the student runner's save rule
SAVE_RULE = ("save at it when (it - start < 2500 and it % s == 0) or (it - start < 5000 and it % 2s == 0) or "
             "(it - start >= 5000 and it % 5s == 0), plus model_<last>.pt after the loop (last = start + its - 1); "
             "parkour/scripts/rsl_rl/modules/on_policy_runner_with_extractor.py 654-658, 664-665")
MODE = {"dry": False}


def say(line: str) -> None:
    print(line, flush=True)


def is_dry() -> bool:
    return MODE["dry"] or os.environ.get("P2_DRY") == "1"


def read_json(p, default=None):
    return S3.read_json(p, default)


def pin() -> dict | None:
    """results/p2/pin.json (the P2 build pin; its code_sha / eval_sha equal results/pin.json's -- checked)."""
    return read_json(PIN_P2)


def pin_problems() -> tuple[list, list]:
    bad, warn = S3.p2_pin_problems()
    for k, v in _CALLER.items():
        want = P2_OUT if k == "RSD_TP_OUT" else RESULTS_P2
        if v and Path(v) != want:
            bad.append(f"caller {k}={v} != {want}")
    return bad, warn


# ============================================================================== policies
_SHA: dict = {}


def sha(path) -> str:
    p = Path(path)
    k = (str(p), p.stat().st_mtime_ns, p.stat().st_size)
    if k not in _SHA:
        _SHA[k] = M.sha256_file(p)
    return _SHA[k]


def c1_policy() -> dict:
    return {"name": "C1", "checkpoint": str(C1_CK), "sha256": C1_SHA, "sha8": C1_SHA[:8], "role": "teacher",
            "iteration": C1_ITER}


_C1: dict = {}


def c1_problems(check_iter: bool = True) -> list:
    if "why" in _C1:
        return _C1["why"]
    why = []
    if not C1_CK.is_file():
        why.append(f"no c1 checkpoint {C1_CK}")
    else:
        s = sha(C1_CK)
        if s != C1_SHA:
            why.append(f"{C1_CK} sha256 {s[:8]} != c1 {C1_SHA[:8]}")
        elif check_iter:
            it = rt_short.iter_field(C1_CK)
            if it != C1_ITER:
                why.append(f"{C1_CK.name} iter field {it!r} != {C1_ITER}")
    _C1["why"] = why
    return why


def cid_of(pol: dict, it: int, mode: str) -> str:
    """The candidate id: RSD's ``c<iter>`` in the fork, ``<sha8>_it<iter>`` in P2 (PLAN item 9: by sha)."""
    return f"c{int(it)}" if mode == "rsd" else f"{pol['sha8']}_it{int(it)}"


def placeholder(c: dict) -> dict:
    """A dry-pass stand-in for a projected save that does not exist yet (its rows print; nothing can run)."""
    tag = f"D{int(c['iter'])}"[:8].ljust(8, "0")
    return {"name": None, "checkpoint": c["ckpt"], "sha256": tag.ljust(64, "0"), "sha8": tag, "role": "student",
            "iteration": c["iter"], "placeholder": True}


# ============================================================================== the run and its save points
def project_saves(start: int, its: int = RUN_ITS, s: int = RUN_SAVE_INTERVAL) -> list:
    """The iterations the student runner saves (:data:`SAVE_RULE`)."""
    out = []
    for it in range(int(start), int(start) + int(its)):
        loc = it - int(start)
        if (loc < 2500 and it % s == 0) or (loc < 5000 and it % (2 * s) == 0) or (loc >= 5000 and it % (5 * s) == 0):
            out.append(it)
    last = int(start) + int(its) - 1
    if last not in out:
        out.append(last)
    return out


def nearest_grid(saved: list, start: int, offsets) -> list:
    """``select_student.candidate_checkpoints``' nearest-save rule on a list of iterations (dry projection only;
    a real run calls candidate_checkpoints on the run dir). Ties go to the earlier save; duplicates collapse."""
    out, seen = [], set()
    for off in offsets:
        target = int(start) + int(off)
        i = min(saved, key=lambda x: (abs(x - target), x))
        if i in seen:
            continue
        seen.add(i)
        out.append({"offset": int(off), "target": target, "iter": i})
    return out


def _first_iter_line(log: Path):
    try:
        with open(log, errors="replace") as fh:
            for ln in fh:
                m = M.ITER_RE.search(ln)
                if m:
                    return int(m.group(1)), int(m.group(2))
    except OSError:
        return None
    return None


def distill_end(name: str = RUN_NAME, log: Path = DISTILL_LOG):
    """The last ``TPDISTILL END <name> rc=<n>`` of distill.sh's log (None while the run is going)."""
    rc = None
    try:
        for ln in Path(log).read_text(errors="replace").splitlines():
            m = re.search(r"TPDISTILL END (\S+) rc=(-?\d+)", ln)
            if m and m.group(1) == name:
                rc = int(m.group(2))
    except OSError:
        return None
    return rc


def run_facts() -> dict:
    """RT_ST_s19: its single run dir, params (load_checkpoint = c1, 5000 its, save_interval 100), the first / last
    iteration from the distill log, distill.sh's END sentinel, the saved and the projected save points."""
    dirs = sorted(STUDENT_RUNS.glob(f"*_{RUN_NAME}"))
    f = {"run_dirs": [str(d) for d in dirs], "problems": [], "warnings": []}
    if len(dirs) != 1:
        f["problems"].append(f"{len(dirs)} run dirs *_{RUN_NAME} under {STUDENT_RUNS} (want exactly 1)")
        return f
    d = dirs[0]
    f["run_dir"] = str(d)
    agent = {}
    try:
        agent = M.load_params_yaml(d / "params" / "agent.yaml")
    except Exception as exc:  # noqa: BLE001
        f["problems"].append(f"params/agent.yaml unreadable: {type(exc).__name__}: {exc}")
    f["params"] = {k: agent.get(k) for k in ("load_checkpoint", "max_iterations", "save_interval", "resume")}
    if str(agent.get("load_checkpoint")) != str(C1_CK):
        f["problems"].append(f"params load_checkpoint {agent.get('load_checkpoint')} != c1 {C1_CK}")
    if int(agent.get("max_iterations") or -1) != RUN_ITS:
        f["problems"].append(f"params max_iterations {agent.get('max_iterations')} != {RUN_ITS}")
    if int(agent.get("save_interval") or -1) != RUN_SAVE_INTERVAL:
        f["problems"].append(f"params save_interval {agent.get('save_interval')} != {RUN_SAVE_INTERVAL}")
    fl = _first_iter_line(RUN_LOG)
    if fl is None:
        f["problems"].append(f"no 'Learning iteration N/M' line in {RUN_LOG}")
        start, last = C1_ITER, C1_ITER + RUN_ITS - 1
    else:
        start, last = fl[0], fl[1] - 1
        if fl[1] - fl[0] != RUN_ITS:
            f["problems"].append(f"{RUN_LOG.name}: first line {fl[0]}/{fl[1]} is not {RUN_ITS} iterations")
    if start != C1_ITER:
        f["problems"].append(f"the run's first iteration {start} != c1's iter {C1_ITER}")
    f.update(first_iteration=start, last_iteration=last, start=start)
    f["distill_end_rc"] = distill_end()
    f["finished"] = f["distill_end_rc"] == 0 and (d / f"model_{last}.pt").is_file()
    saved = sorted(i for p in d.glob("model_*.pt") if (i := L.its_of(p)) is not None)
    f["saved"] = saved
    f["projected"] = project_saves(start)
    f["save_rule"] = SAVE_RULE
    if f["finished"] and saved != f["projected"]:
        f["warnings"].append(f"saved points differ from the projection: only saved {sorted(set(saved) - set(f['projected']))[:8]}, "
                             f"only projected {sorted(set(f['projected']) - set(saved))[:8]} (the rule uses the saved)")
    if f["distill_end_rc"] not in (None, 0):
        f["problems"].append(f"distill.sh END {RUN_NAME} rc={f['distill_end_rc']}")
    return f


# ============================================================================== the selection context
def _rename(row: dict, old: str, new: str) -> dict:
    if not row["name"].startswith(old):
        raise ValueError(f"row name {row['name']} does not start {old}")
    return dict(row, name=new + row["name"][len(old):])


class Ctx:
    """One selection read. ``mode`` 'p2' (c1, RT_ST_s19, rough_terrain_p2 stamps, the P2 pin, every launch check) or
    'rsd' (the fork check: TDR, RSD_ST_s19, rod_speed_dr stamps located as rsd_gate's Runner located them)."""

    def __init__(self, mode: str, teacher: dict, run_dir: Path, start: int, *, p=None, dry: bool = False,
                 cands: list | None = None):
        self.mode, self.teacher, self.run_dir, self.start, self.pin, self.dry = mode, teacher, Path(run_dir), \
            int(start), p, dry
        self.cands = cands
        self.rows: dict = {}
        self.missing: list = []
        self.invalid: list = []
        self.block_stamps: dict = {}
        self._reuse = None

    # ---- candidates
    def candidates(self) -> list:
        if self.cands is None:
            self.cands = L.ss().candidate_checkpoints(self.run_dir, self.start, L.sel_spec()["offsets"])
        return self.cands

    def policy(self, c: dict) -> dict:
        if c.get("placeholder"):
            P = placeholder(c)
        elif self.mode == "rsd":
            return L.ephemeral_policy(c["ckpt"], "student", f"c{c['iter']}")
        else:
            P = L.ephemeral_policy(c["ckpt"], "student", "p2_candidate")
        P["name"] = cid_of(P, c["iter"], self.mode)
        return P

    # ---- rows
    def student_rows(self, P: dict, which: str) -> dict:
        rows = L.student_sel_rows(P, which)
        if self.mode == "p2":
            rows = {k: _rename(r, "RSDSEL_", PREFIX["student"]) for k, r in rows.items()}
        return rows

    def teacher_rows(self) -> dict:
        rows = L.teacher_sel_rows(self.teacher)
        if self.mode == "p2":
            out = {}
            for k, r in rows.items():
                if k in NOT_CONSUMED:
                    continue
                old = "RSDSELT_" if r["name"].startswith("RSDSELT_") else "RSDT_"
                out[k] = _rename(r, old, PREFIX["teacher"])
            rows = out
        return rows

    # ---- stamps
    def reuse(self) -> dict:
        if self._reuse is None:
            self._reuse = reuse_proof(self.pin)
        return self._reuse

    def locate(self, key: str, row: dict) -> tuple:
        """``(stamp | None, problems, source)``."""
        if self.mode == "rsd":
            st = L.row_stamp(row, RSD_OUT)
            return st, [], "rod_speed_dr slot (rsd_lib.row_stamp: newest complete; RSD's runner checked it at launch)"
        slot = M.slot_dir(P2_OUT, row["name"], row["scenario"], row["es"])
        done = M.complete_stamps(slot, pin=self.pin) if self.pin else []
        if done:
            st = done[-1]
            return st, L.row_problems(st, row, self.pin) + S3.raw_problems(st), f"P2 slot {slot}"
        if key in REUSE_SUITE and row["role"] == "teacher":
            r = self.reuse().get(key) or {}
            if r.get("ok"):
                return Path(r["stamp"]), [], f"reused c1 stamp {REUSE_SUITE[key]} (row_problems clean vs this row)"
        return None, [], f"P2 slot {slot} (no complete stamp at the pin)"

    def get(self, rkey: str, row: dict) -> Path | None:
        st, probs, src = self.locate(rkey.split(":", 1)[1].split("@L")[0] if ":" in rkey else rkey, row)
        self.rows[rkey] = {"name": row["name"], "scenario": row["scenario"], "es": row["es"],
                           "ckpt_sha": row["ckpt_sha"], "stamp": None if st is None else str(st), "source": src,
                           "problems": probs[:6]}
        if st is None:
            self.missing.append(rkey)
            return None
        if probs:
            self.invalid.append(rkey)
        return st


def reuse_proof(p) -> dict:
    """c1's RTSW (es1) / RTA (es2) R_suite stamps against the fork's own teacher suite rows (name aside)."""
    rows = L.teacher_sel_rows(c1_policy())
    out = {}
    for key, nm in REUSE_SUITE.items():
        row = dict(rows[key], name=nm)
        slot = M.slot_dir(RT_OUT, nm, row["scenario"], row["es"])
        done = M.complete_stamps(slot, pin=p) if p else []
        rec = {"name": nm, "slot": str(slot), "fork_row": {k: row[k] for k in ("env_file", "level_file", "manifest",
                                                                              "scenario", "args", "es", "strip")},
               "rule": "rsd_lib.row_problems(stamp, the fork's teacher row with name := the RT name, results/p2/pin) "
                       "== [] AND raw npz count == episodes"}
        if not done:
            rec.update(ok=False, stamp=None, problems=[f"no complete stamp at the pin in {slot}"])
        else:
            st = done[-1]
            pr = L.row_problems(st, row, p) + S3.raw_problems(st)
            rec.update(ok=not pr, stamp=str(st), problems=pr[:8], n_stamps=len(done))
        out[key] = rec
    return out


# ============================================================================== the selection read (rsd_gate.ph_sel)
def selection(ctx: Ctx, stop_after_rank: bool = False) -> dict:
    """``rsd_gate.ph_sel``'s read, line for line (its row launches replaced by stamp lookups): the candidates, the
    prescreen records, the rank, the teacher rows, the full blocks in rank order until next_action stops, the gate
    record, the pick. Returns the base doc (RSD sel.json's keys) with ``verdict`` SELECTED | HALT, or NEXT (the next
    block's rows are not all complete), INCOMPLETE (prescreen / teacher rows missing), STOP (a stamp fails a check)."""
    st = L.sel_spec()
    kg = int(st["prescreen_level"]["kg"])
    mf = int(st.get("max_failed_blocks", 2))
    T = ctx.teacher
    cands = ctx.candidates()
    doc = {"stage": "sel", "teacher": T, "student_run": str(ctx.run_dir), "candidates": cands}
    pres, pols, by_iter = {}, {}, {}
    for c in cands:
        if not c.get("ckpt"):
            continue
        P = ctx.policy(c)
        cid = P["name"]
        pols[cid] = P
        by_iter[int(c["iter"])] = cid
        stamps = {k: ctx.get(f"{cid}:{k}", r) for k, r in ctx.student_rows(P, "prescreen").items()}
        lst = {k: ctx.get(f"{cid}:{k}@L{kg:03d}", r) for k, r in ctx.student_rows(P, "prescreen_level").items()}
        if all(v is not None for v in (*stamps.values(), *lst.values())) and not P.get("placeholder"):
            pres[cid] = L.prescreen_read({"iter": c["iter"], "ckpt": c["ckpt"]}, stamps, lst)
    doc["prescreen"] = pres
    doc["_pols"] = pols
    if ctx.invalid:
        return dict(doc, verdict="STOP", why=f"stamps fail the launch checks: {ctx.invalid[:6]}")
    if ctx.missing or len(pres) != len(pols):
        return dict(doc, verdict="INCOMPLETE", stage_due="prescreen", why=f"prescreen rows missing: {ctx.missing[:8]}")
    ranked = [by_iter[int(p["iter"])] for p in L.ss().rank(list(pres.values()))]
    doc.update(ranked=ranked, level_kg=90, max_failed_blocks=mf)
    if stop_after_rank:
        return dict(doc, verdict="RANKED")
    tst = {k: ctx.get(f"teacher:{k}", r) for k, r in ctx.teacher_rows().items()}
    t_missing = [k for k, v in tst.items() if v is None]
    blocks, bstate = [], {}
    act = None
    while True:
        act = L.ss().next_action({"ranked": ranked, "blocks": bstate}, mf)
        if act["do"] != "block":
            break
        cid = act["cand"]
        P = pols[cid]
        n0 = len(ctx.missing)
        stamps = {k: ctx.get(f"{cid}:{k}", r) for k, r in ctx.student_rows(P, "full").items()}
        lst = {k: ctx.get(f"{cid}:{k}@L090", r) for k, r in ctx.student_rows(P, "full_level").items()}
        ctx.block_stamps[cid] = {"full": stamps, "full_level": lst}
        if ctx.invalid:
            return dict(doc, verdict="STOP", why=f"stamps fail the launch checks: {ctx.invalid[:6]}",
                        blocks={b["cand"]: b for b in blocks}, block_state=bstate)
        if len(ctx.missing) > n0:
            return dict(doc, verdict="NEXT", action=act, next={"cand": cid, "block": len(blocks) + 1,
                                                                "due_rows": ctx.missing[n0:]},
                        blocks={b["cand"]: b for b in blocks}, block_state=bstate,
                        teacher_missing=t_missing)
        if t_missing:
            return dict(doc, verdict="INCOMPLETE", stage_due="teacher", why=f"teacher rows missing: {t_missing}",
                        blocks={b["cand"]: b for b in blocks}, block_state=bstate)
        blk = L.full_block_read(cid, stamps, lst, tst, [], lvl=90)
        blocks.append(blk)
        bstate[cid] = "PASS" if blk["pass"] else "FAIL"
    doc.update(blocks={b["cand"]: b for b in blocks}, gate=L.sel_gate_doc(blocks), action=act,
               failed_blocks=[c for c, v in bstate.items() if v == "FAIL"])
    if act["do"] == "selected":
        c = pres[act["cand"]]
        P = pols[act["cand"]]
        pick = {"cand": act["cand"], "ckpt": c["ckpt"], "sha256": L.sha256_file(Path(c["ckpt"])), "iter": c["iter"]}
        if pick["sha256"] != P["sha256"]:
            return dict(doc, verdict="STOP", why=f"the pick's file now hashes {pick['sha256'][:8]}, read as "
                                                 f"{P['sha256'][:8]}")
        if ctx.mode == "rsd":
            pick.update(teacher="TDR", policy="PICK")
        else:
            pick.update(teacher="C1", teacher_sha256=C1_SHA, policy="C1S")
        doc.update(verdict="SELECTED", pick=pick)
        return doc
    doc.update(verdict="HALT", why=act.get("why"))
    return doc


def base_doc(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if not k.startswith("_")}


# ============================================================================== WATCH terrain + field vs c31000
def watch_row(rid: str) -> dict:
    return S3.dprime_row(rid)


def watch_name(P: dict, rid: str) -> str:
    return M.row_name(PREFIX["watch"], P["sha256"], rid)


def watch_lines(P: dict, p=None, served_prefix: str | None = None) -> dict:
    """The WATCH terrain lines of one candidate (report only): success paired with c1's RTSW es1-2 rows."""
    sha_ = P["sha256"]
    prefix = served_prefix or f"{PREFIX['watch']}_{sha_[:8]}_"
    R = S3.P2Rows([RT_OUT / "lr", P2_OUT / "lr"], WATCH_ES, {C1_SHA, sha_}, {sha_})
    out = {}
    for kind, rid in WATCH_KINDS.items():
        pt = readout.pair_tables(R, rid, WATCH_ES, sha_, C1_SHA, "success")
        rec = {"row": rid, "kind": kind, "env_seeds": list(WATCH_ES), "watch": True, "decisive": False,
               "cross_r": R.all[rid]["cross_r"],
               "statistic": "success = complete AND crossed (readout.episode_table at rows_rt cross_r), paired on "
                            "(es, env_index) via readout.pair_tables against c1's RTSW es1-2 rows; "
                            "readout.retention_line: student >= c1 - max(0.05, 2 SE)",
               "stamps": {side: [{"es": f["es"], "status": f["status"], "stamp": f["stamp"]}
                                 for f in pt["stamps"][side]] for side in ("new", "ref")}}
        if pt["status"] != "OK":
            rec.update(status=pt["status"], why=pt.get("why"))
            out[kind] = rec
            continue
        notes = []
        for f in pt["stamps"]["new"]:
            nm = Path(f["stamp"]).parent.parent.parent.name
            if not nm.startswith(prefix):
                notes.append(f"student es{f['es']} served from {nm}, not a {prefix}* row")
        off = [f["stamp"] for f in pt["stamps"]["new"] + pt["stamps"]["ref"] if S3.stamp_build_problems(f["stamp"], p)]
        if off:
            rec.update(status="INVALID", why=f"stamps off the pin: {off}")
            out[kind] = rec
            continue
        ln = readout.retention_line(pt["a"], pt["b"], row=rid, what=f"WATCH {kind} {sha_[:8]} vs c1")
        per_es = {f"es{es}": {"student": readout.share(pt["tables"]["new"][es]["success"]),
                              "c1": readout.share(pt["tables"]["ref"][es]["success"])} for es in WATCH_ES}
        rec.update(status="OK", student=ln["head"], c1=ln["ref"], diff=ln["diff"], se_diff=ln["se_diff"],
                   margin=ln["margin"], binds=ln["binds"], line=ln["line"], n=ln["n"],
                   noninferior=ln["noninferior"], flag=not ln["noninferior"], le_noise=ln["le_noise"], per_es=per_es,
                   notes=notes, comparator_note="c1's es1-2 values are the seeds it was picked on (slopes 0.935 vs "
                                                "0.920 fresh, ledge_down 0.590 vs 0.520): the comparator is "
                                                "inflated by ~0.015-0.07 (P2.md)")
        out[kind] = rec
    return out


def c31000_field_stamps() -> tuple[dict, list]:
    """c31000's recorded RSDSEL field rows (rod_speed_dr results/sel.rows.json), sha / role / seed checked."""
    rec = read_json(RSD_SEL_ROWS, {}) or {}
    out, bad = {}, []
    for fam in L.ss().FIELD_ROWS:
        for es in (1, 2):
            k = f"{fam}@{es}"
            r = rec.get(f"c31000:{k}")
            if not r:
                bad.append(f"no c31000:{k} in {RSD_SEL_ROWS}")
                continue
            st = Path(r["stamp"])
            meta = read_json(st / "run_meta.json", {}) or {}
            if meta.get("checkpoint_sha256") != C31000_SHA or meta.get("role") != "student" or \
                    int((meta.get("env_overrides") or {}).get("seed", -1)) != es:
                bad.append(f"{st}: not c31000 / student / es{es}")
                continue
            out[k] = st
    return out, bad


def field_vs_c31000(stamps: dict) -> dict:
    """P2.md Left item 25 (report only): a candidate's field rows against c31000's recorded RSDSEL rows, the very
    completion_pairs statistic full_block_read applies against the teacher (CROSS-PIN: c31000's rows ran at code
    de3772e2, P2's at 89c1916b; the 2026-11-12 bridge matched c31000 bit for bit across the pins on L090 rows)."""
    TH, S = L.thm(), L.ss()
    ref, bad = c31000_field_stamps()
    if bad:
        return {"status": "MISSING", "why": bad}
    a_all, b_all = {}, {}
    for row in S.FIELD_ROWS:
        a, b = {}, {}
        ks = sorted(k for k in stamps if k.startswith(row + "@"))
        for i, k in enumerate(ks):
            a.update({(i, e): v for e, v in TH.field_episodes(stamps[k]).items()})
            b.update({(i, e): v for e, v in TH.field_episodes(ref[k]).items()})
        a_all[row], b_all[row] = a, b
    pairs = S.completion_pairs(a_all, b_all, L.completion_cfg())
    for p_ in pairs:
        p_["c31000"] = p_.pop("tstar", None)
    return {"status": "OK", "report_only": True, "cross_pin": True, "pairs": pairs,
            "c31000_stamps": {k: str(v) for k, v in ref.items()},
            "rule": "select_student.completion_pairs (student vs c31000 on the same rows / env seeds), "
                    "line c31000 - max(0.05, 2 SE_diff); report only (P2.md Left item 25)"}


# ============================================================================== the P2 read
def p2_ctx(dry: bool | None = None) -> tuple[Ctx, dict]:
    dry = is_dry() if dry is None else dry
    facts = run_facts()
    cands = None
    if not facts.get("finished"):
        rd = Path(facts.get("run_dir") or STUDENT_RUNS / f"<unborn>_{RUN_NAME}")
        grid = nearest_grid(facts.get("projected") or project_saves(C1_ITER), facts.get("start", C1_ITER),
                            L.sel_spec()["offsets"])
        # an unfinished run: every candidate is a PROJECTED stand-in (even a save that exists already is not read
        # before distill.sh's END sentinel and the final save)
        cands = [dict(g, ckpt=str(rd / f"model_{g['iter']}.pt"), placeholder=True, projected=True) for g in grid]
    ctx = Ctx("p2", c1_policy(), Path(facts.get("run_dir") or STUDENT_RUNS), facts.get("start", C1_ITER), p=pin(),
              dry=dry, cands=cands)
    return ctx, facts


def p2_read(write: bool) -> tuple[dict, int]:
    ctx, facts = p2_ctx()
    bad, warn = pin_problems()
    bad += c1_problems()
    bad += facts.get("problems") or []
    if not facts.get("finished"):
        bad.append(f"the distill {RUN_NAME} has not finished (distill.sh END rc={facts.get('distill_end_rc')}, "
                   f"final save model_{facts.get('last_iteration')}.pt)")
    if bad:
        doc = {"verdict": "INCOMPLETE", "stage_due": "inputs", "why": bad}
    else:
        doc = selection(ctx)
    pols = doc.pop("_pols", {}) or {}
    out = {"schema": "p2_sel_v1", "at": M.now_iso(), **base_doc(doc)}
    out["build"] = {"p2_pin": {k: (pin() or {}).get(k) for k in ("code_sha", "eval_sha", "head")},
                    "results_pin": {k: (M.read_pin() or {}).get(k) for k in ("code_sha", "eval_sha")},
                    "pin_problems": pin_problems()[0], "pin_warnings": warn}
    out["teacher_guard"] = {"rule": C1_GUARD, "c1_problems": c1_problems(), "sha256": C1_SHA,
                            "checkpoint": str(C1_CK)}
    out["candidate_grid"] = {k: facts.get(k) for k in ("run_dir", "first_iteration", "last_iteration", "params",
                                                       "distill_end_rc", "finished", "saved", "projected",
                                                       "save_rule", "warnings")}
    out["candidate_grid"]["rule"] = ("select_student.candidate_checkpoints(run_dir, start=the run's first iteration "
                                     f"{facts.get('start')} (= c1's iter, as RSD's start 28991 = TDR's), offsets "
                                     f"{L.sel_spec()['offsets']}): nearest saved checkpoint, ties to the earlier")
    out["candidates_by_sha"] = {cid: {"sha256": P["sha256"], "iter": P["iteration"], "ckpt": P["checkpoint"]}
                                for cid, P in pols.items()}
    out["names"] = {"student": f"{PREFIX['student']}<sha8>_L000 / _L090", "teacher": f"{PREFIX['teacher']}"
                    f"{C1_SHA[:8]}_L000 / _T00_nominal_L000_A", "watch": f"{PREFIX['watch']}_<sha8>_rt_*_b2",
                    "cand_id": "<sha8>_it<iter> (PLAN item 9: never 'c<iter>', which collides with c31000)"}
    out["teacher_suite_reuse"] = ctx._reuse if ctx._reuse is not None else (reuse_proof(pin()) if pin() else None)
    out["teacher_not_run"] = {k: "full_block_read never reads the teacher's side22 row (side22 FAST/FAST is "
                                 "student-absolute)" for k in NOT_CONSUMED}
    out["rows"] = ctx.rows
    watch = {}
    blocks_seen = list((doc.get("blocks") or {}).keys())
    if doc.get("verdict") == "NEXT":
        blocks_seen.append(doc["next"]["cand"])
    for cid in blocks_seen:
        P = pols.get(cid)
        if P is not None and not P.get("placeholder"):
            try:
                watch[cid] = watch_lines(P, pin())
            except Exception as exc:  # noqa: BLE001
                watch[cid] = {"status": "ERROR", "why": f"{type(exc).__name__}: {exc}"}
    out["watch_terrain"] = watch
    out["watch_rule"] = ("P2.md settled 6c / 7 Q1: WATCH only -- reported per block, never fails a block or moves the "
                         "pick; T at es9+ is the only decisive terrain gate")
    if doc.get("verdict") == "HALT":
        out["field_vs_c31000"] = {}
        for cid, bs in ctx.block_stamps.items():
            try:
                out["field_vs_c31000"][cid] = field_vs_c31000(bs["full"])
            except Exception as exc:  # noqa: BLE001
                out["field_vs_c31000"][cid] = {"status": "ERROR", "why": f"{type(exc).__name__}: {exc}"}
    out["sentinels"] = sentinels(out)
    if write:
        M.write_json(SEL_JSON, out)
    return out, DECIDE_RC.get(out.get("verdict"), 12)


def _watch_word(w: dict) -> str:
    if not w or w.get("status") == "ERROR":
        return "n/a"
    return ",".join(f"{k}:{'MISSING' if v.get('status') != 'OK' else ('%.3f/%.3f%s' % (v['student'], v['c1'], ' FLAG' if v['flag'] else ''))}"
                    for k, v in w.items())


def sentinels(out: dict) -> list:
    v = out.get("verdict")
    s = []
    for cid, w in (out.get("watch_terrain") or {}).items():
        for kind, ln in (w or {}).items() if isinstance(w, dict) and "status" not in w else []:
            if ln.get("status") == "OK":
                s.append(f"P2_SEL_WATCH {cid} {kind}: student {ln['student']:.3f} c1 {ln['c1']:.3f} diff "
                         f"{ln['diff']:+.3f} line {ln['line']:.3f} (m {ln['margin']:.3f} {ln['binds']}) "
                         f"{'FLAG' if ln['flag'] else 'ok'} (WATCH only)")
            else:
                s.append(f"P2_SEL_WATCH {cid} {kind}: {ln.get('status')} {str(ln.get('why'))[:160]} (WATCH only)")
    if v == "SELECTED":
        pk = out["pick"]
        blk = (out.get("blocks") or {}).get(pk["cand"]) or {}
        s.append(f"P2_SEL PICK cand={pk['cand']} sha256={pk['sha256']} iter={pk['iter']} ckpt={pk['ckpt']} "
                 f"teacher=c1:{C1_SHA[:8]} blocks={len(out.get('blocks') or {})} "
                 f"tracking_watch_flags={len(blk.get('watch_flags') or [])} "
                 f"terrain_watch={_watch_word((out.get('watch_terrain') or {}).get(pk['cand']))} json={SEL_JSON}")
    elif v == "HALT":
        s.append(f"P2_SEL HALT why={out.get('why')} failed_blocks={out.get('failed_blocks')} "
                 f"(fallback: a blended mix, newly pre-registered; notify and ask) json={SEL_JSON}")
    elif v == "NEXT":
        n = out["next"]
        s.append(f"P2_SEL NEXT cand={n['cand']} block={n['block']} of <= {out.get('max_failed_blocks', 2)} "
                 f"due_rows={len(n['due_rows'])} json={SEL_JSON}")
    elif v == "STOP":
        s.append(f"P2_SEL STOP {out.get('why')} (decide-record-notify) json={SEL_JSON}")
    else:
        s.append(f"P2_SEL INCOMPLETE stage={out.get('stage_due')} {str(out.get('why'))[:300]} json={SEL_JSON}")
    return s


# ============================================================================== plans (rt_machinery Steps)
def row_step(label: str, row: dict, note: str, placeholder_row: bool = False) -> M.Step:
    st = M.gate_step(label, name=row["name"], ck=row["ckpt"], row=row, es=row["es"], tp_out=P2_OUT)
    st.env["TP_ROW_TIMEOUT"] = str(ROW_TIMEOUT_S)
    if row.get("strip"):
        st.env["TP_STRIP_KEYS"] = " ".join(row["strip"])
    tp_out, nm, sc, e = st.slot
    st.note = (("DRY PLACEHOLDER (the save does not exist yet); " if placeholder_row else "")
               + f"{note}; expected stamp {M.slot_dir(tp_out, nm, sc, e)}/<YYYY-MM-DD_HH-MM-SS>/")
    return st


def _checks(prefix: str, block: str) -> list:
    return [S3.build_check_step(prefix),
            M._cpu(f"{prefix}.inputs.check", HERE / "p2_select.py", "--inputs-check", "--block", block, stop=True,
                   dry=True, note="P2 pin == results/pin.json, c1 sha/iter, the run finished (prescreen/full), the "
                                  "block's own inputs")]


def _lab(fam_es: str) -> str:
    fam, es = fam_es.split("@")[:2]
    return f"{fam}.es{es}"


def plan_teacher() -> list:
    ctx = Ctx("p2", c1_policy(), STUDENT_RUNS, C1_ITER, p=pin())
    steps = _checks("SELT", "teacher")
    rp = ctx.reuse() if pin() else {}
    for k, r in ctx.teacher_rows().items():
        if k in REUSE_SUITE and (rp.get(k) or {}).get("ok"):
            continue
        steps.append(row_step(f"SELT.{_lab(k)}", r, f"c1 teacher row {k} (Student-Play, role teacher)"))
    return steps


def plan_prescreen() -> list:
    ctx, _f = p2_ctx()
    kg = int(L.sel_spec()["prescreen_level"]["kg"])
    steps = _checks("SELP", "prescreen")
    for c in ctx.candidates():
        if not c.get("ckpt"):
            continue
        P = ctx.policy(c)
        ph = bool(P.get("placeholder"))
        for k, r in ctx.student_rows(P, "prescreen").items():
            steps.append(row_step(f"SELP.{P['name']}.{_lab(k)}", r, f"prescreen {P['name']} {k}", ph))
        for k, r in ctx.student_rows(P, "prescreen_level").items():
            steps.append(row_step(f"SELP.{P['name']}.{_lab(k)}.L{kg:03d}", r, f"prescreen {P['name']} {k} @ {kg} kg",
                                  ph))
    return steps


def full_target() -> tuple[dict | None, str]:
    """(the candidate policy whose full block is due, why). Real: the selection state must be NEXT. Dry without a
    rank: the first grid candidate stands in (shape only)."""
    ctx, facts = p2_ctx()
    if facts.get("finished") and not c1_problems() and pin():
        doc = selection(ctx)
        pols = doc.get("_pols") or {}
        if doc["verdict"] == "NEXT":
            return pols[doc["next"]["cand"]], f"selection state NEXT: block {doc['next']['block']}"
        if not is_dry():
            return None, f"selection state {doc['verdict']} ({str(doc.get('why'))[:200]}): no full block is due"
        why = f"selection state {doc['verdict']}"
    else:
        why = "the distill has not finished (no rank yet)"
    if not is_dry():
        return None, why
    c = next(c for c in ctx.candidates() if c.get("ckpt"))
    return ctx.policy(c), f"DRY stand-in: {why}; the first grid candidate shows the block's shape"


def plan_full() -> list:
    P, why = full_target()
    steps = _checks("SELF", "full")
    if P is None:
        raise ValueError(f"no full block is due: {why}")
    ctx = Ctx("p2", c1_policy(), STUDENT_RUNS, C1_ITER, p=pin())
    ph = bool(P.get("placeholder"))
    for k, r in ctx.student_rows(P, "full").items():
        steps.append(row_step(f"SELF.{P['name']}.{_lab(k)}", r, f"full block {P['name']} {k} ({why})", ph))
    for k, r in ctx.student_rows(P, "full_level").items():
        steps.append(row_step(f"SELF.{P['name']}.{_lab(k)}.L090", r, f"full block {P['name']} {k} @ 90 kg", ph))
    for kind, rid in WATCH_KINDS.items():
        for es in WATCH_ES:
            row = dict(watch_row(rid), name=watch_name(P, rid), ckpt=P["checkpoint"], es=es)
            steps.append(row_step(f"SELW.{P['name']}.{kind}.es{es}", row,
                                  f"WATCH terrain {kind} b2 es{es} (role student; report only)", ph))
    return steps


PLANS = {"p2sel_teacher": plan_teacher, "p2sel_prescreen": plan_prescreen, "p2sel_full": plan_full}


def plan_sel(block: str) -> list:
    steps = PLANS[block]()
    labels = [s.label for s in steps]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate step labels")
    slots = [s.slot for s in steps if s.slot]
    if len(slots) != len(set(slots)):
        raise RuntimeError("two rows share an output slot")
    lo, hi = EXPECT_SLOTS[block]
    if not lo <= len(slots) <= hi or (block == "p2sel_prescreen" and len(slots) % 5):
        raise RuntimeError(f"{block}: {len(slots)} rows, P2 pre-registers {lo}-{hi}"
                           + (" (5 per candidate)" if block == "p2sel_prescreen" else ""))
    if any(nm.startswith(("RSD", "RT")) or not nm.startswith(("P2SEL", "P2SW")) for _o, nm, _s, _e in slots):
        raise RuntimeError("a P2 selection row is not named P2SEL_* / P2SELT_* / P2SW_*")
    bad = rt_short.slot_problems(steps)
    if bad:
        raise ValueError("slot collision: " + "; ".join(bad))
    return steps


def install() -> None:
    if getattr(M, "_p2sel_installed", False):
        return
    M.BLOCKS = tuple(PLANS)
    M.plan = lambda block, *a, **k: plan_sel(block)
    M.STOP_FILE = S3._AnyStop(S3.P2_STOP, M.STOP_FILE)
    if os.environ.get("P2_DDC_GROWTH_MAX_GB"):
        M.DDC_GROWTH_MAX_GB = float(os.environ["P2_DDC_GROWTH_MAX_GB"])
    M._p2sel_installed = True


def print_slots(block: str) -> list:
    steps = plan_sel(block)
    rows = [s for s in steps if s.slot]
    p = pin()
    new = 0
    say(f"# P2 {block}: {len(rows)} rows planned -> {P2_OUT}/lr")
    for s in rows:
        tp_out, name, sc, es = s.slot
        slot = M.slot_dir(tp_out, name, sc, es)
        done = M.complete_stamps(slot, pin=p) if p else []
        new += 0 if done else 1
        env = " ".join(f"{k}={v}" for k, v in sorted(s.env.items()) if k != "TP_STRIP_KEYS")
        strip = len((s.env.get("TP_STRIP_KEYS") or "").split())
        say(f"#   {s.label:<44} {name:<36} {sc:<20} es{es:<2} ck={Path(s.argv[3]).name} strip={strip} {env} "
            f"[{len(done)} complete at the pin{' (skipped)' if done else ''}]")
    say(f"# P2 {block}: {len(rows)} planned, {new} to launch now (shared slots skip on complete)")
    return steps


def run_block(blk: str, rest: list) -> int:
    """rt_queue.main for one real block (p2_step3.run_block's rc mapping: REFUSED 20 / HALT 21)."""
    try:
        steps = plan_sel(blk)
        row_labels = {s.label for s in steps if s.slot}
        cands = sorted({s.label.split(".")[1] for s in steps if s.slot and s.label.startswith("SELF.")})
        if cands:
            say(f"P2_SEL_PLAN full cand={cands[0]} rows={len(row_labels)}")
    except (ValueError, RuntimeError, OSError) as exc:
        say(f"P2_SEL_PLAN REFUSED block={blk}: {exc}")
        row_labels = set()
    seen = {"rows": [], "done": None}
    child0, say0 = Q.run_child, Q.say

    def child(argv, env, log):
        if log is not None and Path(log).stem in row_labels:
            seen["rows"].append(Path(log).stem)
        return child0(argv, env, log)

    def say_(line: str) -> None:
        if line.startswith("RTQ_DONE "):
            seen["done"] = line
        say0(line)

    Q.run_child, Q.say = child, say_
    try:
        rc = Q.main(rest)
    finally:
        Q.run_child, Q.say = child0, say0
    if rc == 0:
        return 0
    done = seen["done"] or ""
    if "DerivedDataCache grew" in done or "guard stopped the queue" in done:
        say(f"P2_SEL HALT_QUEUE block={blk} rc={HALT_RC} rows_launched={len(seen['rows'])} -- {done}")
        return HALT_RC
    if not seen["rows"]:
        say(f"P2_SEL REFUSED block={blk} rc={REFUSED_RC} (rt_queue rc {rc}; no row launched) -- {done}")
        return REFUSED_RC
    return rc


# ============================================================================== --inputs-check
def inputs_check(block: str) -> int:
    bad, warn = pin_problems()
    bad += c1_problems()
    info = []
    facts = run_facts()
    if block in ("prescreen", "full"):
        bad += facts.get("problems") or []
        if not facts.get("finished"):
            (warn if is_dry() else bad).append(
                f"the distill {RUN_NAME} has not finished (END rc={facts.get('distill_end_rc')}; saved "
                f"{len(facts.get('saved') or [])} of {len(facts.get('projected') or [])} projected; candidates are "
                f"projected stand-ins)")
        for w in facts.get("warnings") or []:
            warn.append(w)
        info.append(f"run {facts.get('run_dir')} first {facts.get('first_iteration')} last "
                    f"{facts.get('last_iteration')} saved {len(facts.get('saved') or [])}")
    if block == "teacher":
        if pin():
            for k, r in reuse_proof(pin()).items():
                info.append(f"teacher {k}: reuse {r['name']} {'OK' if r['ok'] else 'NO -- the row runs'} "
                            f"{r.get('stamp')} {r.get('problems')[:2] if r.get('problems') else ''}")
    if block == "full":
        refused = set(S3.refused_keys())
        for rid in WATCH_KINDS.values():
            ef = RS.OMNI / S3.rows_by_id()[rid]["env_file"]
            hit = sorted(set(M.krabby_of(ef)) & refused)
            if hit:
                bad.append(f"{ef} carries student-refused runner keys {hit} (the WATCH rows strip nothing)")
        if not bad:
            P, why = full_target()
            if P is None:
                bad.append(why)
            else:
                info.append(f"full block due: {P['name']} ({why})")
    for w in warn:
        say(f"P2_SEL_INPUTS WARN {w}")
    for i in info:
        say(f"P2_SEL_INPUTS INFO {i}")
    say(f"P2_SEL_INPUTS {'OK' if not bad else 'BAD'} block={block}" + (f" -- {'; '.join(bad)}" if bad else ""))
    return 0 if not bad else 1


# ============================================================================== --rank / --decide
def cmd_rank() -> int:
    ctx, facts = p2_ctx()
    bad = pin_problems()[0] + c1_problems() + (facts.get("problems") or [])
    if not facts.get("finished"):
        bad.append(f"the distill {RUN_NAME} has not finished")
    if bad:
        say(f"P2_SEL_RANK INCOMPLETE -- {'; '.join(bad)}")
        return 12
    doc = selection(ctx, stop_after_rank=True)
    if doc["verdict"] != "RANKED":
        say(f"P2_SEL_RANK {doc['verdict']} -- {doc.get('why')}")
        return DECIDE_RC.get(doc["verdict"], 12)
    pols = doc["_pols"]
    for cid in doc["ranked"]:
        p_ = doc["prescreen"][cid]
        say(f"P2_SEL_RANK_ROW {cid} sha256={pols[cid]['sha256']} misses={p_['misses']} field_hard={p_['field_hard']} "
            f"mean_abs_dev={p_['mean_abs_dev']}")
    say(f"P2_SEL_RANK ranked={','.join(doc['ranked'])} first={doc['ranked'][0]}")
    return 0


def cmd_decide(write: bool) -> int:
    out, rc = p2_read(write)
    for ln in out["sentinels"]:
        say(ln)
    say(f"# {'wrote' if write else 'dry: not written'} {SEL_JSON} verdict={out.get('verdict')}")
    return rc


# ============================================================================== --fork-check
def rsd_teacher() -> dict:
    """rsd_gate.ph_sel's T = rsd_lib.policy('TDR') from RSD's own registry (re-hashed)."""
    reg = (read_json(RSD_POLICIES, {}) or {}).get("policies") or {}
    have = reg["TDR"]
    path = Path(have["checkpoint"])
    s = sha(path)
    if s != have["sha256"]:
        raise RuntimeError(f"TDR {path} hashes {s[:8]} != registry {have['sha256'][:8]}")
    return {"name": "TDR", "checkpoint": str(path), "sha256": s, "sha8": s[:8], "role": have["role"],
            "iteration": L.its_of(path)}


def num_compare(a, b, path: str, out: list, cnt: dict, tol: float = FORK_TOL) -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        cnt["bools"] += 1
        if a is not b and a != b:
            out.append(f"{path}: mine {a!r} != record {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        cnt["numbers"] += 1
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return
        d = abs(fa - fb) if math.isfinite(fa) and math.isfinite(fb) else (0.0 if fa == fb else math.inf)
        cnt["max_abs_diff"] = max(cnt["max_abs_diff"], d)
        if d > tol:
            out.append(f"{path}: mine {a!r} != record {b!r} (|d| {d:.3g})")
    elif isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: {'missing in mine' if k not in a else 'missing in record'}")
            else:
                num_compare(a[k], b[k], f"{path}/{k}", out, cnt, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            num_compare(x, y, f"{path}[{i}]", out, cnt, tol)
    else:
        cnt["other"] += 1
        if a != b:
            out.append(f"{path}: mine {str(a)[:90]!r} != record {str(b)[:90]!r}")


def fork_check(out_path: Path | None) -> int:
    rec = read_json(RSD_SEL_JSON)
    if rec is None:
        say(f"P2_SEL_FORK FAIL no {RSD_SEL_JSON}")
        return 1
    dirs = sorted(RSD_STUDENT_RUN.parent.glob("*_RSD_ST_s19"))
    T = rsd_teacher()
    ctx = Ctx("rsd", T, RSD_STUDENT_RUN, T["iteration"])
    doc = selection(ctx)
    mine = L._norm(base_doc(doc))
    diffs, cnt = [], {"numbers": 0, "bools": 0, "other": 0, "max_abs_diff": 0.0}
    num_compare(mine, rec, "", diffs, cnt)
    rows_rec = read_json(RSD_SEL_ROWS, {}) or {}
    stamp_diffs = [f"{k}: located {v['stamp']} != recorded {(rows_rec.get(k) or {}).get('stamp')}"
                   for k, v in ctx.rows.items() if not k.startswith("teacher:") and k in rows_rec
                   and v["stamp"] != rows_rec[k]["stamp"]]
    unrecorded = sorted(k for k in ctx.rows if not k.startswith("teacher:") and k not in rows_rec)
    same_pick = (mine.get("pick") or {}).get("sha256") == (rec.get("pick") or {}).get("sha256") and \
        (mine.get("pick") or {}).get("cand") == (rec.get("pick") or {}).get("cand")
    ok = not diffs and same_pick and not stamp_diffs and not ctx.missing
    res = {"schema": "p2_sel_fork_check_v1", "at": M.now_iso(), "ok": ok, "tol": FORK_TOL, "counts": cnt,
           "diffs": diffs[:60], "n_diffs": len(diffs), "same_pick": same_pick, "pick": mine.get("pick"),
           "record_pick": rec.get("pick"), "verdict": mine.get("verdict"), "record_verdict": rec.get("verdict"),
           "stamp_diffs_vs_sel_rows_json": stamp_diffs, "rows_not_in_sel_rows_json": unrecorded,
           "missing": ctx.missing, "student_run_dirs": [str(d) for d in dirs], "teacher": T,
           "constants": {"teacher": "TDR (RSD results/policies.json)", "student_run": str(RSD_STUDENT_RUN),
                         "start": T["iteration"], "stamps": str(RSD_OUT / "lr"), "names": "RSDSEL_ / RSDT_ / RSDSELT_",
                         "cand_id": "c<iter>", "pick": "teacher TDR, policy PICK"}}
    if out_path:
        M.write_json(out_path, res)
    say(f"P2_SEL_FORK {'PASS' if ok else 'FAIL'} numbers={cnt['numbers']} bools={cnt['bools']} other={cnt['other']} "
        f"max_abs_diff={cnt['max_abs_diff']:.3g} diffs={len(diffs)} same_pick={same_pick} "
        f"pick={(mine.get('pick') or {}).get('cand')} {str((mine.get('pick') or {}).get('sha256'))[:8]} "
        f"verdict={mine.get('verdict')} stamp_diffs={len(stamp_diffs)} missing={len(ctx.missing)}")
    for d in diffs[:20]:
        say(f"  diff {d}")
    return 0 if ok else 1


# ============================================================================== --self-test
class _ReplayCtx(Ctx):
    """Self-test only: the P2 code path (sha-named candidate ids, P2SEL_ / P2SELT_ rows, side22 teacher row dropped,
    the C1-style pick record) read on RSD's recorded stamps (each P2 name mapped back to its RSD twin)."""

    BACK = (("P2SELT_", "_T00_nominal_", "RSDT_"), ("P2SELT_", "", "RSDSELT_"), ("P2SEL_", "", "RSDSEL_"))

    def locate(self, key, row):
        nm = row["name"]
        for new, must, old in self.BACK:
            if nm.startswith(new) and must in nm:
                nm = old + nm[len(new):]
                break
        st = L.row_stamp(dict(row, name=nm), RSD_OUT)
        return st, [], f"replay: {row['name']} -> {nm}"


def replay_p2_path() -> tuple[bool, str]:
    rec = read_json(RSD_SEL_JSON, {}) or {}
    T = rsd_teacher()
    ctx = _ReplayCtx("p2", T, RSD_STUDENT_RUN, T["iteration"])
    doc = selection(ctx)
    pk = doc.get("pick") or {}
    want = f"{C31000_SHA[:8]}_it31000"
    ok = doc.get("verdict") == "SELECTED" and pk.get("cand") == want and pk.get("sha256") == C31000_SHA
    mine = L._norm((doc.get("blocks") or {}).get(want) or {})
    ref = L._norm((rec.get("blocks") or {}).get("c31000") or {})
    mine.pop("cand", None)
    ref.pop("cand", None)
    diffs, cnt = [], {"numbers": 0, "bools": 0, "other": 0, "max_abs_diff": 0.0}
    num_compare(mine, ref, "", diffs, cnt)
    side22_teacher = [k for k in ctx.rows if k == "teacher:side22@1"]
    ok = ok and not diffs and not side22_teacher and not ctx.missing
    sent = sentinels({**base_doc(doc), "watch_terrain": {}})
    return ok, (f"verdict {doc.get('verdict')} pick {pk.get('cand')} {str(pk.get('sha256'))[:8]} ranked "
                f"{doc.get('ranked')}; block numbers vs RSD c31000: {cnt['numbers']} compared, {len(diffs)} diffs; "
                f"teacher side22 read: {bool(side22_teacher)}; sentinel: {sent[-1][:140]}")


def self_test() -> int:
    fails = []

    def check(name, ok, detail=""):
        say(f"P2_SEL_SELFTEST {'PASS' if ok else 'FAIL'} {name}" + (f" -- {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    # 1. the save-schedule projection + the nearest-save grid on RSD's real run
    rsd_saved = sorted(i for p in RSD_STUDENT_RUN.glob("model_*.pt") if (i := L.its_of(p)) is not None)
    rsd_proj = project_saves(28991)
    check("save projection == RSD_ST_s19's saved points", rsd_proj == rsd_saved,
          f"{len(rsd_proj)} projected vs {len(rsd_saved)} saved")
    rsd_grid = [c["iter"] for c in L.ss().candidate_checkpoints(RSD_STUDENT_RUN, 28991, L.sel_spec()["offsets"])]
    rec = read_json(RSD_SEL_JSON, {}) or {}
    check("candidate_checkpoints(RSD run) == RSD sel.json candidates", rsd_grid == [c["iter"] for c in rec.get(
        "candidates") or []], f"{rsd_grid}")
    check("nearest_grid(projection) == candidate_checkpoints (RSD)",
          [g["iter"] for g in nearest_grid(rsd_proj, 28991, L.sel_spec()["offsets"])] == rsd_grid)
    facts = run_facts()
    pg = nearest_grid(facts.get("projected") or project_saves(C1_ITER), facts.get("start", C1_ITER),
                      L.sel_spec()["offsets"])
    say(f"P2_SEL_SELFTEST INFO RT_ST_s19 run {facts.get('run_dir')} params {facts.get('params')} first "
        f"{facts.get('first_iteration')} last {facts.get('last_iteration')} saved now {facts.get('saved')} "
        f"END rc {facts.get('distill_end_rc')}")
    say(f"P2_SEL_SELFTEST INFO projected P2 grid (targets -> saves): "
        + ", ".join(f"{g['target']}->{g['iter']}" for g in pg))
    check("run facts carry no problem", not facts.get("problems"), str(facts.get("problems")))
    # 2. the c1 suite reuse proof
    rp = reuse_proof(pin())
    check("c1 suite es1/es2 reuse proof (RTSW / RTA vs the fork's teacher rows)",
          all(r["ok"] for r in rp.values()), "; ".join(f"{k}: {r.get('stamp')} {r.get('problems')}"
                                                       for k, r in rp.items()))
    # 3. the WATCH machinery on c31000's recorded d-prime rows == step3.json's d' (sign flipped, SE equal)
    s3 = read_json(RESULTS_P2 / "step3.json", {}) or {}
    P = {"sha256": C31000_SHA, "sha8": C31000_SHA[:8]}
    w = watch_lines(P, pin(), served_prefix=f"P2D_{C31000_SHA[:8]}_")
    for kind, ln in w.items():
        ref = ((((s3.get("dprime") or {}).get("kinds") or {}).get(kind) or {}).get("d_c1_minus_c31000") or {})
        ok = ln.get("status") == "OK" and ref.get("status") == "OK" and abs(ln["diff"] + ref["diff"]) <= 1e-12 \
            and abs(ln["se_diff"] - ref["se"]) <= 1e-12
        check(f"WATCH read on c31000's P2D {kind} rows == step3 d' (diff {ln.get('diff')}, se {ln.get('se_diff')})",
              ok, f"step3 {ref.get('diff')} / {ref.get('se')}; line {ln.get('line')} flag {ln.get('flag')}")
    # 4. the field-vs-c31000 report on RSD's c30000 full-block-shaped stamps (c30000 has field_hard only; use
    #    c31000's own rows: the identity read must give diff 0 and pass)
    ref_st, bad = c31000_field_stamps()
    check("c31000 recorded field rows found", not bad, str(bad))
    if not bad:
        fv = field_vs_c31000(ref_st)
        check("field_vs_c31000(c31000 itself) == identity (diff 0, all pass)",
              fv["status"] == "OK" and all(abs(p_["student"] - p_["c31000"]) < 1e-15 and p_["pass"]
                                           for p_ in fv["pairs"]),
              "; ".join(f"{p_['row']} {p_['student']:.3f}" for p_ in fv["pairs"]))
    # 5. names: candidate ids never 'c<iter>'; prefixes distinct from RSD / RT
    check("candidate ids are by sha (never c<iter>)",
          cid_of({"sha8": "155b5981"}, 31000, "p2") == "155b5981_it31000" and not re.match(r"^c\d+$", cid_of(
              {"sha8": "deadbeef"}, 31000, "p2")))
    # 6. the P2 code path on RSD's stamps: the same pick, by sha, and the same block numbers
    ok, why = replay_p2_path()
    check("P2 code path (sha ids, P2 names, no teacher side22) replayed on RSD stamps == RSD's pick + block", ok, why)
    # 7. the dry slot counts (plan only, no launch)
    MODE["dry"] = True
    os.environ["P2_DRY"] = "1"
    for blk in BLOCKS.values():
        try:
            steps = plan_sel(blk)
            n = len([s for s in steps if s.slot])
            check(f"plan {blk}: {n} rows within the pre-registered {EXPECT_SLOTS[blk]}", True)
        except Exception as exc:  # noqa: BLE001
            check(f"plan {blk}", False, f"{type(exc).__name__}: {exc}")
    say(f"P2_SEL_SELFTEST {'PASS' if not fails else 'FAIL'} ({len(fails)} failed: {fails})")
    return 0 if not fails else 1


# ============================================================================== main
def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    ap.add_argument("--block", choices=tuple(BLOCKS), default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--from", dest="from_label", default=None)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--inputs-check", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--decide", action="store_true")
    ap.add_argument("--fork-check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="--fork-check: also write the result here")
    a = ap.parse_args(argv)
    if S3._CALLER_RT_TP_OUT and Path(S3._CALLER_RT_TP_OUT) != P2_OUT:
        say(f"P2_SEL REFUSED caller RT_TP_OUT={S3._CALLER_RT_TP_OUT} (this driver writes only {P2_OUT})")
        return 2
    MODE["dry"] = a.dry
    S3.MODE["dry"] = a.dry
    if a.dry:
        os.environ["P2_DRY"] = "1"
    if a.inputs_check:
        return inputs_check(a.block or "teacher")
    if a.rank:
        return cmd_rank()
    if a.decide:
        return cmd_decide(write=not a.dry)
    if a.fork_check:
        return fork_check(a.out)
    if a.self_test:
        return self_test()
    if a.block is None and not a.dry:
        ap.error("one of --block / --dry / --inputs-check / --rank / --decide / --fork-check / --self-test")
    install()
    blocks = [BLOCKS[a.block]] if a.block else list(BLOCKS.values())
    rc_all = 0
    for blk in blocks:
        rest = ["--block", blk]
        if a.dry:
            rest.append("--dry")
        if a.list:
            rest.append("--list")
        if a.from_label:
            rest += ["--from", a.from_label]
        for o in a.only or []:
            rest += ["--only", o]
        if a.dry or a.list:
            try:
                print_slots(blk)
            except (ValueError, RuntimeError, OSError) as exc:
                say(f"# P2 {blk}: plan refused: {exc}")
            rc = Q.main(rest)
        else:
            why = c1_problems()
            if why:
                say(M.done_line(blk, 1, f"(c1 identity: {'; '.join(why)})"))
                say(f"P2_SEL REFUSED block={blk} rc={REFUSED_RC} (c1 identity; no row launched)")
                return REFUSED_RC
            rc = run_block(blk, rest)
        rc_all = rc_all or rc
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
