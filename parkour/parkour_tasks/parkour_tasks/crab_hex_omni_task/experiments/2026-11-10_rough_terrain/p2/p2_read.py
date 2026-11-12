#!/usr/bin/env python3
"""P2 steps 7-8 (p2/P2.md Gates, Class handling, Decision rules): the decisive es9+ student read -- the row plan, the
one-sided relaunch read and the verdict. Rows run through the rough-terrain campaign's OWN row machinery (the
``p2_step3.py`` pattern: ``scripts/rt_machinery`` blocks patched, ``scripts/rt_queue.main`` executes); nothing here
launches Kit itself.

    p2_read.py --dry [--block front|rows|relaunch]   (CPU) the slot plan, the counts and every command; no launch
    p2_read.py --queue-dry --block B                 rt_queue's own dry pass of block B (gate_row.sh TP_ROW_DRY=1)
    p2_read.py --block front|rows|relaunch           GPU: run the block (rt_queue: flock, busy wait, guards, skip)
    p2_read.py --inputs-check --block B              (CPU) the block's pre-flight (also the first step of every block)
    p2_read.py --near-miss [--out JSON]              (CPU) -> results/p2/read_near_miss.json; P2_READ NEARMISS ...
                                                     exit 0 NONE / 10 DUE / 11 STOP / 12 INCOMPLETE / 13 HALT
    p2_read.py --verdict [--out JSON]                (CPU) -> results/p2/read.json; P2_READ VERDICT ...
                                                     exit 0 computed / 12 INCOMPLETE / 13 HALT / 14 PENDING
    common: [--pairs-cache JSON] (classflip results cached by stamp pair; default results/p2/read_pairs.json)

POLICIES (named by sha256 in every record: a P2 save ``model_31000.pt`` would collide with 'c31000'):
* C1S = the selection pick, ``results/p2/sel.json`` (verdict SELECTED, ``pick.sha256`` + ``pick.ckpt``, the sha
  recomputed from the file; committed and clean): every mode except --dry refuses without it (--dry shows a
  ``DRY PLACEHOLDER`` pick); role student.
* c31000 = RSD_ST_s19 model_31000.pt from MAIN logs by path + sha 155b5981 (+ iter 31000; ``p2_step3``); student.
* c1 = RT_s19 model_29500 (6d1f29be, this worktree's logs) and TDR = policy/rsd_9_p3b_rodspeed_dr/model_28991.pt
  (03c3e472): the Q3 teacher rows only (report only), role teacher.

ROW PLAN (84 rows = P2.md step 7's 36 base + 20 second launches (Q4) + 20 kind rows (Q2) + 8 teacher rows (Q3)):
* T (decisive): rt_slopes_b2 / rt_ledge_down_b2, es9-12, C1S and c31000, each launched TWICE up front (32 rows);
* T-reg (decisive): rt_ledge_up_b2 / rt_pits_b2 / rt_rubble_b3 (the frozen gate bands), es9-10, both students (12);
* N (decisive): R_field_hard es9-12 (8), R_field_light (L000) es9-10 (4), R_x07_L090_field_light es9-10 TWICE (8);
* Tr (decisive): R_suite (L000) es9-10, both students (4);
* report only: rt_hills_b3 / rt_gravel_b3 es9-10, both students (8; the gravel rows carry rows_rt's
  KRABBY_RT_FRICTION contract, checked by readout.find); Q3: c1 + TDR on the T kinds at es9-10 (8).
Every row is rows_rt.json's row with ONLY ``--policy-role`` swapped for a student (env file, level file, manifest,
scenario and the other args unchanged). Stamps go to ``<MAIN>/parkour/outputs/omni_plan/rough_terrain_p2/lr`` as
``P2R_<sha8>_<row>`` (+ ``_L2`` second launch, ``_R1`` one-sided relaunch). Order (Left 22 / Q4): the C1S es9 rows of
ledge_up, pits, rubble, hills and gravel FIRST (block ``front``, so a role or env refusal costs minutes), then T,
T-reg, N, Tr, report, Q3, with the two students interleaved within each (kind, es) (block ``rows`` lists all 84; the
front 5 are skipped there once complete). Block ``relaunch`` = the rows a FRESH near-miss read marks due (``_R1``).
Each block: ``build_check.py check --pin-path results/p2/pin.json`` (Left 24) and ``--inputs-check`` first.

THE READ (``P2ReadRows``, the readout_fresh pattern: ``readout.Rows`` with ONLY ``found()`` overridden): logical env
seeds 1-4 map to real 9-12 (one point; so the suite rows es9 / es10 reach zero_kg_check as ``suite es1`` / ``es2``,
the keys of v1.2's drift dict, Left 5); the index scans BOTH output roots (rough_terrain for c1 / TDR stamps,
rough_terrain_p2 for the student and the new teacher rows) restricted to the seed dirs 9-12 before any run_meta is
opened (no es1-8 stamp is read); a student sha's row args are role-swapped; an optional launch name restricts a lookup
to one name dir, so each launch is served separately. Served stamps must carry the real seed (run_meta + tp_row),
the sha, the P2 pin's build, an intact raw/ (npz count == episodes, Left 11).

POOLED COMPUTATION (P2.md Class handling, exactly): per (es, env_index) key the outcome is the mean over that key's
launches (0, 0.5 or 1; float(bool) per launch); share = the plain mean over the keys; SE =
``thresholds.paired_completion_se`` on the fractional dicts; gain / CI / regress = ``readout.g_line``'s clauses on
those numbers (gain: diff >= max(0.10, 3 SE) - eps and diff - 1.96 SE >= 0.05 - eps and not le_noise; regress: diff <
-max(0.05, 2 SE) - eps and not le_noise; the rule read from thresholds_rough_v1.json, as readout.gates does); N =
select_student.completion_margin (readout.retention_line's margin) on those numbers. Pooled dicts NEVER pass through
``readout.share`` / ``paired`` / ``retention_line`` (they cast to bool: 0.5 -> 1) and launches are NEVER keyed
(launch, env_index), so n is unchanged and a bitwise-equal second launch reproduces the single-launch line exactly
(p2_read_check.py proves it). le_noise cannot bind at the 0.10 / 0.05 floors (Left 13; recorded).

GATES (read at step 8): T per kind KEEPS (2 gains) / PARTIAL (1) / LOST (0), a T regress recorded beside it;
T-reg per kind no-regress; N per row (C1S >= c31000 - max(0.05, 2 SE)); Tr = the 18 decisive suite lines (v1.2's 11
per seed minus track_err back and yaw_progress turn), read per line from zero_kg_check's rows (never its verdict
field; drift 0, its_past_p2a 0; ``readout_fresh.suite_only``); E flags on worse_by: track_err back > 0.132, yaw_progress
turn > 0.0136; the diag label (worse_by <= 0.035 -> c1's predicted cost; beyond -> a student loss, beside sel.json's
tracking_watch). Routing: T LOST -> recommend no bake (governs every overlap, Left 20); T KEEPS and T-reg, N, Tr pass
with no E flag -> recommend yes with the Q3 split; otherwise -> the user decides.

CLASS HANDLING: T and x0.7 L090 rows are two launches decided pooled (no window); field_hard / L000 / T-reg values
that MISS their line by less than the window (0.04, 0.04, 0.07) relaunch once on BOTH students (``_R1``) and are
decided pooled; a pass is never relaunched; suite lines are never relaunched; more than 16 relaunch rows -> P2_READ
STOP (notify and ask); a same-row launch pair (L1 vs L2 / R1) that is neither BITWISE-EQUAL nor CLASS-FLIP by
classflip_check.py (same pin, single-wave rows, so its own verdict holds) -> P2_READ HALT.

REPORT ONLY: each launch's share and step-0 action fingerprint beside every T line (Left 23); slopes up / down;
the 0.14 m split on the ledges (|spawn pad level|; pits not split here); fall classes and falls by approximate hold;
back ratio and turn yaw rate (suite medians); hills / gravel lines; the Q3 split C1S - c31000 = (C1S - c1) +
(c1 - TDR) + (TDR - c31000) at es9-10; step 3's d' rows.

FAIL-CLOSED: no sel.json, a pick whose file sha differs, a pick equal to a fixed policy, sel.json changed (git) after
the first es9+ stamp of the read (a re-pick), any es9+ stamp in either root outside the planned names (a runner-up
read or a foreign row), a served stamp off the P2 pin -> refused (exit 2, nothing written).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import collections  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent                    # 2026-11-10_rough_terrain/p2
sys.path.insert(0, str(HERE))
import p2_step3 as P3  # noqa: E402  (sets RT_TP_OUT = rough_terrain_p2 before the machinery import; reused helpers)

M, Q, readout, S, rt_short = P3.M, P3.Q, P3.readout, P3.S, P3.rt_short
import readout_fresh as RF  # noqa: E402  (short_arm/, on sys.path via p2_step3; suite_only, committed_clean, relabel)

assert Path(RF.__file__).resolve().parent == P3.SHORT, RF.__file__
assert RF.readout is readout and M.OUT == P3.P2_OUT

# ============================================================================== pre-registered constants (P2.md)
CAMP, P2_OUT, RT_OUT = P3.CAMP, P3.P2_OUT, P3.RT_OUT
RESULTS_P2 = P3.RESULTS_P2
P2_PIN = P3.P2_PIN
SEL_JSON = RESULTS_P2 / "sel.json"
READ_JSON = RESULTS_P2 / "read.json"
NEAR_JSON = RESULTS_P2 / "read_near_miss.json"
PAIRS_JSON = RESULTS_P2 / "read_pairs.json"
STEP3_JSON = P3.STEP3_JSON
PREFIX = "P2R"
EMAP = {1: 9, 2: 10, 3: 11, 4: 12}                         # logical -> real env seed (one mapping point)
REAL_OK = (9, 10, 11, 12)
assert len(set(EMAP.values())) == len(EMAP)
C1_CK = M.W / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19/model_29500.pt"
C1_SHA = P3.C1_SHA
TDR_CK, TDR_SHA = M.TDR_CK, M.TDR_SHA
C31000_CK, C31000_SHA = P3.C31000_CK, P3.C31000_SHA
STUDENTS = ("C1S", "c31000")
TEACHERS = ("c1", "TDR")
T_KINDS = {"slopes": "rt_slopes_b2__omni", "ledge_down": "rt_ledge_down_b2__omni"}
TREG_KINDS = {"ledge_up": "rt_ledge_up_b2__omni", "pits": "rt_pits_b2__omni", "rubble": "rt_rubble_b3__omni"}
REPORT_KINDS = {"hills": "rt_hills_b3__omni", "gravel": "rt_gravel_b3__omni"}
N_ROWS = {"field_hard": ("R_field_hard", (1, 2, 3, 4), 1), "field_light_L000": ("R_field_light", (1, 2), 1),
          "x07_L090": ("R_x07_L090_field_light", (1, 2), 2)}
SUITE_RID, SUITE_LES = "R_suite", (1, 2)
T_LES, TREG_LES, REPORT_LES, Q3_LES = (1, 2, 3, 4), (1, 2), (1, 2), (1, 2)
FRONT = (("treg", "ledge_up"), ("treg", "pits"), ("treg", "rubble"), ("report", "hills"), ("report", "gravel"))
NEAR_WINDOW = {"field_hard": 0.04, "field_light_L000": 0.04, "treg": 0.07}
RELAUNCH_CAP = 16
E_BOUNDS = {"track_err back": 0.132, "yaw_progress turn": 0.0136}   # c1-vs-TDR es5-6 max worse_by + R (P2.md E)
TR_EXCLUDED = tuple(E_BOUNDS)                                        # those lines are E only (settled 6g)
TR_DECISIVE_STATS = tuple(s for s in RF.SUITE_STATS if s not in TR_EXCLUDED)
assert len(TR_DECISIVE_STATS) == 9 and set(TR_EXCLUDED) <= set(RF.SUITE_STATS)
DIAG_LABEL_BOUND = 0.035                                             # amendment 1 (audit pass 3)
SPLIT_M = S.ENVELOPE_M                                               # 0.14 m
N_HOLDS = 8                                                          # rows_rt's 8-hold 80 s schedule
STUDENT_ROW_TIMEOUT_S = P3.STUDENT_ROW_TIMEOUT_S
FROZEN_SHAS = {"scripts/readout.py": "14e8717c", "thresholds_rough_v1.json": "ce45e01b",
               "manifests/rows_rt.json": "0b038f03"}                 # P2.md Design "Reuse": byte-unchanged
BLOCKS = {"front": "p2read_front", "rows": "p2read_rows", "relaunch": "p2read_relaunch"}
EXPECT_SLOTS = {"p2read_front": 5, "p2read_rows": 84}
EXPECT_GROUPS = {"T": 32, "treg": 12, "N": 20, "Tr": 4, "report": 8, "Q3": 8}
NEAR_RC = {"NONE": 0, "DUE": 10, "STOP": 11, "INCOMPLETE": 12, "HALT": 13}
VERDICT_RC = {"COMPUTED": 0, "INCOMPLETE": 12, "HALT": 13, "PENDING": 14}
REFUSED_RC, HALT_RC = P3.REFUSED_RC, P3.HALT_RC
PLACEHOLDER_SHA = "d" * 64
MODE = {"dry": False}


class ReadRefused(Exception):
    """A fail-closed guard refused the read (exit 2, nothing written)."""


def say(line: str) -> None:
    print(line, flush=True)


def sha8(s) -> str:
    return str(s)[:8]


# ============================================================================== policies
def load_pick(strict: bool = True) -> dict:
    """results/p2/sel.json's pick (verdict SELECTED; sha recomputed from the checkpoint; not a fixed policy;
    committed and clean when ``strict``). ``ok`` False + ``problems`` otherwise (never raises)."""
    out = {"ok": False, "problems": [], "path": str(SEL_JSON)}
    doc = P3.read_json(SEL_JSON)
    if doc is None:
        out["problems"].append(f"no {SEL_JSON} (the step-6 selection writes it, and the read refuses without it)")
        return out
    pick = doc.get("pick") or {}
    sha, ck = pick.get("sha256"), pick.get("ckpt")
    out.update(sha=sha, ck=ck, iter=pick.get("iter"), cand=pick.get("cand"), teacher=pick.get("teacher"),
               verdict=doc.get("verdict"), sel_sha256=M.sha256_file(SEL_JSON))
    pr = out["problems"]
    if doc.get("verdict") != "SELECTED":
        pr.append(f"sel.json verdict {doc.get('verdict')!r} != 'SELECTED'")
    if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
        pr.append(f"sel.json pick.sha256 {sha!r} is not a sha256")
    elif not ck or not Path(ck).is_absolute() or not Path(ck).is_file():
        pr.append(f"sel.json pick.ckpt {ck!r} is not an existing absolute path")
    else:
        got = M.sha256_file(Path(ck))
        if got != sha:
            pr.append(f"pick checkpoint {ck} sha256 {got[:8]} != sel.json {sha[:8]}")
    if sha in (C31000_SHA, C1_SHA, TDR_SHA):
        pr.append(f"the pick {sha8(sha)} is a fixed policy (c31000 / c1 / TDR)")
    if ck and Path(str(ck)).name == "model_31000.pt":
        out["name_collision"] = "the pick's file is model_31000.pt: it collides with 'c31000' by name; named by sha"
    if strict:
        try:
            cc = RF.committed_clean(SEL_JSON)
        except ValueError as exc:                          # a path outside the checkout (repo_rel)
            cc = {"ok": False, "detail": f"not inside the checkout ({exc})"}
        out["committed"] = cc
        if not cc["ok"]:
            pr.append(f"sel.json: {cc['detail']} (commit it before any es9+ row)")
    out["ok"] = not pr
    return out


def policies(pick: dict | None) -> dict:
    """{key: {sha, ck, role, label, placeholder}}; a missing pick -> the DRY PLACEHOLDER (plans only)."""
    ph = not (pick and pick.get("ok"))
    pols = {
        "C1S": {"sha": PLACEHOLDER_SHA if ph else pick["sha"],
                "ck": "/DRY_PLACEHOLDER/pick_from_results_p2_sel.json.pt" if ph else str(pick["ck"]),
                "role": "student", "placeholder": ph},
        "c31000": {"sha": C31000_SHA, "ck": str(C31000_CK), "role": "student", "placeholder": False},
        "c1": {"sha": C1_SHA, "ck": str(C1_CK), "role": "teacher", "placeholder": False},
        "TDR": {"sha": TDR_SHA, "ck": str(TDR_CK), "role": "teacher", "placeholder": False},
    }
    for k, p in pols.items():
        p["label"] = f"{k} {sha8(p['sha'])}"
    s8 = [sha8(p["sha"]) for p in pols.values()]
    if len(set(s8)) != len(s8):
        raise ReadRefused(f"two policies share a sha8 {s8}: row names would collide")
    return pols


def fixed_policy_problems(check_iter: bool = True) -> list:
    bad = list(P3.c31000_problems(check_iter))
    for lab, ck, sha in (("c1", C1_CK, C1_SHA), ("TDR", TDR_CK, TDR_SHA)):
        if not Path(ck).is_file():
            bad.append(f"{lab}: no checkpoint {ck}")
        elif M.sha256_file(ck) != sha:
            bad.append(f"{lab}: {ck} sha256 != {sha8(sha)}")
    return bad


# ============================================================================== the row plan
def rows_by_id() -> dict:
    return P3.rows_by_id()


def unit(group: str, line: str, rid: str, les: int, pol: str, tag: str = "") -> dict:
    return {"group": group, "line": line, "rid": rid, "les": int(les), "es": int(EMAP[int(les)]), "pol": pol,
            "tag": tag}


def ukey(u: dict) -> tuple:
    return (u["rid"], u["les"], u["pol"], u["tag"])


def plan_units() -> list:
    """The 84 rows in launch order (see the module docstring)."""
    out = []
    front = []
    for grp, kind in FRONT:
        rid = (TREG_KINDS if grp == "treg" else REPORT_KINDS)[kind]
        front.append(unit(grp, kind, rid, 1, "C1S"))
    out += front
    fk = {ukey(u) for u in front}

    def add(u):
        if ukey(u) not in fk:
            out.append(u)

    for kind, rid in T_KINDS.items():
        for les in T_LES:
            for tag in ("", "_L2"):
                for pol in STUDENTS:
                    add(unit("T", kind, rid, les, pol, tag))
    for kind, rid in TREG_KINDS.items():
        for les in TREG_LES:
            for pol in STUDENTS:
                add(unit("treg", kind, rid, les, pol))
    for line, (rid, less, n_l) in N_ROWS.items():
        for les in less:
            for tag in ("", "_L2")[:n_l]:
                for pol in STUDENTS:
                    add(unit("N", line, rid, les, pol, tag))
    for les in SUITE_LES:
        for pol in STUDENTS:
            add(unit("Tr", "suite", SUITE_RID, les, pol))
    for kind, rid in REPORT_KINDS.items():
        for les in REPORT_LES:
            for pol in STUDENTS:
                add(unit("report", kind, rid, les, pol))
    for kind, rid in T_KINDS.items():
        for les in Q3_LES:
            for pol in TEACHERS:
                add(unit("Q3", kind, rid, les, pol))
    keys = [ukey(u) for u in out]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate row units")
    return out


def front_units() -> list:
    return plan_units()[:len(FRONT)]


def unit_name(u: dict, pols: dict) -> str:
    return M.row_name(PREFIX, pols[u["pol"]]["sha"], u["rid"]) + u["tag"]


def unit_row(u: dict, pols: dict) -> dict:
    r = dict(rows_by_id()[u["rid"]])
    if pols[u["pol"]]["role"] == "student":
        r["args"] = P3.student_args(r["args"])
    return r


def unit_label(u: dict) -> str:
    return f"{PREFIX}.{u['group']}.{u['line']}.{u['pol']}.es{u['es']}{u['tag']}"


def unit_step(u: dict, pols: dict) -> M.Step:
    p = pols[u["pol"]]
    row = unit_row(u, pols)
    name = unit_name(u, pols)
    st = M.gate_step(unit_label(u), name=name, ck=p["ck"], row=row, es=u["es"], tp_out=P2_OUT)
    if p["role"] == "student":
        st.env["TP_ROW_TIMEOUT"] = str(STUDENT_ROW_TIMEOUT_S)
    tp_out, nm, sc, es = st.slot
    st.note = ((("DRY PLACEHOLDER pick (no results/p2/sel.json); " if p["placeholder"] else "")
                + f"{u['group']} {u['line']} {p['label']} {p['role']} es{es}{u['tag'] or ''}; expected stamp "
                + f"{M.slot_dir(tp_out, nm, sc, es)}/<YYYY-MM-DD_HH-MM-SS>/"))
    return st


def all_planned_names(pols: dict) -> set:
    """Every name dir the read may write: the 84 rows' names and the _R1 twin of each decisive relaunchable row."""
    names = {unit_name(u, pols) for u in plan_units()}
    for u in plan_units():
        if u["group"] == "treg" or (u["group"] == "N" and u["line"] in NEAR_WINDOW):
            names.add(unit_name(dict(u, tag="_R1"), pols))
    return names


def _checks(block: str) -> list:
    pre = f"P2R_{block}"
    return [P3.build_check_step(pre),
            M._cpu(f"{pre}.inputs.check", HERE / "p2_read.py", "--inputs-check", "--block", block, stop=True,
                   dry=True, note="P2 pin, the pick (sel.json, by sha), c31000 / c1 / TDR by sha, no refused "
                                  "runner key, frozen files, no foreign es9+ stamp")]


def relaunch_units(pols: dict, near: dict | None = None) -> tuple:
    """``(units or None, how)`` from a FRESH near-miss record (DUE and its base stamps == the slots now). ``None`` in
    a dry pass without one (the conditional plan); a real run refuses a missing, stale or non-DUE record."""
    doc = near if near is not None else P3.read_json(NEAR_JSON)
    why = None
    if doc is None:
        why = f"no {NEAR_JSON} (run --near-miss after the rows)"
    elif doc.get("status") != "DUE":
        why = f"{NEAR_JSON} status {doc.get('status')!r} (only DUE plans relaunch rows)"
    else:
        now = current_base_stamps(pols, [tuple(k) for k in doc.get("base_units") or []])
        if now != doc.get("base_stamps"):
            why = f"{NEAR_JSON} is stale (its base stamps differ from the slots now)"
    if why:
        if is_dry():
            return None, f"CONDITIONAL in this dry pass: {why}; a real run plans only the rows the record marks due"
        raise ValueError(why)
    due = [unit(**{k: v for k, v in d.items() if k in ("group", "line", "rid", "les", "pol", "tag")})
           for d in doc["relaunch_due"]]
    if len(due) > RELAUNCH_CAP:
        raise ValueError(f"{len(due)} relaunch rows > cap {RELAUNCH_CAP}: P2_READ STOP (notify and ask)")
    return due, f"due by {NEAR_JSON.name} written {doc.get('at')}"


def plan_read(block: str, pols: dict | None = None) -> list:
    pols = pols or policies(load_pick(strict=False))
    if block == "p2read_front":
        units, how = front_units(), "front (Left 22)"
    elif block == "p2read_rows":
        units, how = plan_units(), "all 84 (the front 5 skip once complete)"
    elif block == "p2read_relaunch":
        units, how = relaunch_units(pols)
        if units is None:
            units = [dict(u, tag="_R1") for u in plan_units()
                     if u["group"] == "treg" or (u["group"] == "N" and u["line"] in NEAR_WINDOW)]
    else:
        raise ValueError(f"unknown block {block}")
    steps = _checks(block.replace("p2read_", ""))
    for u in units:
        steps.append(unit_step(u, pols))
    labels = [s.label for s in steps]
    if len(labels) != len(set(labels)):
        raise RuntimeError("duplicate step labels")
    slots = [s.slot for s in steps if s.slot]
    if len(slots) != len(set(slots)):
        raise RuntimeError("two rows share an output slot")
    if block in EXPECT_SLOTS and len(slots) != EXPECT_SLOTS[block]:
        raise RuntimeError(f"{block}: {len(slots)} rows, P2 pre-registers {EXPECT_SLOTS[block]}")
    if block == "p2read_relaunch" and not is_dry() and len(slots) > RELAUNCH_CAP:
        raise RuntimeError(f"{block}: {len(slots)} rows > cap {RELAUNCH_CAP}")
    bad = rt_short.slot_problems(steps)
    if bad:
        raise ValueError("slot collision: " + "; ".join(bad))
    return steps


def is_dry() -> bool:
    return MODE["dry"] or os.environ.get("P2_DRY") == "1"


PLAN_BLOCKS = tuple(BLOCKS.values())


def install(pols: dict) -> None:
    """Patch the machinery (idempotent): ONLY the read blocks, their plan dispatch, the two STOP files."""
    if getattr(M, "_p2read_installed", False):
        return
    M.BLOCKS = PLAN_BLOCKS
    M.plan = lambda block, *a, **k: plan_read(block, pols)
    M.STOP_FILE = P3._AnyStop(P3.P2_STOP, M.STOP_FILE)
    if os.environ.get("P2_DDC_GROWTH_MAX_GB"):
        M.DDC_GROWTH_MAX_GB = float(os.environ["P2_DDC_GROWTH_MAX_GB"])
    M._p2read_installed = True


# ============================================================================== --dry
def group_counts(units: list) -> dict:
    return dict(collections.Counter(u["group"] for u in units))


def reconcile(units: list) -> dict:
    """P2.md step 7: 84 = 36 base + 20 second launches (Q4) + 20 kind rows (Q2) + 8 teacher rows (Q3)."""
    second = [u for u in units if u["tag"] == "_L2"]
    kind = [u for u in units if u["group"] in ("treg", "report")]
    teacher = [u for u in units if u["group"] == "Q3"]
    base = [u for u in units if u not in second and u not in kind and u not in teacher]
    got = {"base": len(base), "second": len(second), "kind": len(kind), "teacher": len(teacher), "total": len(units)}
    want = {"base": 36, "second": 20, "kind": 20, "teacher": 8, "total": 84}
    return {"got": got, "want": want, "agree": got == want,
            "base_rows": "T single launch 16 + field_hard 8 + L000 4 + x0.7 L090 single launch 4 + suite 4",
            "second_rows": "T 16 + x0.7 L090 4", "kind_rows": "T-reg 12 + hills / gravel 8",
            "teacher_rows": "c1 + TDR on slopes b2 / ledge_down b2 at es9-10"}


def dry(block: str | None) -> int:
    pick = load_pick(strict=False)
    pols = policies(pick)
    if not pick["ok"]:
        say(f"P2_READ DRY pick: PLACEHOLDER ({'; '.join(pick['problems'])}); a real block refuses")
    else:
        say(f"P2_READ DRY pick: C1S {pick['sha']} {pick['ck']}")
    units = plan_units()
    gc = group_counts(units)
    rc = reconcile(units)
    blocks = [BLOCKS[block]] if block else list(PLAN_BLOCKS)
    n_by = {}
    for blk in blocks:
        try:
            steps = plan_read(blk, pols)
        except (ValueError, RuntimeError, OSError) as exc:
            say(f"# {blk}: plan refused: {exc}")
            continue
        rows = [s for s in steps if s.slot]
        n_by[blk] = len(rows)
        say(f"# {blk}: {len(rows)} rows -> {P2_OUT}/lr")
        if blk == BLOCKS["relaunch"]:
            due, how = relaunch_units(pols)
            say(f"# {blk}: {how}" + (f"; shown: the superset of every possible _R1 row (field_hard 8, L000 4, "
                                      f"T-reg 12); at most {RELAUNCH_CAP} run, more -> P2_READ STOP"
                                      if due is None else ""))
        p = P3.read_json(P2_PIN)
        for s in steps:
            env = " ".join(f"{k}={v}" for k, v in sorted(s.env.items()))
            if s.slot:
                done = M.complete_stamps(M.slot_dir(*s.slot), pin=p) if p else []
                say(f"#   {s.label:<44} {s.slot[1]:<40} {s.slot[2]:<26} es{s.slot[3]:<3} "
                    f"[{len(done)} complete at the pin{' (skipped)' if done else ''}]")
            say(f"      {s.how}: {env + ' ' if env else ''}{' '.join(map(str, s.argv))}")
    say(f"P2_READ DRY slots {' '.join(f'{k}={v}' for k, v in n_by.items())} groups "
        f"{' '.join(f'{k}={v}' for k, v in sorted(gc.items()))} reconcile={'AGREE' if rc['agree'] else 'DIFFER'} "
        f"{rc['got']} (P2.md: 36 base + 20 second + 20 kind + 8 teacher = 84) relaunch_cap={RELAUNCH_CAP}")
    return 0 if rc["agree"] and gc == EXPECT_GROUPS else 1


# ============================================================================== --inputs-check
def foreign_es9_stamps(pols: dict) -> dict:
    """es9-12 name dirs in either root that the read does not plan (directory names only) + the shas of every
    complete es9-12 stamp in them (run_meta only; nothing else is opened)."""
    names = all_planned_names(pols)
    ours = {p["sha"] for p in pols.values()}
    out = {"foreign_name_dirs": [], "foreign_student_shas": [], "refuse": []}
    for root in (RT_OUT / "lr", P2_OUT / "lr"):
        if not root.is_dir():
            continue
        for nd in sorted(p for p in root.iterdir() if p.is_dir()):
            hit = [ed for sd in nd.iterdir() if sd.is_dir() for ed in sd.iterdir()
                   if ed.is_dir() and re.fullmatch(r"seed0*(9|10|11|12)", ed.name)]
            if hit and nd.name not in names:
                out["foreign_name_dirs"].append(str(nd))
                for ed in hit:
                    for st in ed.iterdir():
                        if (st / "run_meta.json").is_file():
                            m = P3.read_json(st / "run_meta.json", {}) or {}
                            s = m.get("checkpoint_sha256")
                            if m.get("role") == "student":
                                out["foreign_student_shas"].append(f"{st}: {sha8(s)}")
                            if m.get("role") == "student" or s in ours:
                                out["refuse"].append(f"{st} ({m.get('role')} {sha8(s)})")
    out["rule"] = ("refused: an es9-12 stamp outside the planned names that is a student-role stamp (a runner-up or "
                   "another student read on es9+) or one of the read's own policies; other teacher stamps are recorded")
    return out


def frozen_problems() -> list:
    bad = []
    for rel, want in FROZEN_SHAS.items():
        got = M.sha256_file(CAMP / rel)
        if not got.startswith(want):
            bad.append(f"{rel} sha256 {got[:8]} != {want} (P2.md Design: byte-unchanged)")
    thr = readout.load_thresholds(readout.THRESHOLDS)
    for kind, rid in {**T_KINDS, **TREG_KINDS}.items():
        kd = thr["kinds"][kind]
        if kd["status"] != "gated" or S.row_id(kind, kd["gate_band"]) != rid:
            bad.append(f"{kind}: thresholds {kd['status']} {kd.get('gate_band')} != the read's {rid}")
    for kind, rid in REPORT_KINDS.items():
        kd = thr["kinds"][kind]
        if kd["status"] != "solved" or S.row_id(kind, kd["r_band"]) != rid:
            bad.append(f"{kind}: thresholds {kd['status']} {kd.get('r_band')} != the read's {rid}")
    return bad


def inputs_check(block: str) -> int:
    bad, warn = P3.p2_pin_problems()
    pick = load_pick(strict=True)
    bad += [f"pick: {x}" for x in pick["problems"]]
    bad += fixed_policy_problems(check_iter=True)
    if pick["ok"]:
        it = rt_short.iter_field(Path(pick["ck"]))
        if pick.get("iter") is not None and it != pick.get("iter"):
            bad.append(f"pick iter field {it!r} != sel.json iter {pick.get('iter')!r}")
    pols = policies(pick if pick["ok"] else None)
    refused = set(P3.refused_keys())
    envs = set()
    for u in plan_units():
        r = rows_by_id()[u["rid"]]
        envs.add(str(S.OMNI / r["env_file"]))
        if r.get("level_file"):
            envs.add(str(S.OMNI / r["level_file"]))
    for ef in sorted(envs):
        hit = sorted(set(M.krabby_of(ef)) & refused)
        if hit:
            bad.append(f"{ef} carries student-refused runner keys {hit} (no TP_STRIP_KEYS here)")
    for rid in REPORT_KINDS.values():
        r = rows_by_id()[rid]
        if r.get("rt_friction") and S.expected_env(r).get(S.FRICTION_VAR) != r["rt_friction"]:
            bad.append(f"{rid}: the composed env does not carry {S.FRICTION_VAR}={r['rt_friction']}")
    bad += frozen_problems()
    fe = foreign_es9_stamps(pols)
    if fe["refuse"]:
        bad.append(f"es9-12 stamps outside the planned names: {fe['refuse'][:6]} (a runner-up or foreign read)")
    elif fe["foreign_name_dirs"]:
        warn.append(f"es9-12 teacher stamps of other policies outside the planned names (recorded, not read): "
                    f"{fe['foreign_name_dirs'][:4]}")
    if block == "relaunch":
        try:
            due, how = relaunch_units(pols)
            say(f"P2_INPUTS INFO relaunch {len(due or [])} rows ({how})")
        except ValueError as exc:
            bad.append(f"relaunch: {exc}")
    for w in warn:
        say(f"P2_INPUTS WARN {w}")
    say(f"P2_INPUTS {'OK' if not bad else 'BAD'} block=read_{block} pick={sha8(pick.get('sha'))}"
        + (f" -- {'; '.join(bad)}" if bad else ""))
    return 0 if not bad else 1


# ============================================================================== the read subclass
class SeedError(Exception):
    """A seed lookup outside the map, or a served stamp at the wrong env seed / sha."""


class P2ReadRows(readout.Rows):
    """``readout.Rows`` over BOTH output roots, restricted to the real seed dirs before any run_meta is opened, whose
    ``found()`` maps a LOGICAL env seed through ``EMAP``, role-swaps a student sha's row args and, given ``name``,
    serves only that launch's name dir. Nothing else differs from readout (find, tables, gates, zero_kg)."""

    def __init__(self, roots, shas, student_shas, emap=EMAP, real_ok=REAL_OK, dup: str = "earliest"):  # noqa: D107
        if len(set(emap.values())) != len(emap):
            raise SeedError(f"seed map {emap} is not injective")
        self.doc = S.load_rows()
        self.kind = {r["id"]: r for r in self.doc["kind_rows"]}
        self.ret = {r["id"]: r for r in self.doc["retention_rows"]}
        self.all = {**self.kind, **self.ret}
        self.dup = dup
        self.roots = [Path(r) for r in roots]
        self.EMAP, self.REAL_OK = dict(emap), tuple(real_ok)
        self.student = set(student_shas)
        self.idx = P3.scan_seeds(self.roots, {r["scenario"] for r in self.all.values()}, self.REAL_OK, set(shas))
        self.served = []

    def found(self, rid: str, es: int, sha: str, name: str | None = None) -> dict:  # noqa: C901
        if isinstance(es, bool) or int(es) != es:
            raise SeedError(f"{rid}: env seed {es!r} is not an integer")
        les = int(es)
        if les not in self.EMAP:
            raise SeedError(f"{rid} {sha8(sha)}: logical es{les} outside the map {sorted(self.EMAP)}")
        real = int(self.EMAP[les])
        if real not in self.REAL_OK:
            raise SeedError(f"{rid}: logical es{les} -> es{real} outside {self.REAL_OK}")
        row = self.all[rid]
        if sha in self.student:
            row = dict(row, args=P3.student_args(row["args"]))
        key = (sha, row["scenario"], real)
        idx = self.idx
        if name is not None:
            idx = {key: [c for c in self.idx.get(key, []) if Path(c["stamp"]).parents[2].name == name]}
        f = readout.find(idx, row, real, sha, self.dup)
        if int(f["es"]) != real:
            raise SeedError(f"{rid}: readout.find answered es{f['es']} for es{real}")
        rec = {"rid": rid, "logical_es": les, "real_es": real, "sha256": sha, "name": name, "status": f["status"],
               "stamp": f.get("stamp"), "name_dir": None, "code_sha": f.get("code_sha"), "eval_sha": f.get("eval_sha"),
               "duplicates": f.get("duplicates") or []}
        if f.get("stamp"):
            st = Path(f["stamp"])
            meta = json.loads((st / "run_meta.json").read_text())
            tp = json.loads((st / "tp_row.json").read_text())
            rm_seed = (meta.get("env_overrides") or {}).get("seed")
            rec["name_dir"] = st.parents[2].name
            if rm_seed is None or int(rm_seed) != real:
                raise SeedError(f"{rid} {st}: run_meta env_overrides.seed {rm_seed!r} != es{real}")
            if tp.get("env_seed") is None or int(tp["env_seed"]) != real:
                raise SeedError(f"{rid} {st}: tp_row env_seed {tp.get('env_seed')!r} != es{real}")
            if meta.get("checkpoint_sha256") != sha:
                raise SeedError(f"{rid} {st}: checkpoint_sha256 {sha8(meta.get('checkpoint_sha256'))} != {sha8(sha)}")
        self.served.append(rec)
        return f


def read_rows(pols: dict) -> P2ReadRows:
    return P2ReadRows([RT_OUT / "lr", P2_OUT / "lr"], {p["sha"] for p in pols.values()},
                      {pols[k]["sha"] for k in STUDENTS})


# ============================================================================== pooled statistics (P2.md, exactly)
def pooled_values(tabs_by_es: dict, key: str) -> dict:
    """``{(es, env_index): mean over that key's launches of float(bool(outcome))}`` from ``{es: [table, ...]}``.
    Keys are never (launch, env_index): n is the number of (es, env_index) keys whatever the launch count."""
    out = {}
    for es, tabs in sorted(tabs_by_es.items()):
        if not tabs:
            raise readout.KeySetError(f"es{es}: no launch")
        ids = set(tabs[0][key])
        for t in tabs[1:]:
            if set(t[key]) != ids:
                raise readout.KeySetError(f"es{es}: launches differ in env_index sets")
        for i in sorted(ids):
            vals = [float(bool(t[key][i])) for t in tabs]
            out[(int(es), int(i))] = sum(vals) / len(vals)
    return out


def _share(d: dict) -> float:
    return float(sum(d.values()) / len(d))


def pooled_g(a: dict, b: dict, rule: dict, what: str = "") -> dict:
    """``readout.g_line``'s clauses on pooled (fractional) dicts; the same keys as g_line's output."""
    readout.assert_keys(what or "pooled", a, b)
    va, vb = _share(a), _share(b)
    se = readout.th().paired_completion_se(a, b)
    d = va - vb
    ci = [d - readout.Z95 * se, d + readout.Z95 * se]
    le = abs(d) <= readout.NOISE_EFFECT + readout.EPS
    need = max(rule["gain_floor"], rule["gain_se_mult"] * se)
    reg_m = max(rule["regress_floor"], rule["regress_se_mult"] * se)
    gain = bool(d >= need - readout.EPS and ci[0] >= rule["ci_lb"] - readout.EPS and not le)
    regress = bool(d < -reg_m - readout.EPS and not le)
    return {"new": va, "tdr": vb, "diff": d, "se": se, "ci95": ci, "n": len(a), "gain_need": need, "ci_lb": ci[0],
            "gain": gain, "regress_margin": reg_m, "regress": regress, "le_noise": le}


def pooled_r(head: dict, ref: dict, cfg: dict, row: str = "") -> dict:
    """``readout.retention_line``'s line (select_student.completion_margin) on pooled dicts; its output keys."""
    readout.assert_keys(row or "pooled", head, ref)
    pa, pb = _share(head), _share(ref)
    se = readout.th().paired_completion_se(head, ref)
    m = readout.ss().completion_margin(se, cfg)
    out = {"row": row, "se_diff": se, **m, "line": pb - m["margin"], "diff": pa - pb, "head": pa, "ref": pb,
           "n": len(head)}
    out["noninferior"] = bool(pa >= pb - m["margin"] - 1e-12)
    out["beats_by_m"] = bool(out["diff"] >= m["margin"] - 1e-12)
    out["pass"] = out["noninferior"]
    out["le_noise"] = abs(out["diff"]) <= readout.NOISE_EFFECT + readout.EPS
    return out


def treg_shortfall(g: dict) -> float:
    """How far a no-regress line misses: (-regress_margin) - diff (> 0 on a fail)."""
    return -g["regress_margin"] - g["diff"]


def near_due(kind: str, line: dict) -> tuple:
    """(fails, shortfall, near): a value that MISSES its line by less than the window; a pass is never near."""
    if kind == "treg":
        fails, short = bool(line["regress"]), treg_shortfall(line)
    else:
        fails, short = (not line["pass"]), line["line"] - line["head"]
    return fails, short, bool(fails and short < NEAR_WINDOW[kind])


# ============================================================================== stamps of a line
def step0_fp(st: Path) -> str:
    a = P3.step0_actions(Path(st))
    h = hashlib.sha256()
    for e in sorted(a):
        h.update(int(e).to_bytes(4, "little"))
        h.update(a[e])
    return h.hexdigest()[:16]


def stamp_problems(st: Path, pin: dict) -> list:
    return M.build_mismatch(M.stamp_build(st), pin) + P3.raw_problems(st)


def side_tables(R: P2ReadRows, rid: str, names_by_les: dict, sha: str, pin: dict, tables: bool = True) -> tuple:
    """({real es: [table per launch]}, [launch records], [problems]) for one side of a line (``tables`` False:
    the stamps only, nothing per episode is read)."""
    radius = R.all[rid].get("cross_r")
    tabs, recs, bad = {}, [], []
    for les, names in sorted(names_by_les.items()):
        for nm in names:
            f = R.found(rid, les, sha, nm)
            rec = {"rid": rid, "es": R.EMAP[les], "logical_es": les,
                   "name": nm, "sha256": sha, "status": f["status"], "stamp": f.get("stamp"),
                   "problems": list(f["problems"][:3]), "duplicates": f.get("duplicates") or []}
            if f["status"] == "OK":
                st = Path(f["stamp"])
                pr = stamp_problems(st, pin)
                if pr:
                    rec["problems"] += pr
                    rec["status"] = "BAD"
                elif tables:
                    t = readout.episode_table(st, radius)
                    tabs.setdefault(R.EMAP[les], []).append(t)
                    rec["table_index"] = len(tabs[R.EMAP[les]]) - 1
            if rec["status"] != "OK":
                bad.append(f"{rid} es{R.EMAP[les]} {sha8(sha)} {nm}: {rec['status']} {rec['problems'][:2]}")
            recs.append(rec)
    return tabs, recs, bad


def pairing_problems(rid: str, recs: list, kind_row: bool) -> list:
    """Every stamp of a line (both sides, every launch) at one es: equal env_index sets, 100 episodes on a kind row,
    spawns within readout.SPAWN_PAIR_TOL_M of the first, one build (readout.pair_tables' checks, over launches)."""
    import numpy as np  # noqa: PLC0415

    bad = []
    by_es = collections.defaultdict(list)
    for r in recs:
        if r["status"] == "OK":
            by_es[r["es"]].append(r)
    for es, rs in sorted(by_es.items()):
        sp = [readout.spawns_of(Path(r["stamp"])) for r in rs]
        ids = set(sp[0])
        for r, s in zip(rs, sp):
            if set(s) != ids:
                bad.append(f"{rid} es{es}: env_index sets differ ({r['stamp']})")
                continue
            if kind_row and len(s) != readout.KIND_ROW_EPISODES:
                bad.append(f"{rid} es{es} {r['stamp']}: {len(s)} episodes (want {readout.KIND_ROW_EPISODES})")
            far = [i for i in s if np.linalg.norm(np.asarray(s[i]) - np.asarray(sp[0][i])) > readout.SPAWN_PAIR_TOL_M]
            if far:
                bad.append(f"{rid} es{es} {r['stamp']}: {len(far)} envs spawn apart from {rs[0]['stamp']}")
    builds = {tuple((M.stamp_build(r["stamp"]) or {}).get(k) for k in ("code_sha", "eval_sha"))
              for r in recs if r["status"] == "OK"}
    if len(builds) > 1:
        bad.append(f"{rid}: stamps at {len(builds)} builds")
    return bad


def launch_report(recs: list, tabs: dict, key: str) -> list:
    """Each launch's own share of ``key`` (and its step-0 action fingerprint) -- report only (Left 23)."""
    out = []
    for r in recs:
        if r["status"] != "OK" or "table_index" not in r:
            out.append({k: r.get(k) for k in ("es", "name", "status", "stamp")})
            continue
        t = tabs[r["es"]][r["table_index"]]
        out.append({"es": r["es"], "name": r["name"], "stamp": r["stamp"], "n": len(t[key]),
                    "share": readout.share(t[key]), "step0_fp": step0_fp(Path(r["stamp"]))})
    return out


def launch_index_shares(recs: list, tabs: dict, key: str) -> dict:
    """The share pooled over es per launch tag ('' / _L2 / _R1): one number per launch round (report only)."""
    by = collections.defaultdict(list)
    for r in recs:
        if r["status"] == "OK" and "table_index" in r:
            tag = re.search(r"(_L2|_R1)?$", r["name"] or "").group(0) or "L1"
            by[tag].extend(bool(v) for v in tabs[r["es"]][r["table_index"]][key].values())
    return {k: {"n": len(v), "share": (sum(v) / len(v) if v else None)} for k, v in sorted(by.items())}


def variant_split(tabs_new: dict, tabs_ref: dict, kind: str) -> dict:
    """Report only: slopes up (sunk pad) / down (raised pad); ledges below / at-or-above 0.14 m (|spawn pad level|,
    the first launch's table; the spawn is shared by every launch of the key). Pooled success per side."""
    def classify(pl):
        if pl is None:
            return None
        if kind == "slopes":
            return ("up" if pl < -readout.PAD_LEVEL_FLAT_M else "down" if pl > readout.PAD_LEVEL_FLAT_M else "flat")
        return "below_0.14" if abs(pl) < SPLIT_M else "at_or_above_0.14"

    cls = {}
    for es, tabs in sorted(tabs_new.items()):
        for i, pl in tabs[0]["pad_level"].items():
            cls[(es, i)] = classify(pl)
    a, b = pooled_values(tabs_new, "success"), pooled_values(tabs_ref, "success")
    out = {}
    for c in sorted({v for v in cls.values() if v is not None}):
        ks = [k for k, v in cls.items() if v == c]
        out[c] = {"n": len(ks), "new": sum(a[k] for k in ks) / len(ks), "ref": sum(b[k] for k in ks) / len(ks)}
    out["rule"] = ("slopes: sign of the spawn pad level (|z| > 0.01 m); ledges: |spawn pad level| vs 0.14 m (pits are "
                   "not split here: their depth needs the floor raycast estimate)") if kind else ""
    return out


def falls_by_hold(st: Path) -> dict:
    """Report only: fall episodes by approximate hold (n_steps / the stamp's complete-episode n_steps x 8)."""
    eps = readout.episode_list(Path(st))
    full = max((int(e["n_steps"]) for e in eps if e["termination_reason"] == "schedule_complete"), default=None)
    out = collections.Counter()
    for e in eps:
        if e["termination_reason"] == "schedule_complete":
            continue
        if not full:
            out["unknown"] += 1
            continue
        out[f"hold{min(N_HOLDS, 1 + int(N_HOLDS * int(e['n_steps']) / full))}"] += 1
    agg = readout.aggregate_of(Path(st))
    return {"by_hold_approx": dict(sorted(out.items())), "fall_classes": agg.get("fall_classes"),
            "termination_reasons": agg.get("termination_reasons")}


def drop_raw_cache() -> None:
    for k in [k for k in readout._CACHE if k and k[0] == "raw"]:
        readout._CACHE.pop(k, None)


# ============================================================================== class-pair check (HALT)
class PairCache:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.doc = (P3.read_json(self.path) if self.path else None) or {"schema": "p2_read_pairs_v1", "pairs": {}}
        self.dirty = False

    def verdict(self, x: Path, y: Path) -> dict:
        k = f"{x}|{y}"
        if k not in self.doc["pairs"]:
            cf = P3.run_json(P3.CLASSFLIP, x, y, "--period", P3.PERIOD)
            self.doc["pairs"][k] = {kk: cf.get(kk) for kk in (
                "verdict", "cfg_identical", "cfg_differs", "dose_identical", "same_code", "structural_mismatch",
                "differing_episodes", "earliest_step", "earliest_keys", "step0_state_identical",
                "earliest_action_only_at_depth_frame")}
            self.dirty = True
        return self.doc["pairs"][k]

    def save(self) -> None:
        if self.path and self.dirty:
            M.write_json(self.path, self.doc)


def pair_check(recs: list, cache: PairCache) -> list:
    """Every same-row pair (one es, one policy): the first launch vs each later one. ok = BITWISE-EQUAL or
    CLASS-FLIP (classflip_check.py; same pin, single-wave rows: its own verdict holds); else HALT."""
    out = []
    by = collections.defaultdict(list)
    for r in recs:
        if r["status"] == "OK":
            by[(r["rid"], r["es"], r["sha256"])].append(r)
    for (rid, es, sha), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: (r["name"] or "").endswith(("_L2", "_R1")))
        for r in rs[1:]:
            meta = P3.read_json(Path(r["stamp"]) / "run_meta.json", {}) or {}
            if meta.get("suite"):
                raise ReadRefused(f"{rid}: a suite row has two launches (suite lines are never relaunched)")
            v = cache.verdict(Path(rs[0]["stamp"]), Path(r["stamp"]))
            a0, a1 = P3.step0_actions(Path(rs[0]["stamp"])), P3.step0_actions(Path(r["stamp"]))
            out.append({"rid": rid, "es": es, "sha256": sha, "x": rs[0]["stamp"], "y": r["stamp"],
                        "verdict": v["verdict"], "ok": v["verdict"] in ("BITWISE-EQUAL", "CLASS-FLIP"),
                        "step0_envs_differ": sorted(e for e in a0 if a0[e] != a1.get(e))[:40],
                        "n_step0_envs_differ": sum(1 for e in a0 if a0[e] != a1.get(e)),
                        "earliest_step": v.get("earliest_step"), "earliest_keys": v.get("earliest_keys")})
    return out


# ============================================================================== lines
def names_of(u_base: dict, pols: dict, tags) -> list:
    return [unit_name(dict(u_base, tag=t), pols) for t in tags]


def r1_missing(pols: dict, pin: dict, rid: str, less) -> list:
    """The _R1 rows of a due line (both students, every es) with no complete stamp at the pin."""
    sc = rows_by_id()[rid]["scenario"]
    out = []
    for les in less:
        for pol in STUDENTS:
            nm = unit_name(unit("x", "x", rid, les, pol, "_R1"), pols)
            if not M.complete_stamps(M.slot_dir(P2_OUT, nm, sc, EMAP[les]), pin=pin):
                out.append(f"{nm} es{EMAP[les]}")
    return out


def decisive_line(R, pols, pin, rid, less, key, double: bool, rel: bool, stop: bool) -> tuple:
    """(line, pending why or None): a due line pools its _R1 launches; a due line whose _R1 rows are not all
    complete (not run yet, or over the cap) is PENDING and carries its base-only numbers as ``provisional``."""
    if rel:
        miss = r1_missing(pols, pin, rid, less)
        if miss:
            ln = line(R, pols, pin, rid, less, "C1S", "c31000", key, tags_for(less, double),
                      tags_for(less, double))
            why = (f"relaunch due, {len(miss)} _R1 rows not complete"
                   + (" (over the cap: STOP, notify and ask)" if stop else "") + f": {miss[:4]}")
            return ln, why
    t = tags_for(less, double, rel)
    return line(R, pols, pin, rid, less, "C1S", "c31000", key, t, t), None


def line(R, pols, pin, rid, less, new_pol, ref_pol, key, tags_new, tags_ref) -> dict:
    """The stamps + pooled dicts of one line (``tags_*``: {les: [tags]}); status OK / INCOMPLETE / INVALID."""
    def names(pol, tags):
        return {les: [unit_name(unit("x", "x", rid, les, pol, t), pols) for t in tags[les]] for les in less}

    tn, rn, bn = side_tables(R, rid, names(new_pol, tags_new), pols[new_pol]["sha"], pin)
    tr, rr, br = side_tables(R, rid, names(ref_pol, tags_ref), pols[ref_pol]["sha"], pin)
    out = {"row": rid, "env_seeds": [EMAP[x] for x in less], "new": pols[new_pol]["label"],
           "ref": pols[ref_pol]["label"], "stamps": {"new": rn, "ref": rr}}
    if bn or br:
        out.update(status="INCOMPLETE", why=(bn + br)[:8])
        return out
    pp = pairing_problems(rid, rn + rr, rid in R.kind)
    if pp:
        out.update(status="INVALID", why=pp[:8])
        return out
    out.update(status="OK", a=pooled_values(tn, key), b=pooled_values(tr, key), tables={"new": tn, "ref": tr},
               launches={"new": launch_report(rn, tn, key), "ref": launch_report(rr, tr, key)},
               launch_rounds={"new": launch_index_shares(rn, tn, key), "ref": launch_index_shares(rr, tr, key)},
               n_launches={"new": {EMAP[x]: len(tags_new[x]) for x in less},
                           "ref": {EMAP[x]: len(tags_ref[x]) for x in less}})
    return out


def public(ln: dict) -> dict:
    return {k: v for k, v in ln.items() if k not in ("a", "b", "tables")}


def tags_for(less, double: bool, relaunched: bool = False) -> dict:
    t = [""] + (["_L2"] if double else []) + (["_R1"] if relaunched else [])
    return {les: list(t) for les in less}


def suite_tr(R, pols, pin) -> dict:
    """Tr + E from readout_fresh.suite_only (zero_kg_check rows per line; drift 0; never its verdict field)."""
    if readout.ITS_PAST_R != 0:
        raise ReadRefused(f"readout.ITS_PAST_R {readout.ITS_PAST_R} != 0")
    so = RF.suite_only(R, pols["C1S"]["sha"], pols["c31000"]["sha"], SUITE_LES)
    return tr_from_suite(so, pin, served_names=None)


def tr_from_suite(so: dict, pin: dict | None, served_names: dict | None = None) -> dict:
    out = {"status": so["status"], "suite_only_status": so["status"], "env_seeds_logical": list(SUITE_LES),
           "env_seeds_real": [EMAP[x] for x in SUITE_LES], "its_past_p2a": so.get("its_past_p2a"),
           "translate_statistic": so.get("translate_statistic"), "thresholds_version": so.get("thresholds_version"),
           "stamps": so.get("stamps")}
    if so["status"] in ("INCOMPLETE", "INVALID") and not so.get("lines"):
        out["why"] = so.get("why") or so.get("problems")
        return out
    if so.get("problems"):
        out.update(status="INVALID", why=so["problems"])
        return out
    pr = []
    if pin is not None:
        for side in ("new", "tdr"):
            for f in (so.get("stamps") or {}).get(side) or []:
                if f.get("stamp"):
                    pr += [f"suite {side} es{f['es']}: {x}" for x in stamp_problems(Path(f["stamp"]), pin)]
    if pr:
        out.update(status="INVALID", why=pr)
        return out
    lines = so["lines"]
    dec = [x for x in lines if x["statistic"] in TR_DECISIVE_STATS]
    eln = [x for x in lines if x["statistic"] in TR_EXCLUDED]
    want = {(f"suite es{es}", st) for es in SUITE_LES for st in TR_DECISIVE_STATS}
    got = {(x["row"], x["statistic"]) for x in dec}
    if got != want or len(dec) != 18:
        out.update(status="INVALID", why=[f"decisive lines {len(dec)} != 18: {sorted(got ^ want)[:6]}"])
        return out
    fails = [x for x in dec if x.get("fails")]
    flags = []
    for x in eln:
        b = E_BOUNDS[x["statistic"]]
        x = dict(x, e_bound=b, e_flag=bool(x.get("worse_by") is not None and x["worse_by"] > b))
        flags.append(x)
    out.update(status="FAIL" if fails else "PASS", n_lines=len(dec), lines=dec,
               fails=[f"{x['statistic']} {x['row']} (real es{EMAP[int(x['row'][8:])]})" for x in fails],
               E={"lines": flags, "flags": [f"{x['statistic']} {x['row']} worse_by {x['worse_by']:.4f} > "
                                            f"{x['e_bound']}" for x in flags if x["e_flag"]],
                  "rule": "flag when worse_by exceeds c1-vs-TDR's es5-6 maximum + R: back 0.0702 + 0.0617 = 0.132, "
                          "yaw 0.0070 + 0.0066 = 0.0136 (P2.md E; report, never a gate)"},
               rule="the 18 decisive suite lines (v1.2's 11 per seed minus track_err back and yaw_progress turn), "
                    "each at its v1.2 margin max(R, 3 SE), drift 0 (its_past_p2a 0); zero_kg_check rows read per line, "
                    "never its verdict field")
    return out


def diag_label(tr: dict, sel_doc: dict | None) -> dict | None:
    """P2.md Odds + amendment 1: a diag-only Tr fail is c1's predicted cost when every failing line is track_err
    diag with worse_by <= 0.035; beyond that (or a survival diag fail) it is presented as a student loss. Both labels carry the
    selection's tracking WATCH diag rows (selection_tracking_watch; MISSING is explicit). A label only, never a gate."""
    if tr.get("status") != "FAIL":
        return None
    fl = [x for x in tr["lines"] if x.get("fails")]
    if not all(x.get("category") == "diag" for x in fl):
        return {"applies": False, "why": "the Tr fail is not diag-only"}
    wb = max(float(x.get("worse_by") or 0.0) for x in fl)
    if all(x["statistic"] == "track_err diag" for x in fl) and wb <= DIAG_LABEL_BOUND + readout.EPS:
        lab = "predicted cost of c1 (P2.md Odds: c1's diag hold 0.973-0.976 vs c31000's 0.998-1.002)"
    else:
        lab = "student loss (beyond the predicted c1 cost)"
    return {"applies": True, "label": lab, "max_worse_by": wb, "bound": DIAG_LABEL_BOUND,
            "failing": [f"{x['statistic']} {x['row']}" for x in fl],
            "selection_tracking_watch": selection_tracking_watch(sel_doc)}


SEL_TW_KEYS = ("id", "statistic", "row", "student", "tstar", "worse_by", "se_diff", "pass", "missing")


def selection_tracking_watch(sel_doc: dict | None):
    """The selection's tracking WATCH diag rows for the pick (report only, shown beside the diag label).

    sel.json (rsd_lib.full_block_read via p2_select) stores it at ``blocks.<pick.cand>.watch.tracking.rows``
    (select_student.tracking_watch: ratio and |ratio-1| per env seed; ``tstar`` is the selection's reference
    teacher). There is no ``tracking_watch`` key. Absent rows read loudly as ``{status: MISSING, why}``, never []."""
    sd = sel_doc if isinstance(sel_doc, dict) else {}
    pk = sd.get("pick") or {}
    cand = pk.get("cand")
    blk = (sd.get("blocks") or {}).get(cand) if cand is not None else None
    tw = ((blk or {}).get("watch") or {}).get("tracking") or {}
    rows = [{k: r.get(k) for k in SEL_TW_KEYS} for r in tw.get("rows") or [] if r.get("category") == "diag"]
    if rows:
        return rows
    why = ("no sel.json" if not sd else "sel.json has no pick.cand" if cand is None
           else f"sel.json has no blocks.{cand}" if blk is None
           else f"sel.json blocks.{cand}.watch.tracking has no diag row")
    return {"status": "MISSING", "why": why}


def suite_report(tr: dict) -> dict:
    """Report only: back hold ratio and turn yaw rate (suite medians) per side and es."""
    out = {}
    for side in ("new", "tdr"):
        for f in (tr.get("stamps") or {}).get(side) or []:
            if not f.get("stamp"):
                continue
            bc = (json.loads((Path(f["stamp"]) / "scenario_metrics.json").read_text())["aggregate"]["suite"]
                  ["by_category"])
            out.setdefault("C1S" if side == "new" else "c31000", {})[f"es{f['es']}"] = {
                "back_ratio_median": (bc.get("back") or {}).get("ratio", {}).get("median"),
                "turn_yaw_progress_median": (bc.get("turn") or {}).get("yaw_progress", {}).get("median")}
    return out


# ============================================================================== near-miss / verdict
def base_line_specs() -> list:
    """The near-miss lines: (kind, line, rid, less) -- field_hard / L000 (window 0.04), T-reg kinds (0.07)."""
    out = [("field_hard", "field_hard", N_ROWS["field_hard"][0], N_ROWS["field_hard"][1]),
           ("field_light_L000", "field_light_L000", N_ROWS["field_light_L000"][0], N_ROWS["field_light_L000"][1])]
    out += [("treg", k, rid, TREG_LES) for k, rid in TREG_KINDS.items()]
    return out


def current_base_stamps(pols: dict, base_units) -> dict:
    """{name|scenario|es: [complete stamps at the pin]} of the given (rid, les, pol, tag) units (staleness key)."""
    p = P3.read_json(P2_PIN)
    out = {}
    for rid, les, pol, tag in base_units:
        u = unit("x", "x", rid, les, pol, tag)
        sc = rows_by_id()[rid]["scenario"]
        nm = unit_name(u, pols)
        out[f"{nm}|{sc}|{u['es']}"] = [str(s) for s in M.complete_stamps(M.slot_dir(P2_OUT, nm, sc, u["es"]), pin=p)]
    return out


def _thr():
    thr = readout.load_thresholds(readout.THRESHOLDS)
    return thr, readout._frozen_rule(thr, "G", readout.G_RULE), readout._frozen_rule(thr, "R_completion",
                                                                                      readout.R_COMPLETION_CFG)


def guards(strict: bool = True) -> tuple:
    pick = load_pick(strict=strict)
    if not pick["ok"]:
        raise ReadRefused("pick: " + "; ".join(pick["problems"]))
    pin = P3.read_json(P2_PIN)
    if not pin or pin.get("schema") != "rt_pin_v1":
        raise ReadRefused(f"no rt_pin_v1 {P2_PIN}")
    fb = fixed_policy_problems(check_iter=False) + frozen_problems()
    if fb:
        raise ReadRefused("; ".join(fb))
    pols = policies(pick)
    fe = foreign_es9_stamps(pols)
    if fe["refuse"]:
        raise ReadRefused(f"es9-12 stamps outside the planned names: {fe['refuse'][:6]} (a runner-up or foreign read)")
    return pick, pin, pols


def repick_check(R: P2ReadRows) -> dict:
    """sel.json's last commit must predate the first es9+ stamp the read serves (a re-pick after an es9+ number)."""
    rel = RF.repo_rel(SEL_JSON)
    r = RF.git("log", "-1", "--format=%ct", "--", rel)
    try:
        ct = int(r.stdout.strip())
    except ValueError:
        return {"ok": False, "why": f"git log on {rel}: {r.stdout.strip()[:80]} {r.stderr.strip()[:120]}"}
    first = min((os.path.getmtime(Path(s["stamp"]) / "run_meta.json") for s in R.served if s.get("stamp")),
                default=None)
    ok = first is None or ct < first
    return {"ok": ok, "sel_commit_time": ct, "first_es9_stamp_mtime": first,
            "why": None if ok else "sel.json was committed AFTER the first es9+ stamp (a re-pick): refused"}


def near_miss_doc(R, pols, pin, g_rule, r_cfg, cache: PairCache, with_pairs: bool = True) -> dict:
    lines, due, problems, all_recs = {}, [], [], []
    base_units = []
    for kind, ln_name, rid, less in base_line_specs():
        for pol in STUDENTS:
            for les in less:
                base_units.append((rid, les, pol, ""))
        key = "success" if kind == "treg" else "completion"
        ln = line(R, pols, pin, rid, less, "C1S", "c31000", key, tags_for(less, False), tags_for(less, False))
        rec = public(ln)
        if ln["status"] == "OK":
            stat = pooled_g(ln["a"], ln["b"], g_rule, rid) if kind == "treg" else pooled_r(ln["a"], ln["b"], r_cfg,
                                                                                           rid)
            fails, short, near = near_due(kind, stat)
            rec.update(stat=stat, fails=fails, shortfall=short, window=NEAR_WINDOW[kind], near_miss=near)
            if near:
                for les in less:
                    for pol in STUDENTS:
                        due.append(unit("treg" if kind == "treg" else "N", ln_name, rid, les, pol, "_R1"))
        else:
            problems += [f"{ln_name}: {ln['status']} {x}" for x in (ln.get("why") or [])[:3]]
        lines[ln_name] = rec
        all_recs += ln["stamps"]["new"] + ln["stamps"]["ref"]
        drop_raw_cache()
    pairs, pair_recs = [], []
    if with_pairs:
        dbl = [(rid, T_LES) for rid in T_KINDS.values()] + [(N_ROWS["x07_L090"][0], N_ROWS["x07_L090"][1])]
        for rid, less in dbl:
            for pol in STUDENTS:
                _t, recs, bad = side_tables(R, rid, {les: names_of(unit("x", "x", rid, les, pol), pols,
                                                                   ("", "_L2")) for les in less},
                                            pols[pol]["sha"], pin, tables=False)
                problems += bad
                pair_recs += recs
        pairs = pair_check(pair_recs, cache)
    halt = [p for p in pairs if not p["ok"]]
    if halt:
        status = "HALT"
    elif problems:
        status = "INCOMPLETE"
    elif len(due) > RELAUNCH_CAP:
        status = "STOP"
    elif due:
        status = "DUE"
    else:
        status = "NONE"
    return {"schema": "p2_read_near_miss_v1", "at": M.now_iso(), "status": status, "lines": lines,
            "relaunch_due": due, "n_relaunch_rows": len(due), "cap": RELAUNCH_CAP, "pairs": pairs,
            "halt_pairs": halt, "problems": problems[:30], "base_units": [list(b) for b in base_units],
            "base_stamps": current_base_stamps(pols, base_units),
            "policies": {k: {"sha256": p["sha"], "ckpt": p["ck"], "role": p["role"]} for k, p in pols.items()},
            "rule": "one-sided relaunch (P2.md Class handling): field_hard / L000 completion lines (window 0.04) and "
                    "T-reg no-regress lines (0.07) that MISS by less than the window relaunch once on both students "
                    f"(_R1); a pass never; suite never; > {RELAUNCH_CAP} rows -> STOP (notify and ask); a same-row "
                    "pair (T / x0.7 L1 vs L2) neither BITWISE-EQUAL nor CLASS-FLIP -> HALT"}


def near_sentinel(doc: dict, out_path: Path) -> str:
    s = doc["status"]
    if s == "HALT":
        return ("P2_READ HALT same-row pair beyond a class signature: "
                + " ".join(f"{p['rid']}@es{p['es']}:{sha8(p['sha256'])}={p['verdict']}" for p in doc["halt_pairs"][:6])
                + f" (stops P2; notify) near={out_path}")
    if s == "STOP":
        return (f"P2_READ STOP {doc['n_relaunch_rows']} one-sided relaunch rows > cap {RELAUNCH_CAP} "
                f"lines={','.join(k for k, v in doc['lines'].items() if v.get('near_miss'))} (notify and ask) "
                f"near={out_path}")
    if s == "INCOMPLETE":
        return f"P2_READ NEARMISS INCOMPLETE problems={len(doc['problems'])} {doc['problems'][:2]} near={out_path}"
    if s == "DUE":
        return (f"P2_READ NEARMISS DUE rows={doc['n_relaunch_rows']} lines="
                + ",".join(f"{k}(short {v['shortfall']:.3f}<{v['window']})" for k, v in doc["lines"].items()
                           if v.get("near_miss")) + f" near={out_path}")
    return f"P2_READ NEARMISS NONE pairs_ok={len(doc['pairs'])} near={out_path}"


def cmd_near_miss(out_path: Path, pairs_path: Path) -> int:
    pick, pin, pols = guards()
    thr, g_rule, r_cfg = _thr()
    R = read_rows(pols)
    cache = PairCache(pairs_path)
    doc = near_miss_doc(R, pols, pin, g_rule, r_cfg, cache)
    rp = repick_check(R)
    doc["repick_check"] = rp
    cache.save()
    if not rp["ok"]:
        raise ReadRefused(rp["why"])
    M.write_json(out_path, doc)
    say(near_sentinel(doc, out_path))
    return NEAR_RC[doc["status"]]


def t_verdict(kinds: dict) -> dict:
    ok = {k: v for k, v in kinds.items() if v.get("status") == "OK"}
    if len(ok) != len(T_KINDS):
        return {"status": "INCOMPLETE"}
    gains = sorted(k for k, v in ok.items() if v["gain"])
    regs = sorted(k for k, v in ok.items() if v["regress"])
    st = {2: "KEEPS", 1: "PARTIAL", 0: "LOST"}[len(gains)]
    return {"status": st, "gain_kinds": gains, "regressed": regs,
            "rule": "PLAN G rule per kind on the pooled es9-12 success: gain = C1S - c31000 >= max(0.10, 3 SE) and "
                    "CI lower >= 0.05; KEEPS = 2 gains, PARTIAL = 1, LOST = 0; a T regress (below c31000 - max(0.05, "
                    "2 SE)) is recorded beside the count (with 2 kinds it never changes the routing: 1 gain + 1 "
                    "regress is PARTIAL -> the user decides)",
            "le_noise_note": "g_line's 'and not le_noise' clause cannot bind at the 0.10 / 0.05 floors (|diff| <= "
                             "0.04 never reaches them; Left 13, CHARTER 06:50)"}


def route(T: dict, treg: str, N: str, Tr: str, E_flags: list) -> dict:
    """P2.md Decision rules; T LOST governs every overlap (Left 20)."""
    if T["status"] == "LOST":
        return {"route": "T_LOST", "action": "the proposal recommends no bake; the stand runs unless you stop it "
                                             "(this row governs any overlapping N / Tr / T-reg fail)"}
    if T["status"] == "KEEPS" and treg == "PASS" and N == "PASS" and Tr == "PASS" and not E_flags:
        return {"route": "RECOMMEND_YES", "action": "stand, videos; the proposal recommends yes and shows the Q3 "
                                                    "split"}
    why = [w for w, c in (("T PARTIAL", T["status"] == "PARTIAL"), ("T-reg fail", treg != "PASS"),
                          ("N fail", N != "PASS"), ("Tr fail", Tr != "PASS"), ("E flag", bool(E_flags))) if c]
    return {"route": "USER_DECIDES", "why": why, "action": "stand and videos run; the proposal presents the numbers; "
                                                           "you decide"}


def q3_split(R, pols, pin, g_rule) -> dict:
    """Report only: C1S - c31000 = (C1S - c1) + (c1 - TDR) + (TDR - c31000) at es9-10 per T kind (pooled over the
    students' two launches; the teachers' single launch). The three terms add up to the total exactly."""
    out = {}
    for kind, rid in T_KINDS.items():
        tags = {"C1S": tags_for(Q3_LES, True), "c31000": tags_for(Q3_LES, True), "c1": tags_for(Q3_LES, False),
                "TDR": tags_for(Q3_LES, False)}
        k = {}
        for lab, (a, b) in {"C1S-c31000": ("C1S", "c31000"), "C1S-c1": ("C1S", "c1"), "c1-TDR": ("c1", "TDR"),
                            "TDR-c31000": ("TDR", "c31000")}.items():
            ln = line(R, pols, pin, rid, Q3_LES, a, b, "success", tags[a], tags[b])
            if ln["status"] != "OK":
                k[lab] = {"status": ln["status"], "why": ln.get("why")}
                continue
            g = pooled_g(ln["a"], ln["b"], g_rule, rid)
            k[lab] = {"status": "OK", "new": g["new"], "ref": g["tdr"], "diff": g["diff"], "se": g["se"],
                      "ci95": g["ci95"], "n": g["n"]}
        if all(v.get("status") == "OK" for v in k.values()):
            s = k["C1S-c1"]["diff"] + k["c1-TDR"]["diff"] + k["TDR-c31000"]["diff"]
            k["sum_of_terms"] = s
            k["identity_error"] = s - k["C1S-c31000"]["diff"]
        out[kind] = k
        drop_raw_cache()
    out["rule"] = "report only (P2.md Q3); es9-10; success; pooled; paired SE per term"
    return out


def verdict_doc(R, pols, pin, pick, cache: PairCache) -> dict:  # noqa: C901
    thr, g_rule, r_cfg = _thr()
    sel_doc = P3.read_json(SEL_JSON)
    near = near_miss_doc(R, pols, pin, g_rule, r_cfg, cache, with_pairs=False)
    due_lines = {k for k, v in near["lines"].items() if v.get("near_miss")}
    rec_near = P3.read_json(NEAR_JSON)
    near_consistency = None
    if rec_near is not None:
        rec_due = {k for k, v in (rec_near.get("lines") or {}).items() if v.get("near_miss")}
        near_consistency = {"recorded": sorted(rec_due), "recomputed": sorted(due_lines), "agree": rec_due == due_lines}
    stop = len(near["relaunch_due"]) > RELAUNCH_CAP
    out = {"schema": "p2_read_v1", "at": M.now_iso(),
           "policies": {k: {"sha256": p["sha"], "ckpt": p["ck"], "role": p["role"], "label": p["label"]}
                        for k, p in pols.items()},
           "pick": {k: pick.get(k) for k in ("sha", "ck", "iter", "cand", "teacher", "sel_sha256", "name_collision")},
           "pin": {k: pin.get(k) for k in ("code_sha", "eval_sha")}, "seed_map_logical_to_real": EMAP,
           "roots": [str(r) for r in R.roots], "thresholds": {"path": str(readout.THRESHOLDS),
                                                               "sha256": M.sha256_file(readout.THRESHOLDS),
                                                               "G": g_rule, "R_completion": r_cfg},
           "near_miss": {"lines": near["lines"], "due_rows": len(near["relaunch_due"]), "stop": stop,
                         "consistency_with_record": near_consistency},
           "computation": "pooled per (es, env_index): mean over the key's launches of float(bool(outcome)); share = "
                          "plain mean over keys; SE = thresholds.paired_completion_se on the fractional dicts; "
                          "g_line / retention_line clauses on those numbers (never readout.share / paired / "
                          "retention_line on pooled dicts; never keyed (launch, env_index))"}
    incomplete, pending = [], []
    all_recs = []
    # ---- T
    tk = {}
    for kind, rid in T_KINDS.items():
        ln = line(R, pols, pin, rid, T_LES, "C1S", "c31000", "success", tags_for(T_LES, True), tags_for(T_LES, True))
        rec = public(ln)
        all_recs += ln["stamps"]["new"] + ln["stamps"]["ref"]
        if ln["status"] == "OK":
            rec.update(pooled_g(ln["a"], ln["b"], g_rule, rid))
            rec["variant_split"] = variant_split(ln["tables"]["new"], ln["tables"]["ref"], kind)
            rec["falls"] = {side: {r["name"] + f"@es{r['es']}": falls_by_hold(Path(r["stamp"]))
                                   for r in ln["stamps"][side] if r["status"] == "OK"} for side in ("new", "ref")}
        else:
            incomplete.append(f"T {kind}: {ln['status']} {(ln.get('why') or [])[:2]}")
        tk[kind] = rec
        drop_raw_cache()
    T = {"kinds": tk, **t_verdict(tk)}
    out["T"] = T
    # ---- T-reg
    tr_k, treg_st = {}, []
    for kind, rid in TREG_KINDS.items():
        rel = kind in due_lines
        ln, pend = decisive_line(R, pols, pin, rid, TREG_LES, "success", False, rel, stop)
        rec = public(ln)
        all_recs += ln["stamps"]["new"] + ln["stamps"]["ref"]
        if ln["status"] == "OK":
            g = pooled_g(ln["a"], ln["b"], g_rule, rid)
            if ln["n_launches"]["new"] != ln["n_launches"]["ref"]:
                raise ReadRefused(f"T-reg {kind}: launch counts differ between the students {ln['n_launches']}")
            if kind == "ledge_up":
                rec["variant_split"] = variant_split(ln["tables"]["new"], ln["tables"]["ref"], kind)
            if pend:
                rec.update(provisional=g, relaunch_due=True, pending=pend)
                pending.append(f"T-reg {kind}: {pend}")
                treg_st.append("PENDING")
            else:
                rec.update(g, no_regress=not g["regress"], relaunched=rel)
                treg_st.append("PASS" if not g["regress"] else "FAIL")
        else:
            incomplete.append(f"T-reg {kind}: {ln['status']} {(ln.get('why') or [])[:2]}")
            treg_st.append("INCOMPLETE")
        tr_k[kind] = rec
        drop_raw_cache()
    out["T_reg"] = {"kinds": tr_k, "status": _agg(treg_st),
                    "rule": "the frozen G regress clause per kind on the pooled es9-10 success: C1S >= c31000 - "
                            "max(0.05, 2 SE)"}
    # ---- N
    n_l, n_st = {}, []
    for lname, (rid, less, n_launch) in N_ROWS.items():
        rel = lname in due_lines
        ln, pend = decisive_line(R, pols, pin, rid, less, "completion", n_launch == 2, rel, stop)
        rec = public(ln)
        all_recs += ln["stamps"]["new"] + ln["stamps"]["ref"]
        if ln["status"] == "OK":
            r = pooled_r(ln["a"], ln["b"], r_cfg, rid)
            if pend:
                rec.update(provisional=r, relaunch_due=True, pending=pend, double_launch=(n_launch == 2))
                pending.append(f"N {lname}: {pend}")
                n_st.append("PENDING")
            else:
                rec.update(r, relaunched=rel, double_launch=(n_launch == 2))
                n_st.append("PASS" if r["pass"] else "FAIL")
        else:
            incomplete.append(f"N {lname}: {ln['status']} {(ln.get('why') or [])[:2]}")
            n_st.append("INCOMPLETE")
        n_l[lname] = rec
        drop_raw_cache()
    out["N"] = {"rows": n_l, "status": _agg(n_st), "rule": "completion: C1S >= c31000 - max(0.05, 2 SE) "
                                                           "(select_student.completion_margin, frozen R_completion)"}
    # ---- Tr + E
    try:
        tr = suite_tr(R, pols, pin)
    except readout.KeySetError as e:
        tr = {"status": "INVALID", "why": [str(e)]}
    if tr["status"] in ("INCOMPLETE", "INVALID"):
        incomplete.append(f"Tr: {tr['status']} {(tr.get('why') or [])[:2]}")
    tr["diag_label"] = diag_label(tr, sel_doc) if tr["status"] == "FAIL" else None
    tr["report"] = suite_report(tr) if tr.get("stamps") else None
    out["Tr"] = tr
    e_flags = (tr.get("E") or {}).get("flags") or []
    # ---- report only
    rep = {}
    for kind, rid in REPORT_KINDS.items():
        ln = line(R, pols, pin, rid, REPORT_LES, "C1S", "c31000", "success", tags_for(REPORT_LES, False),
                  tags_for(REPORT_LES, False))
        rec = public(ln)
        all_recs += ln["stamps"]["new"] + ln["stamps"]["ref"]
        if ln["status"] == "OK":
            g = pooled_g(ln["a"], ln["b"], g_rule, rid)
            rec.update({k: g[k] for k in ("new", "tdr", "diff", "se", "ci95", "n")}, report_only=True)
        rep[kind] = rec
        drop_raw_cache()
    out["report"] = {"kinds": rep, "q3_split": q3_split(R, pols, pin, g_rule),
                     "dprime_step3": (P3.read_json(STEP3_JSON) or {}).get("dprime"),
                     "note": "report only; never gated (P2.md Gates 'Report only')"}
    # ---- the class pairs (HALT) over every double / relaunched row
    pairs = pair_check([r for r in all_recs if r.get("status") == "OK"], cache)
    out["pairs"] = pairs
    halt = [p for p in pairs if not p["ok"]]
    # ---- the decision
    if halt:
        status = "HALT"
    elif incomplete:
        status = "INCOMPLETE"
    elif pending:
        status = "PENDING"
    else:
        status = "COMPUTED"
    out["status"] = status
    out["incomplete"], out["pending"] = incomplete, pending
    if status == "COMPUTED":
        out["decision"] = route(T, out["T_reg"]["status"], out["N"]["status"], tr["status"], e_flags)
    elif status == "PENDING":
        out["decision"] = {"route": "PENDING", "action": "one-sided relaunch rows are due but not run "
                                                         + ("(over the cap: notify and ask)" if stop else "")}
    elif status == "HALT":
        out["decision"] = {"route": "HALT", "action": "a same-row pair beyond a class signature stops P2 (notify)"}
    else:
        out["decision"] = {"route": "INCOMPLETE", "action": "rows missing or invalid: fix before the verdict"}
    out["served"] = [{k: s.get(k) for k in ("rid", "logical_es", "real_es", "sha256", "name", "status", "stamp",
                                            "name_dir")} for s in R.served]
    return out


def _agg(sts: list) -> str:
    if "INCOMPLETE" in sts:
        return "INCOMPLETE"
    if "PENDING" in sts:
        return "PENDING"
    return "FAIL" if "FAIL" in sts else "PASS"


def verdict_sentinel(doc: dict, out_path: Path) -> str:
    T = doc["T"]
    kinds = " ".join(f"{k}:diff={v.get('diff', float('nan')):+.3f},se={v.get('se', float('nan')):.3f},"
                     f"gain={int(bool(v.get('gain')))},reg={int(bool(v.get('regress')))}"
                     for k, v in T["kinds"].items())
    return (f"P2_READ VERDICT {doc['status']} route={doc['decision']['route']} T={T.get('status')} [{kinds}] "
            f"T-reg={doc['T_reg']['status']} N={doc['N']['status']} Tr={doc['Tr'].get('status')} "
            f"E={'none' if not (doc['Tr'].get('E') or {}).get('flags') else len(doc['Tr']['E']['flags'])} "
            f"pick={sha8(doc['pick']['sha'])} read={out_path}")


def cmd_verdict(out_path: Path, pairs_path: Path) -> int:
    pick, pin, pols = guards()
    R = read_rows(pols)
    cache = PairCache(pairs_path)
    doc = verdict_doc(R, pols, pin, pick, cache)
    rp = repick_check(R)
    doc["repick_check"] = rp
    cache.save()
    if not rp["ok"]:
        raise ReadRefused(rp["why"])
    bad = [s for s in R.served if s.get("stamp") and s["real_es"] not in REAL_OK]
    if bad:
        raise ReadRefused(f"served real es outside {REAL_OK}: {bad[:3]}")
    M.write_json(out_path, doc)
    say(verdict_sentinel(doc, out_path))
    return VERDICT_RC[doc["status"]]


# ============================================================================== real blocks
def run_block(blk: str, rest: list, pols: dict) -> int:
    """``rt_queue.main`` for one real block (p2_step3.run_block's rc rule: 20 REFUSED when it ended non-zero with no
    row launched, 21 HALT on the DDC-growth / guard stop, else rt_queue's rc)."""
    try:
        row_labels = {s.label for s in plan_read(blk, pols) if s.slot}
    except (ValueError, RuntimeError, OSError):
        row_labels = set()
    seen = {"rows": [], "done": None}
    child0, say0 = Q.run_child, Q.say

    def child(argv, env, log):
        if log is not None and Path(log).stem in row_labels:
            seen["rows"].append(Path(log).stem)
        return child0(argv, env, log)

    def say_(line_: str) -> None:
        if line_.startswith("RTQ_DONE "):
            seen["done"] = line_
        say0(line_)

    Q.run_child, Q.say = child, say_
    try:
        rc = Q.main(rest)
    finally:
        Q.run_child, Q.say = child0, say0
    if rc == 0:
        return 0
    done = seen["done"] or ""
    if "DerivedDataCache grew" in done or "guard stopped the queue" in done:
        say(f"P2_READ HALT block={blk} rc={HALT_RC} rows_launched={len(seen['rows'])} -- {done}")
        return HALT_RC
    if not seen["rows"]:
        say(f"P2_READ REFUSED block={blk} rc={REFUSED_RC} (rt_queue rc {rc}; no row launched) -- {done}")
        return REFUSED_RC
    return rc


# ============================================================================== main
def main(argv=None) -> int:  # noqa: C901
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], allow_abbrev=False)
    ap.add_argument("--block", choices=tuple(BLOCKS), default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--queue-dry", action="store_true")
    ap.add_argument("--from", dest="from_label", default=None)
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--inputs-check", action="store_true")
    ap.add_argument("--near-miss", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--pairs-cache", type=Path, default=PAIRS_JSON)
    a = ap.parse_args(argv)
    if P3._CALLER_RT_TP_OUT and Path(P3._CALLER_RT_TP_OUT) != P2_OUT:
        say(f"P2_READ REFUSED caller RT_TP_OUT={P3._CALLER_RT_TP_OUT} (this driver writes only {P2_OUT})")
        return 2
    MODE["dry"] = a.dry or a.queue_dry
    if MODE["dry"]:
        os.environ["P2_DRY"] = "1"
    try:
        if a.inputs_check:
            if not a.block:
                ap.error("--inputs-check needs --block")
            return inputs_check(a.block)
        if a.near_miss:
            return cmd_near_miss(a.out or NEAR_JSON, a.pairs_cache)
        if a.verdict:
            return cmd_verdict(a.out or READ_JSON, a.pairs_cache)
        if a.dry:
            return dry(a.block)
        if not a.block:
            ap.error("one of --dry / --block / --inputs-check / --near-miss / --verdict")
        blk = BLOCKS[a.block]
        pick = load_pick(strict=not a.queue_dry)
        if not pick["ok"] and not a.queue_dry:
            say(M.done_line(blk, 1, f"(pick: {'; '.join(pick['problems'])})"))
            say(f"P2_READ REFUSED block={blk} rc={REFUSED_RC} (no valid results/p2/sel.json pick; no row launched)")
            return REFUSED_RC
        if not pick["ok"]:
            say("P2_READ QUEUE-DRY refused: rt_queue's dry pass runs gate_row.sh, which needs a real checkpoint "
                "(no valid sel.json pick); use --dry")
            return 2
        pols = policies(pick)
        if not a.queue_dry:
            why = fixed_policy_problems(check_iter=True)
            if why:
                say(M.done_line(blk, 1, f"(policy identity: {'; '.join(why)})"))
                say(f"P2_READ REFUSED block={blk} rc={REFUSED_RC} (policy identity; no row launched)")
                return REFUSED_RC
        install(pols)
        rest = ["--block", blk] + (["--dry"] if a.queue_dry else [])
        if a.from_label:
            rest += ["--from", a.from_label]
        for o in a.only or []:
            rest += ["--only", o]
        return Q.main(rest) if a.queue_dry else run_block(blk, rest, pols)
    except (ReadRefused, SeedError, readout.KeySetError) as e:
        say(f"P2_READ REFUSED {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
