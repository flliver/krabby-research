#!/usr/bin/env python3
"""P1-B6 reader (CHARTER 12): the pre-registered reads, the t3 stop gate and the final gate of the phase-1 chain
(t3 6000 from scratch -> c4 -> c5 -> c6 -> c6x clock hidden), B lineage vs C0 lineage at seeds 16-18.

CPU only, read-only. Records: $OUT/arms_p1/<ref>.record.json (run_arm.sh, ARMS_DIR=$OUT/arms_p1):
  t3    P1B6__<arm>_t3_s<seed>   head model_5999 (both-hands lr eval + driver suite stamp), model_4500 lr
  tail  P1B6__<arm>_tail_s<seed> head = the c6x head (lr + driver suite stamp, CLOCK HIDDEN), model_13000 lr
        (the cut transient), and the c6 pre-cut lr read <ref>_c6_lr.json (run_arm.sh --lr-eval) whose stamp env
        is the c6 stage's own suite stamp (clock visible)
  W     W_p1b6_lr.json + $OUT/lr/W_p1b6/suite__omni/seed001/<ts> (W's head re-scored at this code_sha, clock hidden)
Classes are verdict.py's (feet AND along, fail-closed NA) via x6k_read.read_point; dose via x6k_read.tail_reads.

Suite reads (aggregate.suite): fwd / back / diag / side ratio (by_category.<c>.ratio.median), turn facing_closed and
yaw_progress (by_category.turn.<k>.median), move_turn ratio (median) and survival (mean of survival_by_hold
move_turn / move_turn_r), and per-category action_jitter / action_fft_freq medians.

Gates (CHARTER 12.4 / 12.5):
  t3 gate   STOP if B has >= 1 FAST hand at its t3 head in <= 1/3 seeds, or B's second (weaker) hand is below
            INTERMEDIATE (SLOW / DEAD) in >= 2/3 seeds (the INC0-like one-hand outcome; FB6k's second hands were all
            >= INT at the same exposure), or C0 is FAST/FAST at t3 in >= 2/3, or a t3 read is missing or NA.
  final     (c6x heads, clock hidden; first match decides)
            INVALID EVAL   a c6x suite or lr stamp whose env lacks KRABBY_OMNI_CLOCK_HIDE=1 -> stop and ask
            PENDING        any needed read missing
            CLOCK-CUT LOSS B FAST/FAST (hands) at the c6 pre-cut read in >= 2/3 but FAST/FAST (hands) at c6x in <= 1/3
            FAIL           B counted FAST/FAST (both hands, untripped, dose ok) <= 1/3
            COST           B counted FAST/FAST >= 2/3 but a gated category is CLEARLY inferior to the C0 lineage
                           (seed-mean < C0's - margin AND B's best < C0's worst; named)
            PASS           B >= 2/3, C0 FAST/FAST <= 1/3, no B trip, every gated category holds (seed-mean >= C0's -
                           margin)
            PARTIAL        otherwise
  gated     fwd / back / diag ratio and turn facing_closed (margin 0.03), turn yaw_progress (margin 0.01, ~10 % of W's
            0.102). move_turn survival and ratio are REPORTED, not gated (the clock cut alone took W's move_turn
            survival 0.865 -> 0.365; pre-launch review 2026-10-04). A NA hand class (a missing feet/along read) is
            PENDING, never a decided not-FAST.
  reported  per-category deltas vs W; the adoption-candidate flag (a B seed FAST/FAST at c6x whose side ratio beats
            W's and no category is > 0.05 below W); the smoothness flag (B's seed-mean jitter > 1.5x C0's in any
            category, or fft > C0's + 0.3).

    p1_read.py [--arms D] [--json OUT] [--md OUT] [--no-tb]
    p1_read.py --t3-gate              exit 0 continue / 4 stop
    p1_read.py --c6 <tail ref>        prints "<c6 head checkpoint> <c6 suite stamp>" (for run_arm.sh --lr-eval)
    p1_read.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
R = Path("/home/nickmagus/krabby/krabby-research")
OUT = R / "parkour/outputs/omni_plan/ppo_construction"
EV = R / "parkour/logs/rsl_rl/gait_eval/omni_phases" / CAMPAIGN.name
_spec = importlib.util.spec_from_file_location("_p1_x6k", HERE / "x6k_read.py")
X = importlib.util.module_from_spec(_spec)
sys.modules["_p1_x6k"] = X
_spec.loader.exec_module(X)

FAST = "FAST"
MATERIAL = 0.03
W_MARGIN = 0.05
TRIP_FLOOR, TRIP_MARGIN, TRIP_COLLAPSE = 0.15, 0.10, 0.25
SEEDS = (16, 17, 18)
HANDS = ("l", "r")
#: the gated categories (name, extractor key): ratio-type and turn metrics, "higher is better"
GATED = ("fwd_ratio", "back_ratio", "diag_ratio", "turn_facing_closed", "turn_yaw_progress")
MARGINS = {"fwd_ratio": 0.03, "back_ratio": 0.03, "diag_ratio": 0.03, "turn_facing_closed": 0.03,
           "turn_yaw_progress": 0.01, "move_turn_survival": 0.03, "move_turn_ratio": 0.03, "side_ratio": 0.03}
REPORTED = GATED + ("move_turn_survival", "move_turn_ratio", "side_ratio")
Q_CATS = ("fwd", "back", "side", "diag", "turn", "move_turn")
W_NAME = "W_p1b6"


def ff(pt) -> bool:
    return bool(pt) and pt["hands"]["l"] == FAST and pt["hands"]["r"] == FAST


def n_fast(pt) -> int:
    return sum(pt["hands"][h] == FAST for h in HANDS) if pt else 0


def has_na(pt) -> bool:
    return bool(pt) and any(pt["hands"][h] == "NA" for h in HANDS)


def second_below_int(pt) -> bool:
    """The weaker hand (by feet speed) is SLOW or DEAD."""
    if not pt:
        return False
    f = pt["feet"]
    weak = min(HANDS, key=lambda h: (f[h] if f[h] is not None else -1))
    return pt["hands"][weak] in ("SLOW", "DEAD")


def _med(d, *keys):
    for k in keys:
        d = (d or {}).get(k)
    return d.get("median") if isinstance(d, dict) else d


def suite_reads(stamp) -> dict | None:
    if not stamp:
        return None
    try:
        a = json.load(open(Path(stamp) / "scenario_metrics.json"))["aggregate"]["suite"]
    except (OSError, ValueError, KeyError):
        return None
    by, surv = a.get("by_category") or {}, a.get("survival_by_hold") or {}
    mt = [surv.get(h) for h in ("move_turn", "move_turn_r") if surv.get(h) is not None]
    out = {"fwd_ratio": _med(by, "fwd", "ratio"), "back_ratio": _med(by, "back", "ratio"),
           "diag_ratio": _med(by, "diag", "ratio"), "side_ratio": _med(by, "side", "ratio"),
           "turn_facing_closed": _med(by, "turn", "facing_closed"), "turn_yaw_progress": _med(by, "turn", "yaw_progress"),
           "move_turn_ratio": _med(by, "move_turn", "ratio"),
           "move_turn_survival": sum(mt) / len(mt) if mt else None,
           "jitter": {c: _med(by, c, "action_jitter") for c in Q_CATS},
           "fft": {c: _med(by, c, "action_fft_freq") for c in Q_CATS},
           "survival": dict(surv)}
    return out


def clock_hidden(stamp) -> bool | None:
    if not stamp:
        return None
    try:
        env = json.load(open(Path(stamp) / "run_meta.json")).get("env_vars") or {}
    except (OSError, ValueError):
        return None
    return str(env.get("KRABBY_OMNI_CLOCK_HIDE")) == "1"


def stamp_of_run_dir(run_dir: str | None) -> str | None:
    """The newest complete driver stamp whose run_meta names this run dir (any row)."""
    if not run_dir:
        return None
    rd = Path(run_dir)
    tag = "_".join(rd.name.split("_")[2:])
    import re  # noqa: PLC0415
    tag = re.sub(r"_(e[0-9]+|c|w)$", "", tag)   # a retry (_e2) or continuation (_c / _w) files evals under <tag>
    hits = []
    for d in sorted((EV / tag).glob("*/seed001/*/")):
        try:
            if rd.name in (d / "run_meta.json").read_text() and (d / "complete.json").is_file():
                hits.append(d)
        except OSError:
            pass
    return str(hits[-1]) if hits else None


def c6_paths(ref: str, state_path: Path = CAMPAIGN / "state.json") -> tuple[str | None, str | None]:
    """(c6 head checkpoint, c6 suite stamp) of a tail reference from state.json's stage_ckpts."""
    try:
        recs = json.load(open(state_path)).get("ladder", {}).get("references", {}).get(ref, [])
    except (OSError, ValueError):
        return None, None
    if not recs:
        return None, None
    ck = (recs[-1].get("stage_ckpts") or {}).get("c6")
    return ck, stamp_of_run_dir(str(Path(ck).parent)) if ck else None


def load_run(rec: dict, arms: Path, with_tb: bool = True) -> dict:
    ref = rec["ref"]
    kind = "t3" if "_t3_" in ref else "tail"
    arm = "B" if "__B_" in ref else "C0"
    pts = {n: X.read_point(e.get("class_json"), e.get("along_json"), f"{ref}_m{n}")
           for n, e in (rec.get("extra_lr") or {}).items()}
    pts["head"] = X.read_point(rec.get("lr_class_json"), rec.get("lr_along_json"), ref)
    run = {"ref": ref, "kind": kind, "arm": arm, "seed": int(rec.get("seed") or 0), "killed": rec.get("killed"),
           "killed_kind": rec.get("killed_kind"), "points": pts, "suite": suite_reads(rec.get("driver_stamp")),
           "tail": X.tail_reads(rec.get("run_dir")) if (with_tb and kind == "tail") else {},
           "clock_hidden": {"suite": clock_hidden(rec.get("driver_stamp")), "lr": clock_hidden(rec.get("lr_stamp"))}}
    if kind == "tail":
        run["points"]["c6"] = X.read_point(str(arms / f"{ref}_c6_lr.json"), str(arms / f"{ref}_c6_lr_along.json"),
                                           f"{ref}_c6")
    return run


def t3_gate(runs: list[dict]) -> dict:
    t3 = {(r["arm"], r["seed"]): r for r in runs if r["kind"] == "t3"}
    miss = [f"{a} s{s}" for a in ("B", "C0") for s in SEEDS
            if (a, s) not in t3 or t3[(a, s)].get("killed") or not t3[(a, s)]["points"].get("head")
            or has_na(t3[(a, s)]["points"]["head"])]
    out = {"missing": miss}
    if miss:
        out.update(stop=True, reading="INCOMPLETE", why=f"t3 reads missing: {miss} -- stop and ask")
        return out
    b_any = sum(n_fast(t3[("B", s)]["points"]["head"]) >= 1 for s in SEEDS)
    b_weak = sum(second_below_int(t3[("B", s)]["points"]["head"]) for s in SEEDS)
    c_ff = sum(ff(t3[("C0", s)]["points"]["head"]) for s in SEEDS)
    out.update(B_with_a_fast_hand=b_any, B_second_hand_below_int=b_weak, C0_fast_fast=c_ff,
               per_seed={f"{a} s{s}": {"head": t3[(a, s)]["points"]["head"]["hands"],
                                       "m4500": (t3[(a, s)]["points"].get("4500") or {}).get("hands")}
                         for a in ("B", "C0") for s in SEEDS})
    if b_any <= 1:
        out.update(stop=True, reading="STOP: B FORMATION", why=f"B has a FAST hand at t3 in {b_any}/3 seeds (<= 1/3)")
    elif b_weak >= 2:
        out.update(stop=True, reading="STOP: SECOND HAND",
                   why=f"B's weaker hand is below INTERMEDIATE at t3 in {b_weak}/3 seeds (the INC0-like one-hand outcome)")
    elif c_ff >= 2:
        out.update(stop=True, reading="STOP: C0 FORMS", why=f"C0 FAST/FAST at t3 in {c_ff}/3 (>= 2/3): the construction effect is unresolved")
    else:
        out.update(stop=False, reading="CONTINUE", why=f"B a FAST hand in {b_any}/3, C0 FAST/FAST {c_ff}/3: run the tails")
    return out


def _cat_compare(b_vals, c_vals, margin=MATERIAL):  # noqa: D103 (margin per metric: MARGINS)
    if len(b_vals) != 3 or len(c_vals) != 3 or None in b_vals or None in c_vals:
        return None
    mb, mc = sum(b_vals) / 3, sum(c_vals) / 3
    return {"B_mean": mb, "C0_mean": mc, "holds": mb >= mc - margin,
            "clearly_inferior": mb < mc - margin and max(b_vals) < min(c_vals), "B": b_vals, "C0": c_vals}


def evaluate(runs: list[dict], w: dict | None = None) -> dict:
    res = {"t3": t3_gate(runs), "seeds": {}, "categories": {}, "notes": [], "invalid": [], "pending": []}
    tail = {(r["arm"], r["seed"]): r for r in runs if r["kind"] == "tail"}
    for a in ("B", "C0"):
        for s in SEEDS:
            r = tail.get((a, s))
            if r is None or r.get("killed") or not r["points"].get("head"):
                res["pending"].append(f"{a} s{s} c6x head" + (" (killed)" if r and r.get("killed") else ""))
                continue
            if r["clock_hidden"]["suite"] is not True or r["clock_hidden"]["lr"] is not True:
                res["invalid"].append(f"{r['ref']}: c6x eval env lacks KRABBY_OMNI_CLOCK_HIDE=1 "
                                      f"(suite {r['clock_hidden']['suite']}, lr {r['clock_hidden']['lr']})")
            if r["suite"] is None:
                res["pending"].append(f"{a} s{s} c6x suite read")
            if r["points"].get("c6") is None:
                res["pending"].append(f"{a} s{s} c6 pre-cut read")
            if has_na(r["points"]["head"]) or (a == "B" and has_na(r["points"].get("c6"))):
                res["pending"].append(f"{a} s{s} NA hand class (a missing feet/along read)")
    c0f = [tail[("C0", s)]["points"]["head"]["fall"] for s in SEEDS
           if ("C0", s) in tail and tail[("C0", s)]["points"].get("head")]
    c0f = [x for x in c0f if x is not None and x <= TRIP_COLLAPSE]
    ref = statistics.median(c0f) if c0f else None
    thr = max(TRIP_FLOOR, ref + TRIP_MARGIN) if ref is not None else TRIP_FLOOR
    res["tripwire"] = {"reference": ref, "threshold": thr}
    for (a, s), r in sorted(tail.items()):
        head = r["points"].get("head")
        if not head:
            continue
        tripped = a == "B" and head["fall"] is not None and head["fall"] > thr
        dose_ok = r["tail"].get("dose_ok") if a == "B" else None
        if a == "B" and ff(head) and not tripped and dose_ok is None:
            res["pending"].append(f"B s{s} dose read")
        res["seeds"][f"{a} s{s}"] = {
            "c6x": head["hands"], "c6x_feet": head["feet"], "fall": head["fall"], "tripped": tripped,
            "dose": {k: r["tail"].get(k) for k in ("mu_med_cam", "mu_med_rod", "dose_ok")},
            "counted": ff(head) and not tripped and (a != "B" or dose_ok is True),
            "c6_fast_fast": ff(r["points"].get("c6")), "c6": (r["points"].get("c6") or {}).get("hands"),
            "m13000": (r["points"].get("13000") or {}).get("hands"), "suite": r["suite"]}
    kB = sum(v["counted"] for k, v in res["seeds"].items() if k.startswith("B "))
    kC = sum(ff({"hands": v["c6x"]}) for k, v in res["seeds"].items() if k.startswith("C0 "))
    k6 = sum(v["c6_fast_fast"] for k, v in res["seeds"].items() if k.startswith("B "))
    k6x_hands = sum(ff({"hands": v["c6x"]}) for k, v in res["seeds"].items() if k.startswith("B "))
    anyB_trip = any(v["tripped"] for k, v in res["seeds"].items() if k.startswith("B "))
    for c in REPORTED:
        bv = [((res["seeds"].get(f"B s{s}") or {}).get("suite") or {}).get(c) for s in SEEDS]
        cv = [((res["seeds"].get(f"C0 s{s}") or {}).get("suite") or {}).get(c) for s in SEEDS]
        res["categories"][c] = _cat_compare(bv, cv, MARGINS[c])
    res["counts"] = {"B_counted_fast_fast_c6x": kB, "C0_fast_fast_c6x": kC, "B_fast_fast_c6": k6,
                     "B_fast_fast_hands_c6x": k6x_hands, "B_tripped": anyB_trip}
    cats = {c: res["categories"][c] for c in GATED}
    if res["invalid"]:
        res["gate"], res["why"] = "INVALID EVAL", "; ".join(res["invalid"]) + " -- stop and ask"
    elif res["pending"] or any(v is None for v in cats.values()):
        res["gate"] = "PENDING"
        res["why"] = "missing: " + ", ".join(res["pending"] + [f"category {c}" for c, v in cats.items() if v is None])
    elif k6 >= 2 and k6x_hands <= 1:
        res["gate"], res["why"] = "CLOCK-CUT LOSS", f"B FAST/FAST (hands) at c6 {k6}/3 but at c6x {k6x_hands}/3 -- stop and ask"
    elif kB <= 1:
        res["gate"], res["why"] = "FAIL", f"B counted FAST/FAST {kB}/3 at c6x (hands {k6x_hands}/3)"
    elif any(v["clearly_inferior"] for v in cats.values()):
        bad = [c for c, v in cats.items() if v["clearly_inferior"]]
        res["gate"], res["why"] = "COST", f"B counted FAST/FAST {kB}/3, but clearly inferior to the C0 lineage on {bad}"
    elif kB >= 2 and kC <= 1 and not anyB_trip and all(v["holds"] for v in cats.values()):
        res["gate"], res["why"] = "PASS", f"B FAST/FAST {kB}/3 at c6x (clock hidden), C0 {kC}/3, no B trip, every category holds"
    else:
        why = []
        if kC >= 2:
            why.append(f"C0 FAST/FAST {kC}/3")
        if anyB_trip:
            why.append("a B run tripped")
        why += [f"{c} short (B {v['B_mean']:.3f} < C0 {v['C0_mean']:.3f} - {MARGINS[c]})" for c, v in cats.items() if not v["holds"]]
        res["gate"], res["why"] = "PARTIAL", f"B FAST/FAST {kB}/3; " + "; ".join(why)
    # ---- reported, not gated: vs W, adoption-candidate flag, smoothness flag
    if w and w.get("suite"):
        ws = w["suite"]
        res["vs_W"] = {c: {f"B s{s}": (None if ((res["seeds"].get(f"B s{s}") or {}).get("suite") or {}).get(c) is None
                                       or ws.get(c) is None else res["seeds"][f"B s{s}"]["suite"][c] - ws[c])
                           for s in SEEDS} for c in REPORTED}
        res["W"] = {"suite": {c: ws.get(c) for c in REPORTED}, "lr": (w.get("lr") or {}).get("hands")}
        cands = []
        for s in SEEDS:
            v = res["seeds"].get(f"B s{s}")
            if not v or not v["counted"] or not v["suite"]:
                continue
            sv = v["suite"]
            if (sv.get("side_ratio") or 0) > (ws.get("side_ratio") or 9) and all(
                    sv.get(c) is not None and ws.get(c) is not None and sv[c] >= ws[c] - W_MARGIN for c in GATED):
                cands.append(f"P1B6__B_tail_s{s}")
        res["adoption_candidates"] = cands
    else:
        res["notes"].append("W re-score missing: vs-W and the adoption-candidate flag are unread")
    flags = []
    for c in Q_CATS:
        bj = [((res["seeds"].get(f"B s{s}") or {}).get("suite") or {}).get("jitter", {}).get(c) for s in SEEDS]
        cj = [((res["seeds"].get(f"C0 s{s}") or {}).get("suite") or {}).get("jitter", {}).get(c) for s in SEEDS]
        bf = [((res["seeds"].get(f"B s{s}") or {}).get("suite") or {}).get("fft", {}).get(c) for s in SEEDS]
        cf = [((res["seeds"].get(f"C0 s{s}") or {}).get("suite") or {}).get("fft", {}).get(c) for s in SEEDS]
        if None in bj + cj + bf + cf or not bj:
            continue
        mbj, mcj, mbf, mcf = (sum(x) / 3 for x in (bj, cj, bf, cf))
        if mbj > 1.5 * mcj or mbf > mcf + 0.3:
            flags.append(f"{c}: B jitter {mbj:.3f} vs C0 {mcj:.3f}, fft {mbf:.2f} vs {mcf:.2f}")
    res["smoothness_flag"] = flags
    return res


def load_all(arms: Path, with_tb: bool = True) -> tuple[list[dict], dict | None]:
    runs = [load_run(json.load(open(p)), arms, with_tb) for p in sorted(arms.glob("P1B6__*.record.json"))]
    w = None
    wst = sorted((OUT / "lr" / W_NAME / "suite__omni" / "seed001").glob("*/"))
    wlr = X.read_point(str(arms / f"{W_NAME}_lr.json"), str(arms / f"{W_NAME}_lr_along.json"), W_NAME)
    if wst or wlr:
        w = {"suite": suite_reads(str(wst[-1])) if wst else None, "lr": wlr,
             "clock_hidden": clock_hidden(str(wst[-1])) if wst else None}
    return runs, w


def _f(x, fmt="{:.3f}"):
    return "--" if x is None else fmt.format(x)


def markdown(res: dict) -> str:
    t = res["t3"]
    L = ["# P1-B6 -- the pre-registered reading (CHARTER 12)", "",
         f"**t3 gate: {t['reading']}** -- {t['why']}", ""]
    if t.get("per_seed"):
        L += [f"- t3 {k}: head {v['head']['l']}/{v['head']['r']}"
              + (f"; model_4500 {v['m4500']['l']}/{v['m4500']['r']}" if v.get("m4500") else "") for k, v in t["per_seed"].items()] + [""]
    L += ["| seed | c6 (pre-cut) | model_13000 | c6x head (clock hidden) | fall | dose cam/rod | counted | "
          + " | ".join(REPORTED) + " |", "|" + "---|" * (7 + len(REPORTED))]
    for k, v in sorted(res["seeds"].items()):
        s = v["suite"] or {}
        hh = lambda h: "/".join(x[:4] for x in (h or {}).values()) if h else "--"  # noqa: E731
        L.append(f"| {k} | {hh(v['c6'])} | {hh(v['m13000'])} | {hh(v['c6x'])} {_f(v['c6x_feet']['l'])}/{_f(v['c6x_feet']['r'])} "
                 f"| {_f(v['fall'])}{' TRIPPED' if v['tripped'] else ''} | {_f(v['dose']['mu_med_cam'], '{:.2f}')}/"
                 f"{_f(v['dose']['mu_med_rod'], '{:.2f}')} | {v['counted']} | " + " | ".join(_f(s.get(c)) for c in REPORTED) + " |")
    if res.get("W"):
        L.append(f"| W (re-scored) | | | {'/'.join(x[:4] for x in (res['W']['lr'] or {}).values()) if res['W']['lr'] else '--'} | | | | "
                 + " | ".join(_f(res["W"]["suite"].get(c)) for c in REPORTED) + " |")
    L += ["", f"Tripwire: reference {_f(res['tripwire']['reference'])}, threshold {_f(res['tripwire']['threshold'])}", ""]
    for c, v in res["categories"].items():
        if v:
            L.append(f"- {c}{'' if c in GATED else ' (reported, not gated)'}: B {v['B_mean']:.3f} vs C0 {v['C0_mean']:.3f} "
                     f"(margin {MARGINS[c]}) -> holds {v['holds']}, clearly inferior {v['clearly_inferior']}")
    L += ["", f"**Final gate: {res['gate']}** -- {res['why']}", "",
          f"- adoption candidates (reported, not a decision): {res.get('adoption_candidates', 'unread')}",
          f"- smoothness flag (reported): {res['smoothness_flag'] or 'none'}"]
    L += [f"- note: {n}" for n in res["notes"]]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ self-test
def _pt(lc, rc, lf=0.2, rf=0.2, fall=0.01):
    return {"hands": {"l": lc, "r": rc}, "feet": {"l": lf, "r": rf}, "fall": fall}


def _suite(v=0.7, jit=0.15, fft=1.4):
    d = {c: v for c in REPORTED}
    d["jitter"] = {c: jit for c in Q_CATS}
    d["fft"] = {c: fft for c in Q_CATS}
    return d


def _run(ref, kind, arm, seed, head, c6=None, suite=None, dose_ok=True, hidden=True):
    pts = {"head": head}
    if kind == "tail":
        pts["c6"] = c6 if c6 is not None else head
    return {"ref": ref, "kind": kind, "arm": arm, "seed": seed, "killed": None, "points": pts,
            "suite": _suite() if suite is None else suite, "tail": {"dose_ok": dose_ok},
            "clock_hidden": {"suite": hidden, "lr": hidden}}


def _self_test() -> int:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}")

    F, I, S = FAST, "INTERMEDIATE", "SLOW"
    FF, SS, FI = _pt(F, F), _pt(S, S, 0.08, 0.08), _pt(F, I)

    def t3s(bh, ch):
        return [_run(f"P1B6__B_t3_s{s}", "t3", "B", s, h) for s, h in zip(SEEDS, bh)] + \
            [_run(f"P1B6__C0_t3_s{s}", "t3", "C0", s, h) for s, h in zip(SEEDS, ch)]
    check("t3: B a FAST hand 2/3, C0 0/3 -> CONTINUE", t3_gate(t3s([FI, FI, SS], [SS] * 3))["stop"], False)
    check("t3: B a FAST hand 1/3 -> STOP", t3_gate(t3s([FI, SS, SS], [SS] * 3))["reading"], "STOP: B FORMATION")
    check("t3: C0 FAST/FAST 2/3 -> STOP", t3_gate(t3s([FF] * 3, [FF, FF, SS]))["reading"], "STOP: C0 FORMS")
    check("t3: a missing read -> STOP", t3_gate(t3s([FF] * 3, [SS] * 3)[:5])["stop"], True)
    FS = _pt(F, S, 0.22, 0.09)
    check("t3: B one FAST hand, second SLOW in 2/3 (INC0-like) -> STOP", t3_gate(t3s([FS, FS, FI], [SS] * 3))["reading"],
          "STOP: SECOND HAND")
    check("t3: an NA hand -> STOP (incomplete)", t3_gate(t3s([FF, FF, _pt(F, "NA")], [SS] * 3))["reading"], "INCOMPLETE")

    def tails(bh, ch, bsuite=None, csuite=None, b6=None, hidden=True, dose=True):
        out = t3s([FF] * 3, [SS] * 3)
        for i, s in enumerate(SEEDS):
            out.append(_run(f"P1B6__B_tail_s{s}", "tail", "B", s, bh[i], c6=(b6 or bh)[i],
                            suite=bsuite[i] if bsuite else _suite(0.8), dose_ok=dose, hidden=hidden))
            out.append(_run(f"P1B6__C0_tail_s{s}", "tail", "C0", s, ch[i],
                            suite=csuite[i] if csuite else _suite(0.7, jit=0.10), hidden=hidden))
        return out
    check("final: B 3/3, C0 0/3, categories hold -> PASS", evaluate(tails([FF] * 3, [SS] * 3))["gate"], "PASS")
    check("final: clock not hidden in an eval -> INVALID EVAL", evaluate(tails([FF] * 3, [SS] * 3, hidden=False))["gate"],
          "INVALID EVAL")
    check("final: B FAST/FAST at c6 3/3 but c6x 1/3 -> CLOCK-CUT LOSS",
          evaluate(tails([FF, SS, SS], [SS] * 3, b6=[FF] * 3))["gate"], "CLOCK-CUT LOSS")
    check("final: B 1/3 (also 1/3 at c6) -> FAIL", evaluate(tails([FF, SS, SS], [SS] * 3))["gate"], "FAIL")
    check("final: hands FAST/FAST at c6x but the dose lost in 2/3 -> FAIL, not CLOCK-CUT LOSS",
          evaluate(tails([FF] * 3, [SS] * 3, b6=[FF] * 3, dose=False))["gate"], "FAIL")
    low = [_suite(0.5)] * 3
    check("final: B forms but categories clearly inferior -> COST", evaluate(tails([FF] * 3, [SS] * 3, bsuite=low))["gate"], "COST")
    mt = [dict(_suite(0.8), move_turn_survival=0.1, move_turn_ratio=0.1)] * 3
    check("final: move_turn far below C0 is reported, not gated -> PASS", evaluate(tails([FF] * 3, [SS] * 3, bsuite=mt))["gate"], "PASS")
    yp = [dict(_suite(0.8), turn_yaw_progress=0.70 - 0.02)] * 3
    check("final: turn yaw_progress 0.02 short (margin 0.01) -> not PASS", evaluate(tails([FF] * 3, [SS] * 3, bsuite=yp))["gate"] != "PASS", True)
    check("final: an NA hand at c6x -> PENDING", evaluate(tails([FF, FF, _pt(F, "NA")], [SS] * 3))["gate"], "PENDING")
    mix = [_suite(v) for v in (0.62, 0.75, 0.63)]
    check("final: a category short but not clearly inferior -> PARTIAL",
          evaluate(tails([FF] * 3, [SS] * 3, bsuite=mix))["gate"], "PARTIAL")
    check("final: C0 FAST/FAST 2/3 -> PARTIAL", evaluate(tails([FF] * 3, [FF, FF, SS]))["gate"], "PARTIAL")
    check("final: a missing dose on a FAST/FAST B -> PENDING", evaluate(tails([FF] * 3, [SS] * 3, dose=None))["gate"], "PENDING")
    r = evaluate(tails([FF] * 3, [SS] * 3))
    check("smoothness flag: B jitter 0.15 vs C0 0.10 (> 1.5x? no: 1.5x exactly) -> none", r["smoothness_flag"], [])
    rj = evaluate(tails([FF] * 3, [SS] * 3, bsuite=[_suite(0.8, jit=0.2)] * 3))
    check("smoothness flag fires at 2x jitter", bool(rj["smoothness_flag"]), True)
    w = {"suite": _suite(0.75), "lr": FF}
    rw = evaluate(tails([FF] * 3, [SS] * 3), w)
    check("adoption candidates: B (0.8 everywhere) beats W (0.75) on side and no category 0.05 below",
          len(rw["adoption_candidates"]), 3)
    rw2 = evaluate(tails([FF] * 3, [SS] * 3), {"suite": _suite(0.9), "lr": FF})
    check("no adoption candidate when W is better", rw2["adoption_candidates"], [])
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", type=Path, default=OUT / "arms_p1")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--no-tb", action="store_true")
    ap.add_argument("--t3-gate", action="store_true")
    ap.add_argument("--c6", default=None, metavar="TAIL_REF")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    if a.c6:
        ck, st = c6_paths(a.c6)
        if not ck or not st:
            print(f"no c6 head/stamp for {a.c6}: {ck} {st}", file=sys.stderr)
            return 2
        print(ck, st)
        return 0
    runs, w = load_all(a.arms, with_tb=not (a.no_tb or a.t3_gate))
    if a.t3_gate:
        g = t3_gate(runs)
        print(json.dumps(g))
        return 4 if g["stop"] else 0
    res = evaluate(runs, w)
    md = markdown(res)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=str))
    if a.md:
        a.md.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
