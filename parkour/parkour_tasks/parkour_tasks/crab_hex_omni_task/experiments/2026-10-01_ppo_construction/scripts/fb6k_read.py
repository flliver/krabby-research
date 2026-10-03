#!/usr/bin/env python3
"""FB6k + INC0 reader (CHARTER 11): the pre-registered reads, the INC0 attribution gate, the extension decision
and the FB6k gate.

CPU only, read-only. Records: $OUT/arms_fb6k/<ref>.record.json (run_arm.sh with ARMS_DIR=$OUT/arms_fb6k; the
6000-iteration runs with EXTRA_LR_EVALS="3000 4500", the x9k forks with RESUME_READ=1 EXTRA_LR_EVALS="6100 7500").
Hand classes are verdict.py's (feet AND along-command, fail-closed NA) through x6k_read.read_point; the dose window
is the run's own last 100 logged iterations (x6k_read.tail_reads). The suite read is the driver ``suite__omni``
stamp of the run's head: ``aggregate.suite.by_category.<cat>.ratio.median`` and ``survival_by_hold``.

Rules (CHARTER 11.4 / 11.5):
  INC0 gate     INC0 (C_lr C0 + action_rate 0, 6000 its, seeds 11-13) FAST/FAST at its head in >= 2 of 3 -> STOP
                (the flip belongs to action_rate: stop and ask before FB6k); 1/3 -> note; 0/3 -> action_rate 0 alone
                does not flip C_lr by 6000 continuous iterations. Fewer than 3 INC0 reads, or a dumped action_rate
                weight that is not 0, -> STOP (stop and ask).
  extension     (pre-launch review 2026-10-03) every 6000-iteration B run that is not FAST/FAST at its head and has
                no DEAD hand is forked +3000 (x9k) -- x6k showed flat stretches before the second hand crosses, so a
                rise trigger would miss them; the 4500 -> 5999 rise is reported, not a trigger. The C0 run of the
                SAME seed is forked whenever B's is (equal budget per seed), or if a C0 hand rose >= 0.03 from
                model_4500 (and it is not FAST/FAST). A fork's final read is its head (8998); a FAST-at-5999 hand
                below FAST and -0.03 at model_6100 is a RESUME ARTIFACT (stop and ask).
  tripwire      threshold max(0.15, ref + 0.10), ref = median fall share of the FB6k C0 final reads (runs above 0.25
                excluded, CHARTER 4.4); applies to B.
  gate          FAIL if B (FAST/FAST, untripped, dose ok, at its final read) <= 1/3, or fwd or back is CLEARLY
                inferior (B's seed-mean suite ratio < C0's - 0.03 AND B's best seed < C0's worst seed, at model_5999);
                PASS if B >= 2/3, C0 0/3, no B run tripped, and fwd and back hold (B mean >= C0 mean - 0.03);
                otherwise PARTIAL. PENDING while ANY needed read is missing (a head, model_4500 for the C0 rise,
                model_6100 for the artifact rule, a B dose read on an otherwise counted run, a suite read) or a
                triggered extension has not run. A DEAD/DIVERGED-killed seed is a valid not-FAST run and is left out
                of the fwd/back comparison (n reported).

    fb6k_read.py [--arms D] [--json OUT] [--md OUT] [--no-tb]
    fb6k_read.py --inc0-gate          exit 0 continue / 4 stop (prints why)
    fb6k_read.py --extensions         prints the x9k fork references the rule triggers
    fb6k_read.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/ppo_construction")
_spec = importlib.util.spec_from_file_location("_fb6k_x6k", HERE / "x6k_read.py")
X = importlib.util.module_from_spec(_spec)
sys.modules["_fb6k_x6k"] = X
_spec.loader.exec_module(X)

FAST = "FAST"
MATERIAL = 0.03
TRIP_FLOOR, TRIP_MARGIN, TRIP_COLLAPSE = 0.15, 0.10, 0.25
SEEDS_FB, SEEDS_INC0 = (16, 17, 18), (11, 12, 13)
CATS = ("fwd", "back", "side", "diag")
HANDS = ("l", "r")


def ff(pt) -> bool:
    return bool(pt) and pt["hands"]["l"] == FAST and pt["hands"]["r"] == FAST


def suite_reads(stamp: str | None) -> dict:
    out = {}
    if not stamp:
        return out
    try:
        a = json.load(open(Path(stamp) / "scenario_metrics.json"))["aggregate"]["suite"]
    except (OSError, ValueError, KeyError):
        return out
    by = a.get("by_category") or {}
    surv = a.get("survival_by_hold") or {}
    for c in CATS:
        r = (by.get(c) or {}).get("ratio") or {}
        out[c] = {"ratio": r.get("median"), "survival": surv.get(c) if c != "side" else
                  (None if surv.get("lat_l") is None else (surv["lat_l"] + surv.get("lat_r", surv["lat_l"])) / 2)}
    return out


def load_run(rec: dict, with_tb: bool = True) -> dict:
    ref = rec["ref"]
    chain = rec.get("chain") or ""
    arm = "INC0" if "INC0" in chain else ("B" if "__B" in chain else "C0")
    fork = "_x9k" in ref
    pts = {}
    for n, e in (rec.get("extra_lr") or {}).items():
        pts[n] = X.read_point(e.get("class_json"), e.get("along_json"), f"{ref}_m{n}")
    pts["head"] = X.read_point(rec.get("lr_class_json"), rec.get("lr_along_json"), ref)
    odd = ((X._lj(rec.get("odd_rms_json")) or {}).get(ref) or {})
    dump = None
    if arm == "INC0" and rec.get("run_dir"):
        dump = X.V.dump_checks(Path(rec["run_dir"]), {"KRABBY_ACTION_RATE_W": "0"})
    return {"ref": ref, "arm": arm, "seed": int(rec.get("seed") or 0), "fork": fork, "killed": rec.get("killed"),
            "killed_kind": rec.get("killed_kind"),
            "points": pts, "suite": suite_reads(rec.get("driver_stamp")),
            "q": X.V.q_reads(rec.get("lr_stamp")), "odd": {h: odd.get(f"rod_odd_rms_{h}") for h in HANDS},
            "dump": dump, "tail": X.tail_reads(rec.get("run_dir")) if with_tb else {}}


def rise(run, h, frm="4500", to="head"):
    a, b = run["points"].get(frm), run["points"].get(to)
    if not a or not b or a["feet"][h] is None or b["feet"][h] is None:
        return None
    return b["feet"][h] - a["feet"][h]


def extension_wanted(run: dict, b_forked_same_seed: bool = False) -> bool | None:
    """CHARTER 11.4 on a 6000-iteration run (None = a read the rule needs is missing)."""
    head = run["points"].get("head")
    if not head:
        return None
    if ff(head) or any(head["hands"][h] == "DEAD" for h in HANDS):
        return False
    if run["arm"] == "B":
        return True
    if run["arm"] == "C0":
        if b_forked_same_seed:
            return True
        if run["points"].get("4500") is None:
            return None
        return any((rise(run, h) or 0) >= MATERIAL for h in HANDS)
    return False


def fork_artifact(fork_run: dict, base_run: dict) -> bool | None:
    p6100, base = fork_run["points"].get("6100"), base_run["points"].get("head")
    if not p6100 or not base:
        return None
    return any(base["hands"][h] == FAST and p6100["hands"][h] != FAST and p6100["feet"][h] is not None
               and base["feet"][h] is not None and p6100["feet"][h] - base["feet"][h] <= -MATERIAL for h in HANDS)


def inc0_rule(runs: list[dict]) -> dict:
    inc = [r for r in runs if r["arm"] == "INC0" and not r["fork"] and r["seed"] in SEEDS_INC0]
    bad_dump = [r["ref"] for r in inc if not ((r.get("dump") or {}).get("action_rate_w") or {}).get("ok")]
    readable = [r for r in inc if not r.get("killed") and r["points"].get("head")]
    k = sum(ff(r["points"]["head"]) for r in readable)
    out = {"n": len(readable), "k_fast_fast": k,
           "per_seed": {r["seed"]: [r["points"]["head"]["hands"][h] for h in HANDS] for r in readable}}
    if len(readable) < 3:
        out.update(stop=True, reading="INCOMPLETE", why=f"{len(readable)}/3 INC0 reads: stop and ask")
    elif bad_dump:
        out.update(stop=True, reading="INCOMPLETE",
                   why=f"dumped action_rate weight is not 0 (or unreadable) in {bad_dump}: INC0 not wired -- stop and ask")
    elif k >= 2:
        out.update(stop=True, reading="ACTION_RATE",
                   why=f"INC0 FAST/FAST {k}/3 >= 2/3: the flip belongs to action_rate 0, not the construction -- stop and ask")
    elif k == 1:
        out.update(stop=False, reading="NOTE", why="INC0 FAST/FAST 1/3: noted; B is labelled 'construction + action_rate 0'")
    else:
        out.update(stop=False, reading="ACTION_RATE ALONE INSUFFICIENT",
                   why="INC0 FAST/FAST 0/3: action_rate 0 alone does not flip C_lr by 6000 continuous iterations; the "
                       "construction attribution is read together with FB6k, whose from-scratch B runs test the "
                       "fork == continuous premise of the x6k comparison")
    return out


def evaluate(runs: list[dict]) -> dict:
    base = {(r["arm"], r["seed"]): r for r in runs if not r["fork"] and r["arm"] in ("B", "C0") and r["seed"] in SEEDS_FB}
    forks = {(r["arm"], r["seed"]): r for r in runs if r["fork"]}
    res = {"inc0": inc0_rule(runs), "seeds": {}, "extensions": [], "artifacts": [], "notes": []}
    pending = []
    final = {}
    b_forked = set()
    for arm in ("B", "C0"):
        for s in SEEDS_FB:
            r = base.get((arm, s))
            if r is not None and r.get("killed") and r.get("killed_kind") in ("dead", "diverged"):
                # CHARTER 4.2 / 11.4: a DEAD or DIVERGED kill is a VALID not-FAST run (no extension)
                dead = {"hands": {"l": "DEAD", "r": "DEAD"}, "feet": {"l": None, "r": None}, "fall": None}
                r = dict(r, points={"head": dead})
                final[(arm, s)] = (r, r)
                continue
            if r is not None and r.get("killed") and r.get("killed_kind") == "dose":
                res["notes"].append(f"{r['ref']}: dose kill -- CHARTER 4.3 stop (check wiring; wiring right = stop and ask)")
            if r is None or r.get("killed") or not r["points"].get("head"):
                pending.append(f"{arm} s{s}" + (" (killed)" if r and r.get("killed") else ""))
                continue
            want = extension_wanted(r, b_forked_same_seed=(arm == "C0" and s in b_forked))
            if want is None:
                pending.append(f"{arm} s{s} model_4500 read (the C0 rise rule)")
                continue
            f = forks.get((arm, s))
            if want:
                if arm == "B":
                    b_forked.add(s)
                res["extensions"].append(f"FB6k__{arm}_x9k_s{s}")
                if f is None or f.get("killed") or not f["points"].get("head"):
                    pending.append(f"{arm} s{s} extension")
                    continue
                art = fork_artifact(f, r)
                if art is None:
                    pending.append(f"{arm} s{s} fork model_6100 read (the resume-artifact rule)")
                    continue
                if art:
                    res["artifacts"].append(f["ref"])
                final[(arm, s)] = (f, r)
            else:
                final[(arm, s)] = (r, r)
    # tripwire: the C0 final reads' fall shares
    c0f = [final[("C0", s)][0]["points"]["head"]["fall"] for s in SEEDS_FB if ("C0", s) in final]
    c0f = [x for x in c0f if x is not None and x <= TRIP_COLLAPSE]
    ref = statistics.median(c0f) if c0f else None
    thr = max(TRIP_FLOOR, ref + TRIP_MARGIN) if ref is not None else TRIP_FLOOR
    res["tripwire"] = {"reference": ref, "threshold": thr}
    for (arm, s), (fin, r6k) in final.items():
        head = fin["points"]["head"]
        tripped = arm == "B" and head["fall"] is not None and head["fall"] > thr
        dose_ok = fin["tail"].get("dose_ok") if arm == "B" else None
        if arm == "B" and ff(head) and not tripped and dose_ok is None:
            pending.append(f"B s{s} dose read")
        counted = ff(head) and not tripped and (arm != "B" or dose_ok is True)
        res["seeds"][f"{arm} s{s}"] = {
            "dead_kill": fin.get("killed_kind") in ("dead", "diverged"), "q": fin.get("q"), "odd": fin.get("odd"),
            "final": fin["ref"], "extended": fin is not r6k, "hands": head["hands"], "feet": head["feet"],
            "fall": head["fall"], "tripped": tripped, "dose": {k: fin["tail"].get(k) for k in ("mu_med_cam", "mu_med_rod", "dose_ok")},
            "fast_fast_counted": counted, "suite_5999": r6k["suite"],
            "trajectory": {n: (p["hands"], p["feet"]) for n, p in fin["points"].items() if p} if fin is r6k else
            {**{n: (p["hands"], p["feet"]) for n, p in r6k["points"].items() if p},
             **{f"fork {n}": (p["hands"], p["feet"]) for n, p in fin["points"].items() if p}}}
    kB = sum(v["fast_fast_counted"] for k, v in res["seeds"].items() if k.startswith("B "))
    kC = sum(ff({"hands": v["hands"]}) for k, v in res["seeds"].items() if k.startswith("C0 "))
    anyB_trip = any(v["tripped"] for k, v in res["seeds"].items() if k.startswith("B "))
    fb = {}
    for c in ("fwd", "back"):
        cols = {}
        for arm in ("B", "C0"):
            live = [k for k in (f"{arm} s{s}" for s in SEEDS_FB) if k in res["seeds"] and not res["seeds"][k]["dead_kill"]]
            vals = [res["seeds"][k]["suite_5999"].get(c, {}).get("ratio") for k in live]
            cols[arm] = (vals, len(live))
        (b, nb), (z, nz) = cols["B"], cols["C0"]
        if nb and nz and len(b) == nb and len(z) == nz and None not in b and None not in z:
            mb, mz = sum(b) / nb, sum(z) / nz
            fb[c] = {"B_mean": mb, "C0_mean": mz, "holds": mb >= mz - MATERIAL, "n_B": nb, "n_C0": nz,
                     "clearly_inferior": mb < mz - MATERIAL and max(b) < min(z), "B": b, "C0": z}
        else:
            fb[c] = None
    res["fwd_back"] = fb
    res["counts"] = {"B_fast_fast": kB, "C0_fast_fast": kC, "B_tripped": anyB_trip}
    if res["artifacts"]:
        res["gate"], res["why"] = "RESUME ARTIFACT", f"a FAST hand dropped at model_6100 in {res['artifacts']}: stop and ask"
    elif pending or any(v is None for v in fb.values()):
        res["gate"], res["why"] = "PENDING", "missing: " + (", ".join(pending) or "suite reads")
    elif kB <= 1 or any(v["clearly_inferior"] for v in fb.values()):
        res["gate"] = "FAIL"
        res["why"] = (f"B FAST/FAST {kB}/3" if kB <= 1 else "") + \
            ("; fwd/back clearly inferior: " + ", ".join(c for c, v in fb.items() if v["clearly_inferior"])
             if any(v["clearly_inferior"] for v in fb.values()) else "")
    elif kB >= 2 and kC == 0 and not anyB_trip and all(v["holds"] for v in fb.values()):
        res["gate"], res["why"] = "PASS", f"B FAST/FAST {kB}/3, C0 0/3, no B trip, fwd and back hold"
    else:
        why = []
        if kC:
            why.append(f"C0 FAST/FAST {kC}/3 (the FB pool lets C0 form; the construction effect is unresolved)")
        if anyB_trip:
            why.append("a B run tripped")
        why += [f"{c} B mean {v['B_mean']:.3f} < C0 mean {v['C0_mean']:.3f} - 0.03 (not clearly inferior)"
                for c, v in fb.items() if not v["holds"]]
        res["gate"], res["why"] = "PARTIAL", f"B FAST/FAST {kB}/3; " + "; ".join(why)
    return res


def load_all(arms: Path, with_tb: bool = True) -> list[dict]:
    return [load_run(json.load(open(p)), with_tb) for p in sorted(arms.glob("*.record.json"))]


def _f(x, fmt="{:.3f}"):
    return "--" if x is None else fmt.format(x)


def markdown(res: dict) -> str:
    L = ["# FB6k + INC0 -- the pre-registered reading (CHARTER 11)", "",
         f"**INC0 gate: {res['inc0']['reading']}** -- {res['inc0']['why']}", ""]
    if res["inc0"].get("per_seed"):
        L += [f"- INC0 s{s}: {'/'.join(h)}" for s, h in sorted(res["inc0"]["per_seed"].items())] + [""]
    L += ["| seed | final | extended | hands L / R (feet) | fall | dose cam/rod | FAST/FAST counted | "
          "suite@5999 fwd / back / side / diag ratio |", "|" + "---|" * 8]
    for k, v in sorted(res["seeds"].items()):
        s = v["suite_5999"]
        L.append(f"| {k} | {v['final']} | {v['extended']} | {v['hands']['l'][:4]} {_f(v['feet']['l'])} / "
                 f"{v['hands']['r'][:4]} {_f(v['feet']['r'])} | {_f(v['fall'])}{' TRIPPED' if v['tripped'] else ''} | "
                 f"{_f(v['dose']['mu_med_cam'], '{:.2f}')}/{_f(v['dose']['mu_med_rod'], '{:.2f}')} | {v['fast_fast_counted']} | "
                 + " / ".join(_f((s.get(c) or {}).get("ratio")) for c in CATS) + " |")
    L += ["", f"Tripwire: reference {_f(res['tripwire']['reference'])}, threshold {_f(res['tripwire']['threshold'])}",
          f"Extensions the rule triggers: {res['extensions'] or 'none'}", ""]
    for c, v in (res.get("fwd_back") or {}).items():
        if v:
            L.append(f"- {c}: B mean {v['B_mean']:.3f} vs C0 mean {v['C0_mean']:.3f} -> holds {v['holds']}, "
                     f"clearly inferior {v['clearly_inferior']}")
    L += ["", f"**FB6k gate: {res['gate']}** -- {res['why']}", ""]
    for k, v in sorted(res["seeds"].items()):
        L.append(f"- {k} trajectory: " + "; ".join(f"{n}: {h['l'][:4]}/{h['r'][:4]} {_f(f['l'])}/{_f(f['r'])}"
                                                    for n, (h, f) in v["trajectory"].items()))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ self-test
def _pt(lc, rc, lf, rf, fall=0.01):
    return {"hands": {"l": lc, "r": rc}, "feet": {"l": lf, "r": rf}, "fall": fall}


def _run(ref, arm, seed, pts, suite=None, dose_ok=True, fork=False, killed=None, dump_ok=True):
    return {"ref": ref, "arm": arm, "seed": seed, "fork": fork, "killed": killed, "points": pts,
            "suite": {c: {"ratio": 0.5, "survival": 0.9} for c in CATS} if suite is None else suite,
            "tail": {"dose_ok": dose_ok}, "dump": {"action_rate_w": {"ok": dump_ok}}}


def _self_test() -> int:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}")

    F, I, S = FAST, "INTERMEDIATE", "SLOW"
    inc = [_run(f"C_lr6k__INC0_s{s}", "INC0", s, {"head": _pt(S, S, 0.08, 0.08)}) for s in SEEDS_INC0]
    check("INC0 0/3 -> continue", (inc0_rule(inc)["stop"], inc0_rule(inc)["reading"]),
          (False, "ACTION_RATE ALONE INSUFFICIENT"))
    bad = inc[:2] + [_run("C_lr6k__INC0_s13", "INC0", 13, {"head": _pt(S, S, 0.08, 0.08)}, dump_ok=False)]
    check("INC0 whose dumped action_rate is not 0 -> stop", (inc0_rule(bad)["stop"], inc0_rule(bad)["reading"]),
          (True, "INCOMPLETE"))
    inc2 = [_run(f"C_lr6k__INC0_s{s}", "INC0", s, {"head": _pt(F, F, 0.2, 0.2)}) for s in (11, 12)] + inc[2:]
    check("INC0 2/3 -> stop (action_rate)", (inc0_rule(inc2)["stop"], inc0_rule(inc2)["reading"]), (True, "ACTION_RATE"))
    check("INC0 with 2 reads -> stop (incomplete)", inc0_rule(inc[:2])["stop"], True)

    one = _run("FB6k__B_s16", "B", 16, {"4500": _pt(F, I, 0.22, 0.13), "head": _pt(F, I, 0.23, 0.165)})
    check("B one-FAST, second hand +0.035 since 4500 -> extend", extension_wanted(one), True)
    flat = _run("FB6k__B_s16", "B", 16, {"4500": _pt(F, I, 0.22, 0.15), "head": _pt(F, I, 0.23, 0.16)})
    check("B one-FAST in a flat stretch (+0.01) -> still extended (review: x6k's s15 lull)", extension_wanted(flat), True)
    check("B with a DEAD hand -> no extension", extension_wanted(_run("x", "B", 16, {"head": _pt(F, "DEAD", .2, .02)})), False)
    c0flat = _run("FB6k__C0_s16", "C0", 16, {"4500": _pt(S, S, 0.08, 0.08), "head": _pt(S, S, 0.08, 0.09)})
    check("C0 flat -> no extension unless B of its seed is forked", (extension_wanted(c0flat),
          extension_wanted(c0flat, b_forked_same_seed=True)), (False, True))
    check("B FAST/FAST -> no extension", extension_wanted(_run("x", "B", 16, {"4500": _pt(F, I, .2, .1), "head": _pt(F, F, .2, .2)})), False)
    c0m = _run("FB6k__C0_s16", "C0", 16, {"4500": _pt(S, S, 0.08, 0.08), "head": _pt(S, I, 0.08, 0.14)})
    check("C0 with a hand +0.06 -> extend", extension_wanted(c0m), True)

    def fb(Bheads, C0heads, bsuite=None, csuite=None, forks=()):
        runs = []
        for s, h in zip(SEEDS_FB, Bheads):
            runs.append(_run(f"FB6k__B_s{s}", "B", s, {"4500": h, "head": h}, suite=bsuite and bsuite[s - 16]))
        for s, h in zip(SEEDS_FB, C0heads):
            runs.append(_run(f"FB6k__C0_s{s}", "C0", s, {"4500": h, "head": h}, suite=csuite and csuite[s - 16]))
        return evaluate(runs + list(forks) + inc)

    FFp, SSp = _pt(F, F, 0.24, 0.23), _pt(S, S, 0.08, 0.08)

    def fk(arm, seeds, head):        # x9k forks (the rule forks every non-FAST/FAST B and the same seed's C0)
        return [_run(f"FB6k__{arm}_x9k_s{s}", arm, s, {"6100": head, "head": head}, fork=True) for s in seeds]
    check("B 3/3, C0 0/3 -> PASS", fb([FFp] * 3, [SSp] * 3)["gate"], "PASS")
    check("B 1/3 (forks stay SLOW) -> FAIL",
          fb([FFp, SSp, SSp], [SSp] * 3, forks=fk("B", (17, 18), SSp) + fk("C0", (17, 18), SSp))["gate"], "FAIL")
    check("B 1/3 before its forks ran -> PENDING", fb([FFp, SSp, SSp], [SSp] * 3)["gate"], "PENDING")
    check("B 2/3 but C0 1/3 -> PARTIAL",
          fb([FFp, FFp, SSp], [FFp, SSp, SSp], forks=fk("B", (18,), SSp) + fk("C0", (18,), SSp))["gate"], "PARTIAL")
    low = [{c: {"ratio": 0.40} for c in CATS}] * 3
    high = [{c: {"ratio": 0.50} for c in CATS}] * 3
    check("fwd/back clearly inferior -> FAIL", fb([FFp] * 3, [SSp] * 3, bsuite=low, csuite=high)["gate"], "FAIL")
    mix = [{c: {"ratio": v} for c in CATS} for v in (0.43, 0.52, 0.44)]
    check("B mean 0.463 < 0.50 - 0.03 but best seed above C0's worst -> PARTIAL",
          fb([FFp] * 3, [SSp] * 3, bsuite=mix, csuite=high)["gate"], "PARTIAL")
    ext_head = _pt(F, I, 0.23, 0.165)
    runs = [_run(f"FB6k__B_s{s}", "B", s, {"4500": _pt(F, I, 0.22, 0.13), "head": ext_head}) for s in SEEDS_FB] + \
        [_run(f"FB6k__C0_s{s}", "C0", s, {"4500": SSp, "head": SSp}) for s in SEEDS_FB]
    r = evaluate(runs + inc)
    check("triggered extensions (3 B + their 3 C0) are PENDING until the forks are read",
          (r["gate"], len(r["extensions"])), ("PENDING", 6))
    forks = [_run(f"FB6k__B_x9k_s{s}", "B", s, {"6100": _pt(F, I, 0.23, 0.17), "head": FFp}, fork=True) for s in SEEDS_FB] + \
        fk("C0", SEEDS_FB, SSp)
    check("forks FAST/FAST (C0 forks SLOW) -> PASS on the fork heads", evaluate(runs + forks + inc)["gate"], "PASS")
    art = [_run("FB6k__B_x9k_s16", "B", 16, {"6100": _pt(I, I, 0.18, 0.17), "head": FFp}, fork=True)] + forks[1:]
    check("a first hand dropped at 6100 -> RESUME ARTIFACT", evaluate(runs + art + inc)["gate"], "RESUME ARTIFACT")
    nodose = [_run(f"FB6k__B_s{s}", "B", s, {"4500": FFp, "head": FFp}, dose_ok=False) for s in SEEDS_FB] + \
        [_run(f"FB6k__C0_s{s}", "C0", s, {"4500": SSp, "head": SSp}) for s in SEEDS_FB]
    check("B FAST/FAST without dose does not count -> FAIL", evaluate(nodose + inc)["gate"], "FAIL")
    tr = [_run(f"FB6k__B_s{s}", "B", s, {"4500": FFp, "head": _pt(F, F, 0.24, 0.23, fall=0.3)}) for s in SEEDS_FB] + \
        [_run(f"FB6k__C0_s{s}", "C0", s, {"4500": SSp, "head": SSp}) for s in SEEDS_FB]
    check("tripped B runs do not count -> FAIL", evaluate(tr + inc)["gate"], "FAIL")
    dk = [_run("FB6k__B_s16", "B", 16, {}, killed="dead", suite={}),
          *[_run(f"FB6k__B_s{s}", "B", s, {"4500": FFp, "head": FFp}) for s in (17, 18)]]
    dk[0]["killed_kind"] = "dead"
    dk += [_run(f"FB6k__C0_s{s}", "C0", s, {"4500": SSp, "head": SSp}) for s in SEEDS_FB]
    check("a DEAD-killed B run (no suite stamp) counts as not FAST (2/3 -> PASS, not PENDING)", evaluate(dk + inc)["gate"], "PASS")
    dk1 = [dk[0], _run("FB6k__B_s17", "B", 17, {"4500": FFp, "head": FFp}),
           _run("FB6k__B_s18", "B", 18, {"4500": SSp, "head": _pt("DEAD", "DEAD", 0.01, 0.01)})] + dk[3:]
    check("B 1/3 with a DEAD kill -> FAIL (not PENDING)", evaluate(dk1 + inc)["gate"], "FAIL")
    nod = [_run(f"FB6k__B_s{s}", "B", s, {"4500": FFp, "head": FFp}, dose_ok=None) for s in SEEDS_FB] + dk[3:]
    check("a missing dose read on a FAST/FAST B -> PENDING", evaluate(nod + inc)["gate"], "PENDING")
    no6100 = [_run(f"FB6k__B_x9k_s{s}", "B", s, {"head": FFp}, fork=True) for s in SEEDS_FB] + fk("C0", SEEDS_FB, SSp)
    check("a fork without its model_6100 read -> PENDING", evaluate(runs + no6100 + inc)["gate"], "PENDING")
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arms", type=Path, default=OUT / "arms_fb6k")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--no-tb", action="store_true")
    ap.add_argument("--inc0-gate", action="store_true")
    ap.add_argument("--extensions", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    runs = load_all(a.arms, with_tb=not (a.no_tb or a.inc0_gate or a.extensions))
    if a.inc0_gate:
        g = inc0_rule(runs)
        print(json.dumps(g))
        return 4 if g["stop"] else 0
    res = evaluate(runs)
    if a.extensions:
        print(" ".join(res["extensions"]))
        return 0
    md = markdown(res)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=str))
    if a.md:
        a.md.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
