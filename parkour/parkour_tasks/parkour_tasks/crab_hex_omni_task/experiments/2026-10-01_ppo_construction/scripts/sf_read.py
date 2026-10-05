#!/usr/bin/env python3
"""Turn-band diagnostics, t3x and side-first reader (CHARTER 13): the pre-registered reads and rules of the user's
"Run 1-3" (p1/next_options.md steps 1-3).

CPU only, read-only. Records live in $OUT/arms_p1 (run_arm.sh ARMS_DIR), eval stamps in $OUT/lr/<name>/<row>/seed001/<ts>.
Classes are verdict.py's (feet AND along, fail-closed NA) via x6k_read.read_point; the final side-first gate is CHARTER
12.5's, applied by p1_read.evaluate to the side-first tails.

Reads and rules:
  turn band  (13.2) in-place turns inside the trained facing band (turn: +-0.5 rad, the read) and at its edge (turn_mid:
             +-1.0 rad, descriptive), `scenarios_turn_band.yaml` row turn_band__omni, on: the six P1-B6 t3 heads, the three
             FB6k B heads (untrained-turn reference), W, the t3x heads, the side-first c6 and c6x heads. Sampled-action
             suite__omni runs on the B t3 heads + C0 t3 s16 are DIAGNOSTIC ONLY (never gated). Metrics:
             by_category.<turn|turn_mid>.facing_closed / yaw_progress (medians), survival_by_hold.
  t3x rule   (13.3) the P1-B6 B t3 heads forked +3000 (head model_8998). "In-band settles" is in RADIANS: the B t3x heads'
             in-band turn heading_final (median final |heading error|) seed-mean <= the C0 t3 heads' + 0.05 rad (the
             facing P-control's dead band is 0.133 rad, so a relative facing_closed margin would be decided by resting error).
             Readings: INCOMPLETE (a head, dose, fall, band or model_6100 read missing / NA) -> notify; RESUME ARTIFACT (a
             FAST-at-5999 hand below FAST and -0.03 at model_6100) -> stop and ask; TAILS CANDIDATE (B counted FAST/FAST,
             untripped at 0.15, dose ok, in >= 2/3 AND in-band settles) -> stop and ask; FORMED, TURN NOT IN-BAND (counted
             >= 2/3, in-band does not settle) -> stop and ask; PLATEAU (counted <= 1/3). Never stops the side-first block.
  c4 gate    (13.4) side-first c4 heads (the FB6k B heads after the first turn-including stage): STOP and ask if the
             stroke is lost -- neither hand >= INTERMEDIATE -- in >= 2/3 seeds, or a c4 read is missing / NA.
  C0 trigger (13.4) the C0 side-first tails run only if B's c6x heads are FAST/FAST by hands in >= 2/3 (only then does
             the COST comparison need a matched lineage).
  final      (13.4) INVALID EVAL (a c6x stamp without KRABBY_OMNI_CLOCK_HIDE=1) and PENDING (a c6 / c6x read missing or NA)
             come first on both paths. Without the C0 tails: CLOCK-CUT LOSS if B FAST/FAST (hands) at c6 >= 2/3 and at c6x
             <= 1/3, else FAIL. With them: CHARTER 12.5 exactly (p1_read.evaluate), W as reference; a COST whose only clearly
             inferior gated categories are turn ones, with B's c6x in-band heading_final within +0.05 rad of the C0 lineage,
             reads "COST (turn, out-of-band)" and names the facing-band c6x variant as the pre-registered next lever.

    sf_read.py [--json OUT] [--md OUT] [--no-tb]
    sf_read.py --t3x-rule | --c4-gate | --c0-trigger      (exit 0 / 4 stop for --c4-gate; 0 run / 3 skip for --c0-trigger)
    sf_read.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("_sf_p1", HERE / "p1_read.py")
P1 = importlib.util.module_from_spec(_spec)
sys.modules["_sf_p1"] = P1
_spec.loader.exec_module(P1)
X = P1.X
OUT = P1.OUT
ARMS = OUT / "arms_p1"
FAST = "FAST"
SEEDS = (16, 17, 18)
HANDS = ("l", "r")
BAND_ROW = "turn_band__omni"
TRIP = 0.15
BAND_RAD = 0.05            # in-band "settles": B heading_final seed-mean <= C0's + 0.05 rad


def newest_stamp(name: str, row: str) -> str | None:
    hits = [d for d in sorted((OUT / "lr" / name / row / "seed001").glob("*/")) if (d / "complete.json").is_file()]
    return str(hits[-1]) if hits else None


def band_read(stamp: str | None) -> dict | None:
    if not stamp:
        return None
    try:
        a = json.load(open(Path(stamp) / "scenario_metrics.json"))["aggregate"]["suite"]
    except (OSError, ValueError, KeyError):
        return None
    by, surv = a.get("by_category") or {}, a.get("survival_by_hold") or {}
    out = {}
    for cat in ("turn", "turn_mid"):
        out[cat] = {"facing_closed": P1._med(by, cat, "facing_closed"), "yaw_progress": P1._med(by, cat, "yaw_progress"),
                    "heading_final": P1._med(by, cat, "heading_final")}
    out["survival"] = dict(surv)
    return out


def head_point(rec: dict, ref: str, n: str | None = None):
    if n is None:
        return X.read_point(rec.get("lr_class_json"), rec.get("lr_along_json"), ref)
    e = (rec.get("extra_lr") or {}).get(n) or {}
    return X.read_point(e.get("class_json"), e.get("along_json"), f"{ref}_m{n}")


def _rec(ref: str, arms: Path = ARMS) -> dict | None:
    p = arms / f"{ref}.record.json"
    return json.load(open(p)) if p.is_file() else None


def _lost(pt) -> bool:
    return bool(pt) and not any(pt["hands"][h] in (FAST, "INTERMEDIATE") for h in HANDS)


# ------------------------------------------------------------------------------------------------ rules (pure)
def t3x_rule(b_runs: list[dict], c0_band_hf: list, b_band_hf: list) -> dict:
    """b_runs: [{seed, head, m6100, base(5999 point), fall, dose_ok}]; *_band_hf: in-band turn heading_final (rad) per seed."""
    out = {"n": len(b_runs)}
    arts = [r["seed"] for r in b_runs if r.get("m6100") and r.get("base") and any(
        r["base"]["hands"][h] == FAST and r["m6100"]["hands"][h] != FAST
        and None not in (r["m6100"]["feet"][h], r["base"]["feet"][h])
        and r["m6100"]["feet"][h] - r["base"]["feet"][h] <= -P1.MATERIAL for h in HANDS)]
    counted = [r["seed"] for r in b_runs if P1.ff(r.get("head")) and r.get("fall") is not None and r["fall"] <= TRIP
               and r.get("dose_ok") is True]
    out.update(artifacts=arts, counted=counted)
    miss = []
    if len(b_runs) < 3 or any(P1.has_na(r.get("head")) or not r.get("head") for r in b_runs):
        miss.append("a t3x head missing or NA")
    for r in b_runs:
        if P1.ff(r.get("head")) and (r.get("dose_ok") is None or r.get("fall") is None):
            miss.append(f"s{r['seed']} dose/fall read")
        if r.get("base") and any(r["base"]["hands"][h] == FAST for h in HANDS) and not r.get("m6100"):
            miss.append(f"s{r['seed']} model_6100 read")
    if sum(x is not None for x in b_band_hf) < 3 or sum(x is not None for x in c0_band_hf) < 3:
        miss.append("in-band turn reads")
    if miss:
        out.update(reading="INCOMPLETE", why="missing: " + ", ".join(miss) + " -- notify")
        return out
    if arts:
        out.update(reading="RESUME ARTIFACT", why=f"a FAST hand dropped at model_6100 in seeds {arts}: stop and ask")
        return out
    bf, cf = list(b_band_hf), list(c0_band_hf)
    settles = sum(bf) / 3 <= sum(cf) / 3 + BAND_RAD
    out.update(in_band_heading_final_B=bf, in_band_heading_final_C0=cf, in_band_settles=settles)
    if len(counted) >= 2 and settles:
        out.update(reading="TAILS CANDIDATE", why=f"B FAST/FAST {len(counted)}/3 at 8998 and the in-band turn settles: "
                                                  "running the P1 tails from these heads is the user's call (stop and ask)")
    elif len(counted) >= 2:
        out.update(reading="FORMED, TURN NOT IN-BAND", why=f"B FAST/FAST {len(counted)}/3 at 8998 but the in-band turn does not "
                                                           "settle (B heading_final > C0 + 0.05 rad): stop and ask")
    else:
        out.update(reading="PLATEAU", why=f"B FAST/FAST {len(counted)}/3 at 8998 (in-band settles {settles})")
    return out


def c4_gate(c4_points: dict) -> dict:
    """c4_points: {seed: point or None} for the B side-first c4 heads."""
    miss = [s for s in SEEDS if not c4_points.get(s) or P1.has_na(c4_points[s])]
    if miss:
        return {"stop": True, "reading": "INCOMPLETE", "why": f"c4 reads missing or NA: seeds {miss} -- stop and ask"}
    lost = [s for s in SEEDS if _lost(c4_points[s])]
    out = {"lost": lost, "per_seed": {s: c4_points[s]["hands"] for s in SEEDS}}
    if len(lost) >= 2:
        out.update(stop=True, reading="STOP: STROKE LOST", why=f"neither hand >= INT at the c4 head in seeds {lost}")
    else:
        out.update(stop=False, reading="CONTINUE", why=f"stroke lost in {len(lost)}/3 seeds at c4")
    return out


def sf_checks(bsf: dict) -> str | None:
    """INVALID EVAL / PENDING before any side-first decision (13.4); None = reads complete and valid."""
    for s in SEEDS:
        r = bsf.get(s)
        if not r or not r["points"].get("head"):
            return "PENDING"
        if r["clock_hidden"]["suite"] is not True or r["clock_hidden"]["lr"] is not True:
            return "INVALID EVAL"
    for s in SEEDS:
        r = bsf[s]
        if P1.has_na(r["points"]["head"]) or not r["points"].get("c6") or P1.has_na(r["points"].get("c6")):
            return "PENDING"
    return None


def c0_trigger(c6x_points: dict) -> dict:
    k = sum(P1.ff(c6x_points.get(s)) for s in SEEDS)
    return {"run_c0": k >= 2, "B_fast_fast_hands_c6x": k}


def final_without_c0(c6_points: dict, c6x_points: dict) -> dict:
    k6 = sum(P1.ff(c6_points.get(s)) for s in SEEDS)
    k6x = sum(P1.ff(c6x_points.get(s)) for s in SEEDS)
    if k6 >= 2 and k6x <= 1:
        return {"gate": "CLOCK-CUT LOSS", "why": f"B FAST/FAST (hands) at c6 {k6}/3 but at c6x {k6x}/3 -- stop and ask"}
    return {"gate": "FAIL", "why": f"B FAST/FAST (hands) at c6x {k6x}/3 (c6 {k6}/3); the C0 tails were not triggered"}


# ------------------------------------------------------------------------------------------------ loading
def load(with_tb: bool = True) -> dict:
    res = {"band": {}, "sampled": {}, "t3x": {}, "c4": {}, "sf": {}}
    names = [f"P1B6__{a}_t3_s{s}" for a in ("B", "C0") for s in SEEDS] + [f"FB6k__B_s{s}" for s in SEEDS] + \
        ["W_p1b6"] + [f"P1B6__B_t3x_s{s}" for s in SEEDS] + ["P1B6__C0_t3x_s16"] + \
        [f"SF__{a}_rest_s{s}{x}" for a in ("B", "C0") for s in SEEDS for x in ("", "_c6")]
    for n in names:
        res["band"][n] = band_read(newest_stamp(f"{n}_tb", BAND_ROW))
    for n in [f"P1B6__B_t3_s{s}" for s in SEEDS] + ["P1B6__C0_t3_s16"]:
        st = newest_stamp(f"{n}_sampled", "suite__omni")
        res["sampled"][n] = {k: v for k, v in (P1.suite_reads(st) or {}).items()
                             if k in ("turn_facing_closed", "turn_yaw_progress", "side_ratio", "fwd_ratio")} if st else None
    for a, seeds in (("B", SEEDS), ("C0", (16,))):
        for s in seeds:
            ref = f"P1B6__{a}_t3x_s{s}"
            rec = _rec(ref)
            base_rec = _rec(f"P1B6__{a}_t3_s{s}")
            if rec is None:
                res["t3x"][ref] = None
                continue
            head = head_point(rec, ref)
            res["t3x"][ref] = {"seed": s, "arm": a, "head": head, "m6100": head_point(rec, ref, "6100"),
                               "m7500": head_point(rec, ref, "7500"),
                               "base": head_point(base_rec, f"P1B6__{a}_t3_s{s}") if base_rec else None,
                               "fall": head["fall"] if head else None, "suite": P1.suite_reads(rec.get("driver_stamp")),
                               **({"dose_ok": X.tail_reads(rec.get("run_dir")).get("dose_ok")} if (with_tb and a == "B") else {})}
    for a in ("B", "C0"):
        for s in SEEDS:
            ref = f"SF__{a}_c4_s{s}"
            rec = _rec(ref)
            res["c4"][ref] = {"head": head_point(rec, ref), "suite": P1.suite_reads(rec.get("driver_stamp"))} if rec else None
            rref = f"SF__{a}_rest_s{s}"
            rrec = _rec(rref)
            res["sf"][rref] = P1.load_run(rrec, ARMS, with_tb) if rrec else None
    return res


def evaluate(res: dict, w: dict | None) -> dict:
    out = {}
    b = [r for k, r in res["t3x"].items() if r and r["arm"] == "B"]
    out["t3x"] = t3x_rule(b, [((res["band"].get(f"P1B6__C0_t3_s{s}") or {}).get("turn") or {}).get("heading_final") for s in SEEDS],
                          [((res["band"].get(f"P1B6__B_t3x_s{s}") or {}).get("turn") or {}).get("heading_final") for s in SEEDS])
    out["c4"] = c4_gate({s: (res["c4"].get(f"SF__B_c4_s{s}") or {}).get("head") for s in SEEDS})
    bsf = {s: res["sf"].get(f"SF__B_rest_s{s}") for s in SEEDS}
    c6x = {s: (r or {}).get("points", {}).get("head") for s, r in bsf.items()}
    c6 = {s: (r or {}).get("points", {}).get("c6") for s, r in bsf.items()}
    chk = sf_checks(bsf)
    out["c0_trigger"] = dict(c0_trigger(c6x), checks=chk)
    c0_present = all(res["sf"].get(f"SF__C0_rest_s{s}") for s in SEEDS)
    if chk:
        out["final"] = {"gate": chk, "why": "side-first B c6/c6x reads " + ("missing or NA" if chk == "PENDING" else
                                                                          "without the clock hidden -- stop and ask")}
    elif not out["c0_trigger"]["run_c0"]:
        out["final"] = final_without_c0(c6, c6x)
    elif not c0_present:
        out["final"] = {"gate": "PENDING", "why": "the C0 side-first tails are triggered and not read yet"}
    else:
        runs = [r for r in res["sf"].values() if r]
        fin = P1.evaluate(runs, w)
        if fin.get("gate") == "COST":
            bad = [c for c, v in fin["categories"].items() if c in P1.GATED and v and v["clearly_inferior"]]
            bh = [((res["band"].get(f"SF__B_rest_s{s}") or {}).get("turn") or {}).get("heading_final") for s in SEEDS]
            ch = [((res["band"].get(f"SF__C0_rest_s{s}") or {}).get("turn") or {}).get("heading_final") for s in SEEDS]
            if bad and all(c.startswith("turn_") for c in bad) and None not in bh + ch and sum(bh) / 3 <= sum(ch) / 3 + BAND_RAD:
                fin["gate"] = "COST (turn, out-of-band)"
                fin["why"] += ("; B's c6x in-band turn holds against the C0 lineage -> the pre-registered next lever is the "
                               "facing-band c6x variant (KRABBY_OMNI_FACING widened)")
        out["final"] = fin
    return out


def markdown(res: dict, ev: dict) -> str:
    f = P1._f
    L = ["# Turn band, t3x and side-first -- the pre-registered reading (CHARTER 13)", "",
         "## Turn band (in-band +-0.5 rad = the read; +-1.0 rad = 0.4 rad beyond the trained band, descriptive)", "",
         "| head | turn heading_final (rad) | turn facing_closed | turn yaw_progress | turn_mid heading_final | turn_mid facing_closed |",
         "|---|---|---|---|---|---|"]
    for n, b in res["band"].items():
        if b:
            L.append(f"| {n} | {f(b['turn']['heading_final'])} | {f(b['turn']['facing_closed'])} | {f(b['turn']['yaw_progress'])} | "
                     f"{f(b['turn_mid']['heading_final'])} | {f(b['turn_mid']['facing_closed'])} |")
    L += ["", "Sampled-action suite (diagnostic only, never gated):", ""]
    for n, s in res["sampled"].items():
        if s:
            L.append(f"- {n}: turn facing_closed {f(s.get('turn_facing_closed'))}, yaw_progress {f(s.get('turn_yaw_progress'))}")
    t = ev["t3x"]
    L += ["", f"## t3x: **{t['reading']}** -- {t['why']}", ""]
    for n, r in res["t3x"].items():
        if r and r.get("head"):
            hh = lambda p: "/".join(x[:4] for x in p["hands"].values()) + " " + "/".join(f(v) for v in p["feet"].values()) if p else "--"  # noqa: E731
            L.append(f"- {n}: 5999 {hh(r['base'])}; 6100 {hh(r['m6100'])}; 7500 {hh(r['m7500'])}; head {hh(r['head'])}")
    c = ev["c4"]
    L += ["", f"## Side-first c4 gate: **{c['reading']}** -- {c['why']}", "",
          f"## C0 trigger: run the C0 side-first tails = {ev['c0_trigger']['run_c0']} "
          f"(B FAST/FAST hands at c6x {ev['c0_trigger']['B_fast_fast_hands_c6x']}/3)", ""]
    fin = ev["final"]
    L += [f"## Side-first final gate: **{fin['gate']}** -- {fin['why']}", ""]
    if "seeds" in fin:
        L.append(P1.markdown(fin))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ self-test
def _self_test() -> int:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}")

    def pt(lc, rc, lf=0.2, rf=0.2, fall=0.0):
        return {"hands": {"l": lc, "r": rc}, "feet": {"l": lf, "r": rf}, "fall": fall}
    F, I, S = FAST, "INTERMEDIATE", "SLOW"
    FFp = pt(F, F)
    b = [{"seed": s, "head": FFp, "m6100": pt(I, S, 0.17, 0.13), "base": pt(I, S, 0.15, 0.13), "fall": 0.0, "dose_ok": True}
         for s in SEEDS]
    check("t3x: B 3/3 + in-band settles (0.14 vs 0.12 rad) -> TAILS CANDIDATE",
          t3x_rule(b, [0.12] * 3, [0.14] * 3)["reading"], "TAILS CANDIDATE")
    check("t3x: B 3/3, in-band 0.30 vs 0.12 rad -> FORMED, TURN NOT IN-BAND",
          t3x_rule(b, [0.12] * 3, [0.30] * 3)["reading"], "FORMED, TURN NOT IN-BAND")
    b1 = [dict(b[0]), dict(b[1], head=pt(I, S)), dict(b[2], head=pt(F, I))]
    check("t3x: B 1/3 -> PLATEAU", t3x_rule(b1, [0.12] * 3, [0.12] * 3)["reading"], "PLATEAU")
    ba = [dict(b[0], base=pt(F, I, 0.19, 0.15), m6100=pt(I, I, 0.15, 0.15))] + b[1:]
    check("t3x: a FAST hand dropped at 6100 -> RESUME ARTIFACT", t3x_rule(ba, [0.12] * 3, [0.12] * 3)["reading"], "RESUME ARTIFACT")
    check("t3x: a missing head -> INCOMPLETE", t3x_rule(b[:2], [0.12] * 3, [0.12] * 3)["reading"], "INCOMPLETE")
    check("t3x: a missing dose on a FAST/FAST head -> INCOMPLETE",
          t3x_rule([dict(b[0], dose_ok=None)] + b[1:], [0.12] * 3, [0.12] * 3)["reading"], "INCOMPLETE")
    check("t3x: a missing in-band read -> INCOMPLETE", t3x_rule(b, [0.12, None, 0.12], [0.12] * 3)["reading"], "INCOMPLETE")
    check("t3x: a FAST-at-5999 hand with no model_6100 read -> INCOMPLETE",
          t3x_rule([dict(b[0], base=pt(F, I), m6100=None)] + b[1:], [0.12] * 3, [0.12] * 3)["reading"], "INCOMPLETE")
    def sfr(head, c6, hidden=True):
        return {"points": {"head": head, "c6": c6}, "clock_hidden": {"suite": hidden, "lr": hidden}}
    check("sf checks: complete -> None", sf_checks({s: sfr(FFp, FFp) for s in SEEDS}), None)
    check("sf checks: a clock-visible c6x eval -> INVALID EVAL", sf_checks({16: sfr(FFp, FFp, False), 17: sfr(FFp, FFp),
                                                                            18: sfr(FFp, FFp)}), "INVALID EVAL")
    check("sf checks: a missing c6 read -> PENDING", sf_checks({16: sfr(FFp, None), 17: sfr(FFp, FFp), 18: sfr(FFp, FFp)}), "PENDING")
    check("c4: stroke kept -> CONTINUE", c4_gate({16: FFp, 17: pt(F, I), 18: pt(I, S)})["reading"], "CONTINUE")
    check("c4: neither hand >= INT in 2/3 -> STOP", c4_gate({16: pt(S, S), 17: pt(S, "DEAD"), 18: FFp})["reading"], "STOP: STROKE LOST")
    check("c4: a NA hand -> INCOMPLETE", c4_gate({16: pt(F, "NA"), 17: FFp, 18: FFp})["reading"], "INCOMPLETE")
    check("C0 trigger: B FAST/FAST hands 2/3 at c6x -> run", c0_trigger({16: FFp, 17: FFp, 18: pt(F, I)})["run_c0"], True)
    check("C0 trigger: 1/3 -> skip", c0_trigger({16: FFp, 17: pt(I, I), 18: pt(F, I)})["run_c0"], False)
    check("final w/o C0: c6 3/3, c6x 1/3 -> CLOCK-CUT LOSS",
          final_without_c0({s: FFp for s in SEEDS}, {16: FFp, 17: pt(I, I), 18: pt(S, S)})["gate"], "CLOCK-CUT LOSS")
    check("final w/o C0: c6 1/3, c6x 1/3 -> FAIL",
          final_without_c0({16: FFp, 17: pt(I, I), 18: pt(I, I)}, {16: FFp, 17: pt(I, I), 18: pt(S, S)})["gate"], "FAIL")
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--no-tb", action="store_true")
    ap.add_argument("--t3x-rule", action="store_true")
    ap.add_argument("--c4-gate", action="store_true")
    ap.add_argument("--c0-trigger", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    gate_only = a.c4_gate or a.c0_trigger
    res = load(with_tb=not (a.no_tb or gate_only))
    _, w = P1.load_all(ARMS, with_tb=False)
    ev = evaluate(res, w)
    if a.t3x_rule:
        print(json.dumps(ev["t3x"], default=str))
        return 0
    if a.c4_gate:
        print(json.dumps(ev["c4"], default=str))
        return 4 if ev["c4"]["stop"] else 0
    if a.c0_trigger:
        print(json.dumps(ev["c0_trigger"]))
        if ev["c0_trigger"].get("checks"):
            return 2                       # INVALID EVAL / PENDING: the queue stops and asks
        return 0 if ev["c0_trigger"]["run_c0"] else 3
    md = markdown(res, ev)
    if a.json:
        a.json.write_text(json.dumps({"reads": res, "rules": ev}, indent=1, default=str))
    if a.md:
        a.md.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
