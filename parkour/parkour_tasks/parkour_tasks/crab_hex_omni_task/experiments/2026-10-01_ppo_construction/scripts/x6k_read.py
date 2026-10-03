#!/usr/bin/env python3
"""x6k reader (CHARTER 10.4 / 10.5): the pre-registered reading of the +3000-iteration forks of the C_lr heads.

CPU only, read-only. Inputs: the fork records ($OUT/arms_x6k/<ref>.record.json, written by run_arm.sh with
RESUME_READ=1 and EXTRA_LR_EVALS="3100 4000 5000"), the step-3 records of the heads they fork ($OUT/arms; the
model_2999 values) and each fork's own tfevents (dose, priv_reg). The class rules are verdict.py's own
(``hand_class`` = feet AND along, fail-closed NA; ``fall_share``), imported, not copied.

Definitions (10.4):
  hand feet       = mean of the hand's lat_<h>_022 and lat_<h>_025 feet_ss;
  second hand     = the hand NOT FAST at model_2999 (one-FAST heads); first hand(s) = the FAST one(s);
  completes       = second hand FAST at the head AND its feet rose >= 0.03 from model_2999 AND every first hand
                    still FAST AND the run not tripped (fall share > 0.15 = not FAST, 4.4);
  drops           = a hand FAST at model_2999, below FAST at the head, whose feet fell >= 0.03;
  resume artifact = such a drop already present at model_3100.
Reading (10.5), precedence RESUME ARTIFACT > INTERFERENCE > LATE FORMATION > PLATEAU > MIXED, over the four one-FAST
B forks (drops over all five B forks): artifact in any fork -> stop and ask; drops in >= 2 of 5 -> INTERFERENCE;
>= 2 of 4 complete -> LATE FORMATION; 0 of 4 complete and a second-hand rise < 0.03 in >= 3 of 4 -> PLATEAU; else
MIXED. s12: STABLE if FAST/FAST at the head. C0: FROZEN if SLOW/SLOW at the head and each hand rose < 0.03 (else
"C0 moved under the 0.04 -> 0.1 priv_reg ramp"). Candidates: a B fork FAST/FAST at the head, not tripped, with
median |mu| <= 1.5 on cams and rods over its own last 100 logged iterations.

    x6k_read.py [--fork-arms D] [--src-arms D] [--candidates F] [--json OUT] [--md OUT] [--no-tb]
    x6k_read.py --self-test
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/ppo_construction")
_spec = importlib.util.spec_from_file_location("_x6k_verdict", HERE / "verdict.py")
V = importlib.util.module_from_spec(_spec)
sys.modules["_x6k_verdict"] = V
_spec.loader.exec_module(V)

FAST = "FAST"
MATERIAL = 0.03                 # the 4.5 materiality convention (10.5)
TRIP = 0.15                     # C_lr's 4.4 threshold (reference 0.0375 -> max(0.15, 0.1375))
DOSE = V.DOSE_MU_MED            # 1.5
TAIL = 100                      # the fork's own last logged iterations (dose, priv_reg)
EXTRA = ("3100", "4000", "5000")
HANDS = ("l", "r")


def _lj(p):
    try:
        return json.load(open(p)) if p else None
    except (OSError, ValueError):
        return None


def hand_feet(class_json: dict | None, h: str) -> float | None:
    if not class_json:
        return None
    L = class_json.get("labels") or {}
    v = [L.get(f"lat_{h}_{s}", {}).get("feet_ss") for s in ("022", "025")]
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if len(v) == 2 else None


def read_point(class_path, along_path, key) -> dict | None:
    """{hands: {l, r}, feet: {l, r}, fall} of one both-hands stamp's class + along jsons, keyed by ``key``."""
    cj, aj = _lj(class_path), _lj(along_path)
    c = (cj or {}).get(key)
    if not c:
        return None
    a = (aj or {}).get(key) or {}
    return {"hands": {"l": V.hand_class(c.get("class"), a.get("class_along")),
                      "r": V.hand_class(c.get("class_r"), a.get("class_along_r"))},
            "feet": {h: hand_feet(c, h) for h in HANDS}, "fall": V.fall_share(c)}


def fork_source(cands: dict, ref: str) -> str:
    chain = cands["references"][ref].get("chain") or ref
    return cands["chains"][chain]["stages"][0]["resume_from"]["reference"]


def tail_reads(run_dir: str | None) -> dict:
    tags = {"mu_med_cam": "PPO/mu_med_cam", "mu_med_rod": "PPO/mu_med_rod", "priv_reg": "Loss/priv_reg_coef"}
    if not run_dir:
        return {}
    ser = V.tb_series(Path(run_dir), tags)
    last = ser.get("_last", 0)
    out = {k: V.window(ser.get(k), last - TAIL + 1, last) for k in ("mu_med_cam", "mu_med_rod")}
    pr = ser.get("priv_reg")
    out["priv_reg_head"] = float(pr[1][-1]) if pr is not None and len(pr[1]) else None
    out["last_step"] = last
    ok = out["mu_med_cam"] is not None and out["mu_med_rod"] is not None
    out["dose_ok"] = bool(ok and out["mu_med_cam"] <= DOSE and out["mu_med_rod"] <= DOSE) if ok else None
    return out


def fork_row(rec: dict, src_rec: dict, tail: dict) -> dict:
    ref = rec["ref"]
    src = src_rec["ref"]
    base = read_point(src_rec.get("lr_class_json"), src_rec.get("lr_along_json"), src)
    head = read_point(rec.get("lr_class_json"), rec.get("lr_along_json"), ref)
    pts = {}
    for n in EXTRA:
        e = (rec.get("extra_lr") or {}).get(n) or {}
        pts[n] = read_point(e.get("class_json"), e.get("along_json"), f"{ref}_m{n}")
    arm = "C0" if "__C0_" in ref else "B"
    row = {"ref": ref, "source": src, "arm": arm, "m2999": base, "points": pts, "head": head, "tail": tail,
           "notes": []}
    if not base or not head:
        row["status"] = "INCOMPLETE"
        row["notes"].append("missing " + ("model_2999 read" if not base else "head read"))
        return row
    row["status"] = "READ"
    row["tripped"] = head["fall"] is not None and head["fall"] > TRIP
    first = [h for h in HANDS if base["hands"][h] == FAST]
    second = [h for h in HANDS if base["hands"][h] != FAST]
    row["first"], row["second"] = first, second

    def fell(pt, h):
        return (pt and pt["hands"][h] != FAST and pt["feet"][h] is not None and base["feet"][h] is not None
                and pt["feet"][h] - base["feet"][h] <= -MATERIAL)

    p3100 = pts.get("3100")
    row["artifact"] = None if p3100 is None else any(fell(p3100, h) for h in first)
    if p3100 is None and first:
        row["notes"].append("model_3100 read missing: the RESUME ARTIFACT rule is unread for this fork")
    row["drops"] = [h for h in first if fell(head, h)]
    row["rise"] = {h: (None if head["feet"][h] is None or base["feet"][h] is None
                       else head["feet"][h] - base["feet"][h]) for h in HANDS}
    if arm == "B" and len(first) == 1:
        s, f = second[0], first[0]
        row["role"] = "one-FAST"
        row["completes"] = bool(head["hands"][s] == FAST and (row["rise"][s] or 0) >= MATERIAL
                                and head["hands"][f] == FAST and not row["tripped"])
        row["second_flat"] = row["rise"][s] is not None and row["rise"][s] < MATERIAL
    elif arm == "B" and len(first) == 2:
        row["role"] = "FAST/FAST"
        row["stable"] = head["hands"]["l"] == FAST and head["hands"]["r"] == FAST and not row["tripped"]
    elif arm == "C0":
        row["role"] = "C0 control"
        row["frozen"] = (all(head["hands"][h] == "SLOW" for h in HANDS)
                         and all(r is not None and r < MATERIAL for r in row["rise"].values()))
    else:
        row["role"] = "other"
    row["candidate"] = bool(arm == "B" and head["hands"]["l"] == FAST and head["hands"]["r"] == FAST
                            and not row["tripped"] and tail.get("dose_ok") is True)
    return row


def reading(rows: list[dict]) -> dict:
    b = [r for r in rows if r["arm"] == "B" and r.get("status") == "READ"]
    one = [r for r in b if r.get("role") == "one-FAST"]
    out = {"n_B": len(b), "n_one_fast": len(one)}
    arts = [r["ref"] for r in rows if r.get("artifact")]
    out["artifacts"] = arts
    out["drops"] = [r["ref"] for r in b if r.get("drops")]
    out["completes"] = [r["ref"] for r in one if r.get("completes")]
    out["second_flat"] = [r["ref"] for r in one if r.get("second_flat")]
    out["candidates"] = [r["ref"] for r in rows if r.get("candidate")]
    if arts:
        out["reading"] = "RESUME ARTIFACT"
        out["why"] = f"a FAST hand already dropped at model_3100 in {arts}: stop and ask (10.5)"
    elif len(b) < 5 or len(one) < 4:
        out["reading"] = "PENDING"
        out["why"] = f"{len(b)}/5 B forks read ({len(one)}/4 one-FAST)"
    elif len(out["drops"]) >= 2:
        out["reading"] = "INTERFERENCE"
        out["why"] = f"a FAST hand dropped in {len(out['drops'])}/5 B forks: {out['drops']}"
    elif len(out["completes"]) >= 2:
        out["reading"] = "LATE FORMATION"
        out["why"] = f"{len(out['completes'])}/4 one-FAST forks complete: {out['completes']}"
    elif not out["completes"] and len(out["second_flat"]) >= 3:
        out["reading"] = "PLATEAU"
        out["why"] = f"0/4 complete; second hand rose < {MATERIAL} in {len(out['second_flat'])}/4"
    else:
        out["reading"] = "MIXED"
        out["why"] = (f"{len(out['completes'])}/4 complete, {len(out['drops'])}/5 drops, "
                      f"{len(out['second_flat'])}/4 flat second hands")
    s12 = [r for r in b if r.get("role") == "FAST/FAST"]
    out["s12"] = ("STABLE" if s12[0].get("stable") else "NOT STABLE") if s12 else "unread"
    c0 = [r for r in rows if r.get("role") == "C0 control"]
    out["c0"] = (("FROZEN" if c0[0].get("frozen") else "C0 moved under the 0.04 -> 0.1 priv_reg ramp")
                 if c0 else "unread")
    return out


def evaluate(fork_arms: Path, src_arms: Path, cands: dict, with_tb: bool = True) -> dict:
    rows = []
    for p in sorted(fork_arms.glob("*.record.json")):
        rec = json.load(open(p))
        if rec.get("killed"):
            rows.append({"ref": rec["ref"], "arm": "?", "status": "KILLED", "notes": [str(rec.get("killed"))[:200]]})
            continue
        src = fork_source(cands, rec["ref"])
        src_rec = _lj(src_arms / f"{src}.record.json") or {"ref": src}
        tail = tail_reads(rec.get("run_dir")) if with_tb else {}
        rows.append(fork_row(rec, src_rec, tail))
    return {"rows": rows, "reading": reading(rows)}


def _f(x, fmt="{:.3f}"):
    return "--" if x is None else fmt.format(x)


def markdown(res: dict) -> str:
    L = ["# x6k: B to 6000 iterations on C_lr -- the pre-registered reading (CHARTER 10)", "",
         "Hands L / R: class (feet = mean of the 0.22 and 0.25 holds). model_2999 is the step-3 head.", "",
         "| fork | role | model_2999 | model_3100 | model_4000 | model_5000 | head (5998) | rise L / R | "
         "fall | dose cam/rod (last 100) | priv_reg | result |",
         "|" + "---|" * 12]

    def pt(p):
        if not p:
            return "--"
        return " / ".join(f"{p['hands'][h][:4]} {_f(p['feet'][h])}" for h in HANDS)

    for r in res["rows"]:
        if r.get("status") != "READ":
            L.append(f"| {r['ref']} | -- | {r.get('status')} | | | | | | | | | {'; '.join(r.get('notes') or [])} |")
            continue
        t = r.get("tail") or {}
        res_txt = []
        if r.get("artifact"):
            res_txt.append("RESUME ARTIFACT")
        if r.get("drops"):
            res_txt.append(f"drops {r['drops']}")
        if "completes" in r:
            res_txt.append("completes" if r["completes"] else ("flat" if r.get("second_flat") else "not complete"))
        if "stable" in r:
            res_txt.append("STABLE" if r["stable"] else "NOT STABLE")
        if "frozen" in r:
            res_txt.append("FROZEN" if r["frozen"] else "moved")
        if r.get("candidate"):
            res_txt.append("**candidate**")
        if r.get("tripped"):
            res_txt.append("TRIPPED")
        L.append(f"| {r['ref']} | {r['role']} | {pt(r['m2999'])} | {pt(r['points'].get('3100'))} "
                 f"| {pt(r['points'].get('4000'))} | {pt(r['points'].get('5000'))} | {pt(r['head'])} "
                 f"| {_f(r['rise']['l'], '{:+.3f}')} / {_f(r['rise']['r'], '{:+.3f}')} | {_f(r['head']['fall'])} "
                 f"| {_f(t.get('mu_med_cam'), '{:.2f}')}/{_f(t.get('mu_med_rod'), '{:.2f}')} "
                 f"| {_f(t.get('priv_reg_head'), '{:.4f}')} | {'; '.join(res_txt)} |")
    g = res["reading"]
    L += ["", f"**Reading: {g['reading']}** -- {g['why']}", "",
          f"- s12 (FAST/FAST head): {g['s12']}", f"- C0 control: {g['c0']}",
          f"- two-direction candidates (FAST/FAST, not tripped, dose ok): {g['candidates'] or 'none'}"]
    notes = [f"- {r['ref']}: {n}" for r in res["rows"] for n in (r.get("notes") or [])]
    if notes:
        L += ["", "Notes:"] + notes
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ self-test
def _pt(lc, rc, lf, rf, fall=0.01):
    return {"hands": {"l": lc, "r": rc}, "feet": {"l": lf, "r": rf}, "fall": fall}


def _row(ref, base, head, p3100=None, dose_ok=True):
    """fork_row on synthetic points (bypasses the json files)."""
    def fake_rp(path, along, key):
        return {"BASE": base, "HEAD": head, "P3100": p3100}.get(path)
    real = globals()["read_point"]
    globals()["read_point"] = fake_rp
    try:
        rec = {"ref": ref, "lr_class_json": "HEAD", "lr_along_json": None,
               "extra_lr": {"3100": {"class_json": "P3100"}} if p3100 else {}}
        return fork_row(rec, {"ref": "SRC", "lr_class_json": "BASE"}, {"dose_ok": dose_ok})
    finally:
        globals()["read_point"] = real


def _self_test() -> int:
    ok = True

    def check(name, got, want):
        nonlocal ok
        good = got == want
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {got}")

    s15_base = _pt("INTERMEDIATE", FAST, 0.1646, 0.2475)            # s15: L 0.1556/0.1736 -> mean 0.1646
    r = _row("C_lr__B_x6k_s15", s15_base, _pt(FAST, FAST, 0.1670, 0.2480))
    check("s15 boundary: a flip to FAST on +0.0024 is NOT a completion", (r["completes"], r["second_flat"]), (False, True))
    r = _row("C_lr__B_x6k_s15", s15_base, _pt(FAST, FAST, 0.1990, 0.2480))
    check("s15: FAST with a +0.034 rise completes", r["completes"], True)
    r = _row("C_lr__B_x6k_s15", s15_base, _pt(FAST, FAST, 0.1990, 0.2480, fall=0.2))
    check("a tripped run never completes", (r["completes"], r["tripped"]), (False, True))
    b11 = _pt(FAST, "INTERMEDIATE", 0.240, 0.149)
    r = _row("C_lr__B_x6k_s11", b11, _pt("INTERMEDIATE", FAST, 0.200, 0.190))
    check("a first hand below FAST and -0.04 drops (and the run does not complete)", (r["drops"], r["completes"]),
          (["l"], False))
    r = _row("C_lr__B_x6k_s11", b11, _pt(FAST, FAST, 0.230, 0.190), p3100=_pt("INTERMEDIATE", "SLOW", 0.200, 0.120))
    check("a drop already at model_3100 is a resume artifact", r["artifact"], True)
    r = _row("C_lr__B_x6k_s11", b11, _pt(FAST, FAST, 0.230, 0.190), p3100=_pt(FAST, "INTERMEDIATE", 0.235, 0.150))
    check("no drop at model_3100: no artifact", r["artifact"], False)
    s12 = _row("C_lr__B_x6k_s12", _pt(FAST, FAST, 0.259, 0.193), _pt(FAST, FAST, 0.26, 0.21))
    check("s12 FAST/FAST at the head is STABLE", (s12["role"], s12["stable"]), ("FAST/FAST", True))
    c0 = _row("C_lr__C0_x6k_s11", _pt("SLOW", "SLOW", 0.084, 0.081), _pt("SLOW", "SLOW", 0.090, 0.085))
    check("C0 SLOW/SLOW with rises < 0.03 is FROZEN", c0["frozen"], True)
    c0m = _row("C_lr__C0_x6k_s11", _pt("SLOW", "SLOW", 0.084, 0.081), _pt("INTERMEDIATE", "SLOW", 0.140, 0.085))
    check("C0 that moves is not FROZEN", c0m["frozen"], False)
    cand = _row("C_lr__B_x6k_s13", _pt(FAST, "INTERMEDIATE", 0.228, 0.137), _pt(FAST, FAST, 0.23, 0.19), dose_ok=True)
    nocand = _row("C_lr__B_x6k_s13", _pt(FAST, "INTERMEDIATE", 0.228, 0.137), _pt(FAST, FAST, 0.23, 0.19), dose_ok=False)
    check("candidate needs FAST/FAST + dose", (cand["candidate"], nocand["candidate"]), (True, False))

    def one(ref, comp):
        h = _pt(FAST, FAST, 0.23, 0.19) if comp else _pt(FAST, "INTERMEDIATE", 0.23, 0.150)
        return _row(ref, _pt(FAST, "INTERMEDIATE", 0.23, 0.140), h)
    rows = [one("C_lr__B_x6k_s11", True), one("C_lr__B_x6k_s13", True), one("C_lr__B_x6k_s14", False),
            one("C_lr__B_x6k_s15", False), s12, c0]
    check("2 of 4 complete -> LATE FORMATION", reading(rows)["reading"], "LATE FORMATION")
    rows_p = [one(f"C_lr__B_x6k_s{s}", False) for s in (11, 13, 14, 15)] + [s12, c0]
    check("0 of 4 complete, flat seconds -> PLATEAU", reading(rows_p)["reading"], "PLATEAU")
    d = _row("C_lr__B_x6k_s14", _pt(FAST, "INTERMEDIATE", 0.23, 0.14), _pt("INTERMEDIATE", FAST, 0.19, 0.20))
    d2 = _row("C_lr__B_x6k_s11", _pt(FAST, "INTERMEDIATE", 0.23, 0.14), _pt("SLOW", "INTERMEDIATE", 0.10, 0.16))
    rows_i = [one("C_lr__B_x6k_s11", True), one("C_lr__B_x6k_s13", True), d, d2, s12, c0]
    check("drops in 2 of 5 outrank 2 completions -> INTERFERENCE", reading(rows_i)["reading"], "INTERFERENCE")
    art = _row("C_lr__B_x6k_s12", _pt(FAST, FAST, 0.259, 0.193), _pt(FAST, FAST, 0.26, 0.21),
               p3100=_pt("INTERMEDIATE", FAST, 0.200, 0.20))
    check("an artifact anywhere outranks everything", reading(rows[:5] + [art])["reading"], "RESUME ARTIFACT")
    check("fewer than 5 B forks read -> PENDING", reading(rows[:3])["reading"], "PENDING")
    print("self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fork-arms", type=Path, default=OUT / "arms_x6k")
    ap.add_argument("--src-arms", type=Path, default=OUT / "arms")
    ap.add_argument("--candidates", type=Path, default=HERE.parent / "candidates.yaml")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--no-tb", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return _self_test()
    import yaml  # noqa: PLC0415

    res = evaluate(a.fork_arms, a.src_arms, yaml.safe_load(open(a.candidates)), with_tb=not a.no_tb)
    md = markdown(res)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=str))
    if a.md:
        a.md.write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
