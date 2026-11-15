#!/usr/bin/env python3
"""CPU checks of p2_read.py (P2.md step 5 "Checks"; nothing launches, nothing under results/ is written).

    p2_read_check.py [--scratch DIR] [--only 1|1b|2|3|4|5|6|7 ...]

1  the pooled read on a synthetic two-launch case: known discordant keys give the expected share (an exact count) and
   the expected SE (an independent re-implementation of thresholds.paired_completion_se's bootstrap on the FRACTIONAL
   values; the bool-cast SE and readout.share differ, so the cast trap is shown avoided); n stays 400; a bitwise-equal
   second launch reproduces readout.g_line / readout.retention_line of the single launch EXACTLY (every key); the
   near-miss windows, the T count and the decision routing on synthetic lines.
1b the same on real stamps: c1 vs TDR, slopes b2 / ledge_down b2 at es5-6 through P2ReadRows (seed map {1:5, 2:6}),
   single launch == results/short_arm/gates_fresh.json's G lines to 1e-12, and the same stamps entered twice as two
   launches == the single-launch line exactly.
2  the subclass reproduces results/short_arm/gates_fresh.json: P2ReadRows over BOTH output roots with readout_fresh's
   map {1:5, 2:6, 3:7, 4:8} (role swap idle: two teachers) -> readout.gates() unchanged -> readout_fresh's own
   post-processing -> every leaf of G / R / report / verdict / thresholds within 1e-12; the Tr reader
   (readout_fresh.suite_only + p2_read.tr_from_suite) on the same suite rows: the 22 lines equal the recorded ones, the
   18 decisive lines pass, the 4 E lines carry no flag.
3  the dry slot counts: front 5, rows 84 = T 32 + T-reg 12 + N 20 + Tr 4 + report 8 + Q3 8 = 36 base + 20 second + 20
   kind + 8 teacher; the front 5 lead; the students interleave within each (kind, es); roles; names by sha; the
   gravel friction knob; the relaunch plan from a synthetic fresh record and its 16-row cap.
4  the class-pair check on step 3's smoke twins (CLASS-FLIP -> ok) and on a deliberate cross-row pair (-> not ok,
   the HALT path); the classflip cache lives in --scratch.
5  fail-closed without results/p2/sel.json: --inputs-check reports BAD on the pick only; --near-miss / --verdict
   refuse (exit 2) and write nothing.
6  the whole verdict path (verdict_doc: T, T-reg, N, Tr + E, report, Q3, pairs, routing) on RECORDED stamps, c1
   standing in for C1S and TDR for c31000 at es5-8 (a second launch = the same stamp again): every line equals
   gates_fresh.json's, the pairs read BITWISE-EQUAL, the Q3 terms add up exactly.
7  the diag label's selection tracking WATCH: read from sel.json's blocks.<pick.cand>.watch.tracking.rows (RSD
   sel.json -> 4 diag rows, max worse_by 0.0091); worse_by 0.036 -> student loss with non-empty rows, 0.035 ->
   predicted cost (rows still shown), a survival diag fail -> student loss; a missing watch reads MISSING, never [].
Prints ``CHECK <n> PASS|FAIL ...`` per check and ``P2_READ_CHECK PASS|FAIL`` last (exit 0 / 1).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p2_read as R  # noqa: E402

import numpy as np  # noqa: E402

P3, M, readout, S, RF = R.P3, R.M, R.readout, R.S, R.RF
GATES_FRESH = R.CAMP / "results" / "short_arm" / "gates_fresh.json"
EMAP_FRESH = RF.EMAP_FRESH
TOL = 1e-12
RESULTS = {}


def report(n: str, ok: bool, detail: str) -> None:
    RESULTS[n] = ok
    print(f"CHECK {n} {'PASS' if ok else 'FAIL'} {detail}", flush=True)


def eq_dict(a: dict, b: dict) -> list:
    """Keys whose values differ (exact; lists elementwise)."""
    return sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))


# ============================================================================== 1
def synth_tables(vals: dict) -> dict:
    """{(es, i): bool} -> {es: table} with the keys a readout episode_table carries for this read."""
    out = {}
    for (es, i), v in vals.items():
        out.setdefault(es, {"success": {}, "completion": {}})
        out[es]["success"][i] = bool(v)
        out[es]["completion"][i] = bool(v)
    return out


def check1() -> None:
    thr, g_rule, r_cfg = R._thr()
    rng = np.random.default_rng(7)
    keys = [(es, i) for es in (9, 10, 11, 12) for i in range(100)]
    l1 = {k: bool(rng.random() < 0.80) for k in keys}
    ref = {k: bool(rng.random() < 0.62) for k in keys}
    ones = [k for k in keys if l1[k]]
    zeros = [k for k in keys if not l1[k]]
    disc = ones[:30] + zeros[:10]                         # 30 launch-2 losses, 10 launch-2 wins: known discordant keys
    l2 = {k: (not l1[k]) if k in set(disc) else l1[k] for k in keys}
    t1, t2, tr = synth_tables(l1), synth_tables(l2), synth_tables(ref)
    two = {es: [t1[es], t2[es]] for es in t1}
    one_ref = {es: [tr[es]] for es in tr}
    a = R.pooled_values(two, "success")
    b = R.pooled_values(one_ref, "success")
    bad = []
    n_half = sum(1 for v in a.values() if v == 0.5)
    want_share = (sum(l1.values()) + sum(l2.values())) / (2 * len(keys))
    g = R.pooled_g(a, b, g_rule, "synthetic")
    if n_half != len(disc):
        bad.append(f"{n_half} half-valued keys != {len(disc)} discordant")
    if g["n"] != len(keys) or len(a) != len(keys):
        bad.append(f"n {g['n']} / {len(a)} != {len(keys)} (launches must never be keyed)")
    if any(not (isinstance(k, tuple) and len(k) == 2 and k[0] in (9, 10, 11, 12)) for k in a):
        bad.append("a pooled key is not (es, env_index)")
    if abs(g["new"] - want_share) > 1e-15:
        bad.append(f"share {g['new']} != expected {want_share}")
    # the expected SE: an independent bootstrap on the fractional values (thresholds' seed and n_boot)
    th = readout.th()
    ks = sorted(set(a) & set(b))
    x = np.array([a[k] for k in ks], float)
    y = np.array([b[k] for k in ks], float)
    idx = np.random.default_rng(th.DEFAULT_SEED).integers(0, len(ks), (4000, len(ks)))
    se_want = float((x[idx].mean(1) - y[idx].mean(1)).std())
    if abs(g["se"] - se_want) > 1e-15:
        bad.append(f"SE {g['se']} != independent {se_want}")
    se_cast = th.paired_completion_se({k: float(bool(v)) for k, v in a.items()}, b)
    sh_cast = readout.share(a)
    if not (abs(se_cast - g["se"]) > 1e-6 and abs(sh_cast - g["new"]) > 1e-6):
        bad.append("the bool-cast SE / share do not differ from the pooled ones: the synthetic case cannot show the trap")
    # g_line clauses applied by hand
    need = max(0.10, 3 * g["se"])
    if g["gain"] != bool(g["diff"] >= need - readout.EPS and g["diff"] - 1.96 * g["se"] >= 0.05 - readout.EPS
                         and abs(g["diff"]) > 0.04 + readout.EPS):
        bad.append("gain clause")
    if g["regress"] != bool(g["diff"] < -max(0.05, 2 * g["se"]) - readout.EPS and abs(g["diff"]) > 0.04 + readout.EPS):
        bad.append("regress clause")
    # bitwise-equal second launch == the single-launch line, exactly (every key)
    once = {es: [t1[es]] for es in t1}
    twice = {es: [t1[es], t1[es]] for es in t1}
    k1 = readout.keyed(t1, "success")
    kr = readout.keyed(tr, "success")
    g1 = readout.g_line(k1, kr, "single", g_rule)
    gp1 = R.pooled_g(R.pooled_values(once, "success"), b, g_rule, "x")
    gp2 = R.pooled_g(R.pooled_values(twice, "success"), b, g_rule, "x")
    d1, d2 = eq_dict(g1, gp1), eq_dict(g1, gp2)
    if d1 or d2:
        bad.append(f"g_line vs pooled single {d1} / doubled {d2}")
    r1 = readout.retention_line(k1, kr, "single", cfg=r_cfg)
    rp2 = R.pooled_r(R.pooled_values(twice, "completion"), R.pooled_values(one_ref, "completion"), r_cfg, "single")
    dr = eq_dict(r1, rp2)
    if dr:
        bad.append(f"retention_line vs pooled doubled {dr}")
    # near-miss windows (synthetic lines)
    for lab, lnx, want in (("N short 0.03", {"pass": False, "line": 0.90, "head": 0.87}, True),
                           ("N short 0.05", {"pass": False, "line": 0.90, "head": 0.85}, False),
                           ("N pass", {"pass": True, "line": 0.90, "head": 0.95}, False)):
        if R.near_due("field_hard", lnx)[2] != want:
            bad.append(f"near_due {lab}")
    for lab, lnx, want in (("treg short 0.06", {"regress": True, "regress_margin": 0.06, "diff": -0.12}, True),
                           ("treg short 0.08", {"regress": True, "regress_margin": 0.06, "diff": -0.14}, False),
                           ("treg pass", {"regress": False, "regress_margin": 0.06, "diff": -0.02}, False)):
        if R.near_due("treg", lnx)[2] != want:
            bad.append(f"near_due {lab}")
    # T count + routing
    def tk(g_a, g_b, r_a=False, r_b=False):
        return {"slopes": {"status": "OK", "gain": g_a, "regress": r_a},
                "ledge_down": {"status": "OK", "gain": g_b, "regress": r_b}}
    cases = [(tk(True, True), "PASS", "PASS", "PASS", [], "KEEPS", "RECOMMEND_YES"),
             (tk(True, True), "PASS", "PASS", "PASS", ["e"], "KEEPS", "USER_DECIDES"),
             (tk(True, False), "PASS", "PASS", "PASS", [], "PARTIAL", "USER_DECIDES"),
             (tk(True, False, False, True), "PASS", "PASS", "PASS", [], "PARTIAL", "USER_DECIDES"),
             (tk(False, False), "PASS", "FAIL", "FAIL", ["e"], "LOST", "T_LOST"),
             (tk(True, True), "FAIL", "PASS", "PASS", [], "KEEPS", "USER_DECIDES")]
    for kinds, treg, n, trs, e, want_t, want_r in cases:
        T = R.t_verdict(kinds)
        rt = R.route(T, treg, n, trs, e)["route"]
        if T["status"] != want_t or rt != want_r:
            bad.append(f"routing {want_t}/{want_r} got {T['status']}/{rt}")
    report("1", not bad, f"synthetic two-launch: n={g['n']} discordant={len(disc)} share={g['new']:.6f} "
                         f"(want {want_share:.6f}; bool-cast share {sh_cast:.6f}) SE={g['se']:.6f} (independent "
                         f"{se_want:.6f}; bool-cast {se_cast:.6f}); doubled launch == g_line / retention_line exactly: "
                         f"{not (d1 or d2 or dr)}; near-miss + routing cases ok: "
                         f"{not [x for x in bad if x.startswith(('near', 'routing'))]}"
                         + (f" -- {bad}" if bad else ""))


def check1b() -> None:
    rec = json.loads(GATES_FRESH.read_text())
    thr, g_rule, r_cfg = R._thr()
    c1, tdr = rec["policies"]["new"], rec["policies"]["tdr"]
    Rr = R.P2ReadRows([R.RT_OUT / "lr", R.P2_OUT / "lr"], {c1, tdr}, set(), emap={1: 5, 2: 6}, real_ok=(5, 6))
    pin = P3.read_json(R.P2_PIN)
    bad, shown = [], []
    for kind, rid in R.T_KINDS.items():
        names = {1: [None], 2: [None]}
        tn, rn, bn = R.side_tables(Rr, rid, names, c1, pin)
        tr, rr, br = R.side_tables(Rr, rid, names, tdr, pin)
        if bn or br:
            bad.append(f"{kind}: {bn + br}")
            continue
        pp = R.pairing_problems(rid, rn + rr, True)
        if pp:
            bad.append(f"{kind}: pairing {pp}")
        b = R.pooled_values(tr, "success")
        g1 = R.pooled_g(R.pooled_values(tn, "success"), b, g_rule, rid)
        g2 = R.pooled_g(R.pooled_values({es: v + v for es, v in tn.items()}, "success"), b, g_rule, rid)
        want = rec["G"]["kinds"][kind]
        for k in ("new", "tdr", "diff", "se", "gain_need", "ci_lb", "regress_margin"):
            if abs(g1[k] - want[k]) > TOL:
                bad.append(f"{kind} {k} {g1[k]} != recorded {want[k]}")
        for k in ("gain", "regress", "le_noise", "n"):
            if g1[k] != want[k]:
                bad.append(f"{kind} {k} {g1[k]} != recorded {want[k]}")
        if any(abs(g1["ci95"][i] - want["ci95"][i]) > TOL for i in (0, 1)):
            bad.append(f"{kind} ci95")
        d = eq_dict(g1, g2)
        if d:
            bad.append(f"{kind}: doubled launch differs in {d}")
        shown.append(f"{kind} diff={g1['diff']:+.3f} se={g1['se']:.4f} gain={g1['gain']} (recorded "
                     f"{want['diff']:+.3f}/{want['se']:.4f}/{want['gain']})")
        R.drop_raw_cache()
    report("1b", not bad, "real stamps c1 vs TDR es5-6 via P2ReadRows: " + "; ".join(shown)
           + "; doubled launch == single exactly" + (f" -- {bad}" if bad else ""))


# ============================================================================== 2
def check2() -> None:
    rec = json.loads(GATES_FRESH.read_text())
    c1, tdr = rec["policies"]["new"], rec["policies"]["tdr"]
    Rr = R.P2ReadRows([R.RT_OUT / "lr", R.P2_OUT / "lr"], {c1, tdr}, set(), emap=EMAP_FRESH,
                      real_ok=RF.REAL_FRESH)
    thr = readout.load_thresholds(readout.THRESHOLDS)
    res = readout.gates(Rr, c1, tdr, thr)
    res["thresholds"] = {"path": str(readout.THRESHOLDS), "sha256": RF.sha256(readout.THRESHOLDS)}
    trk = res["R"].get("tracking") or {}
    if isinstance(trk.get("drift_reading"), dict):
        trk["drift_reading"]["applicability"] = RF.DRIFT_NA
        trk["drift_reading"]["why_not_applicable"] = RF.DRIFT_NA_WHY
    got = RF.relabel(readout._norm(res), EMAP_FRESH)
    keys = ("G", "R", "report", "verdict", "thresholds_decision", "thresholds")
    diffs = []
    n_num = 0
    for k in keys:
        diffs += readout.deep_diff(rec[k], got[k], path=k, tol=TOL, only_recorded=False)
        n_num += readout._count_numbers(rec[k])
    served = Rr.served
    real = sorted({s["real_es"] for s in served})
    pin = P3.read_json(R.P2_PIN)
    off = [s for s in served if s.get("stamp") and M.build_mismatch(M.stamp_build(s["stamp"]), pin)]
    roots_served = sorted({str(Path(s["stamp"]).parents[3]) for s in served if s.get("stamp")})
    # the Tr reader on the same suite rows (logical es 1, 2 = real es5, es6)
    R.drop_raw_cache()
    so = RF.suite_only(Rr, c1, tdr, (1, 2))
    tr = R.tr_from_suite(so, pin)
    rec_lines = {(x["row"], x["statistic"]): x for x in rec["R"]["tracking"]["suite_lines"]}
    line_bad = []
    for x in so.get("lines") or []:
        y = rec_lines.get((x["row"], x["statistic"]))
        if y is None:
            line_bad.append(f"{x['row']} {x['statistic']} not recorded")
            continue
        for k in ("head", "p2a", "worse_by", "se_diff", "R"):
            if abs(float(x[k]) - float(y[k])) > TOL:
                line_bad.append(f"{x['row']} {x['statistic']} {k}")
        if bool(x["fails"]) != bool(y["fails"]):
            line_bad.append(f"{x['row']} {x['statistic']} fails")
    e = tr.get("E") or {}
    e_back = [x["worse_by"] for x in e.get("lines") or [] if x["statistic"] == "track_err back"]
    ok_tr = (so["status"] in ("PASS", "FAIL") and len(so.get("lines") or []) == 22 and not line_bad
             and tr["status"] == "PASS" and tr["n_lines"] == 18 and not e.get("flags"))
    ok = (not diffs and rec["verdict"] == got["verdict"] and set(real) <= set(RF.REAL_FRESH) and not off and ok_tr)
    report("2", ok, f"P2ReadRows (both roots, map {EMAP_FRESH}) -> readout.gates() == gates_fresh.json: "
                    f"numbers={n_num} diffs={len(diffs)} verdict={got['verdict']['verdict']} (recorded "
                    f"{rec['verdict']['verdict']}) lookups={len(served)} real_es={real} roots_served={roots_served} "
                    f"off_pin={len(off)}; Tr reader: suite_only {so['status']} lines={len(so.get('lines') or [])} "
                    f"(line diffs {len(line_bad)}), decisive {tr.get('n_lines')} -> {tr['status']} (fails "
                    f"{tr.get('fails')}), E flags {e.get('flags')} (back worse_by {[round(v, 4) for v in e_back]} vs "
                    f"0.132)" + (f" -- diffs {diffs[:6]} {line_bad[:6]}" if diffs or line_bad else ""))
    R.drop_raw_cache()


# ============================================================================== 3
def check3() -> None:
    bad = []
    R.MODE["dry"] = True
    pols = R.policies(None)                                   # the DRY PLACEHOLDER pick
    front = R.plan_read("p2read_front", pols)
    rows = R.plan_read("p2read_rows", pols)
    fr = [s for s in front if s.slot]
    rr = [s for s in rows if s.slot]
    units = R.plan_units()
    gc = R.group_counts(units)
    rc = R.reconcile(units)
    if len(fr) != 5 or len(rr) != 84:
        bad.append(f"slots front {len(fr)} rows {len(rr)}")
    if gc != R.EXPECT_GROUPS:
        bad.append(f"groups {gc}")
    if not rc["agree"]:
        bad.append(f"reconcile {rc['got']}")
    if [s.slot for s in fr] != [s.slot for s in rr[:5]]:
        bad.append("the front 5 do not lead the rows block")
    fu = units[:5]
    if sorted(u["line"] for u in fu) != sorted(["ledge_up", "pits", "rubble", "hills", "gravel"]) or \
            any(u["pol"] != "C1S" or u["es"] != 9 for u in fu):
        bad.append(f"front units {[(u['line'], u['pol'], u['es']) for u in fu]}")
    # interleaving: every (group, line, es, tag) of a student group holds both students; T alternates C1S, c31000
    cells = {}
    for u in units:
        if u["group"] != "Q3":
            cells.setdefault((u["group"], u["line"], u["es"], u["tag"]), []).append(u["pol"])
    if any(sorted(v) != sorted(R.STUDENTS) for v in cells.values()):
        bad.append("a student cell lacks one student")
    tseq = [u["pol"] for u in units if u["group"] == "T"]
    if tseq != list(R.STUDENTS) * 16:
        bad.append("T rows do not alternate C1S / c31000 within each (kind, es, launch)")
    for kind in R.T_KINDS:
        for es in (9, 10, 11, 12):
            seq = [(u["pol"], u["tag"]) for u in units if u["group"] == "T" and u["line"] == kind and u["es"] == es]
            if seq != [("C1S", ""), ("c31000", ""), ("C1S", "_L2"), ("c31000", "_L2")]:
                bad.append(f"T {kind} es{es} order {seq}")
    # roles, names, friction, timeouts
    for s, u in zip(rr, units):
        role = s.argv[s.argv.index("--policy-role") + 1]
        want = "teacher" if u["pol"] in R.TEACHERS else "student"
        if role != want:
            bad.append(f"{s.label} role {role}")
        if not s.slot[1].startswith(f"P2R_{R.sha8(pols[u['pol']]['sha'])}_") or "c31000" in s.slot[1]:
            bad.append(f"{s.label} name {s.slot[1]}")
        if want == "student" and s.env.get("TP_ROW_TIMEOUT") != str(R.STUDENT_ROW_TIMEOUT_S):
            bad.append(f"{s.label} timeout")
        if s.slot[3] not in R.REAL_OK:
            bad.append(f"{s.label} es{s.slot[3]}")
    gr = R.rows_by_id()["rt_gravel_b3__omni"]
    if S.expected_env(gr).get(S.FRICTION_VAR) != S.GRAVEL_FRICTION:
        bad.append("gravel friction knob not in the composed env")
    # the relaunch plan from a synthetic FRESH record; the cap
    base_units = [(rid, les, pol, "") for _k, _l, rid, less in R.base_line_specs() for les in less
                  for pol in R.STUDENTS]
    due8 = [R.unit("N", "field_hard", "R_field_hard", les, pol, "_R1") for les in (1, 2, 3, 4) for pol in R.STUDENTS]
    doc = {"status": "DUE", "relaunch_due": due8, "base_units": [list(b) for b in base_units],
           "base_stamps": R.current_base_stamps(pols, base_units), "at": "synthetic"}
    R.MODE["dry"] = False
    try:
        got, _how = R.relaunch_units(pols, doc)
        if [R.unit_name(u, pols) for u in got] != [R.unit_name(u, pols) for u in due8]:
            bad.append("relaunch units differ from the record")
        try:
            R.relaunch_units(pols, dict(doc, relaunch_due=due8 + due8 + due8[:4]))
            bad.append("20 relaunch rows were not refused")
        except ValueError:
            pass
        try:
            R.relaunch_units(pols, dict(doc, base_stamps={"x": ["y"]}))
            bad.append("a stale record was not refused")
        except ValueError:
            pass
    finally:
        R.MODE["dry"] = False
    report("3", not bad, f"dry slots front={len(fr)} rows={len(rr)} groups={gc} reconcile={rc['got']} "
                         f"(P2.md 36 + 20 + 20 + 8 = 84: {rc['agree']}); front 5 lead; students interleave; roles, "
                         f"sha names, timeouts, es9-12, gravel friction ok; relaunch plan 8 rows, cap 16 and staleness "
                         f"refused" + (f" -- {bad}" if bad else ""))


# ============================================================================== 4
def check4(scratch: Path) -> None:
    s3 = json.loads(R.STEP3_JSON.read_text())
    pairs = s3["smoke"]["pairs"]
    bad = []
    cache = R.PairCache(scratch / "pairs_cache.json")
    recs = []
    for p in pairs:
        for nm, st in (("x", p["x"]), ("y_L2", p["y"])):
            recs.append({"rid": p["kind"], "es": 90, "sha256": R.C31000_SHA, "name": nm, "stamp": st, "status": "OK"})
    got = R.pair_check(recs, cache)
    if len(got) != 2 or not all(g["ok"] for g in got):
        bad.append(f"smoke twins {[(g['rid'], g['verdict'], g['ok']) for g in got]}")
    neg = [{"rid": "cross", "es": 90, "sha256": R.C31000_SHA, "name": "x", "stamp": pairs[0]["x"], "status": "OK"},
           {"rid": "cross", "es": 90, "sha256": R.C31000_SHA, "name": "y_L2", "stamp": pairs[1]["x"], "status": "OK"}]
    gn = R.pair_check(neg, cache)
    if len(gn) != 1 or gn[0]["ok"]:
        bad.append(f"cross-row pair {[(g['verdict'], g['ok']) for g in gn]} (want not ok)")
    cache.save()
    report("4", not bad, f"class pairs: smoke twins {[(g['rid'], g['verdict'], g['n_step0_envs_differ']) for g in got]}"
                         f" ok; cross-row pair -> {gn[0]['verdict'] if gn else None} (HALT path) "
                         f"cache={scratch / 'pairs_cache.json'}" + (f" -- {bad}" if bad else ""))


# ============================================================================== 5
def check5(scratch: Path) -> None:
    bad = []
    if R.SEL_JSON.exists():
        report("5", True, f"skipped: {R.SEL_JSON} exists (the fail-closed case needs it absent)")
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = R.inputs_check("rows")
    txt = buf.getvalue()
    line = [ln for ln in txt.splitlines() if ln.startswith("P2_INPUTS ")][-1]
    probs = line.split(" -- ", 1)[1].split("; ") if " -- " in line else []
    if rc != 1 or not probs or any(not p.startswith("pick:") for p in probs):
        bad.append(f"inputs-check rc {rc}: {probs}")
    outs = {}
    for mode in ("--near-miss", "--verdict"):
        out = scratch / f"refused{mode.replace('-', '_')}.json"
        b2 = io.StringIO()
        with contextlib.redirect_stdout(b2):
            r2 = R.main([mode, "--out", str(out), "--pairs-cache", str(scratch / "pairs_refused.json")])
        outs[mode] = (r2, out.exists(), b2.getvalue().strip()[:120])
        if r2 != 2 or out.exists():
            bad.append(f"{mode}: rc {r2} wrote {out.exists()}")
    # a synthetic sel.json in --scratch (SEL_JSON patched for this check only): the pick is read by sha
    ck = M.W / "parkour/logs/rsl_rl/crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19/model_29400.pt"
    sha = M.sha256_file(ck)
    saved = R.SEL_JSON
    syn = {}
    try:
        R.SEL_JSON = scratch / "sel.json"
        for lab, pick, want_ok in (("good", {"sha256": sha, "ckpt": str(ck), "iter": 29400}, True),
                                   ("sha mismatch", {"sha256": "0" * 64, "ckpt": str(ck)}, False),
                                   ("fixed policy", {"sha256": R.C1_SHA, "ckpt": str(R.C1_CK)}, False),
                                   ("relative path", {"sha256": sha, "ckpt": "model_29400.pt"}, False)):
            R.SEL_JSON.write_text(json.dumps({"verdict": "SELECTED", "pick": pick}))
            got = R.load_pick(strict=False)
            syn[lab] = got["ok"]
            if got["ok"] != want_ok:
                bad.append(f"synthetic pick {lab}: ok {got['ok']} {got['problems']}")
        R.SEL_JSON.write_text(json.dumps({"verdict": "HALT", "pick": {"sha256": sha, "ckpt": str(ck)}}))
        syn["verdict HALT"] = R.load_pick(strict=False)["ok"]
        if syn["verdict HALT"]:
            bad.append("a non-SELECTED sel.json was accepted")
        R.SEL_JSON.write_text(json.dumps({"verdict": "SELECTED", "pick": {"sha256": sha, "ckpt": str(ck)}}))
        strict = R.load_pick(strict=True)
        syn["strict (not committed)"] = strict["ok"]
        if strict["ok"]:
            bad.append("an uncommitted sel.json passed the strict read")
        pols = R.policies(R.load_pick(strict=False))
        if pols["C1S"]["sha"] != sha or R.unit_name(R.plan_units()[0], pols) != f"P2R_{sha[:8]}_rt_ledge_up_b2":
            bad.append("the pick does not name the C1S rows by its sha8")
    finally:
        R.SEL_JSON = saved
    report("5", not bad, f"no sel.json: inputs-check rc={rc} problems={probs}; near-miss / verdict "
                         f"{ {k: v[:2] for k, v in outs.items()} }; synthetic sel.json ok-flags {syn}"
                         + (f" -- {bad}" if bad else ""))


# ============================================================================== 6
def check6(scratch: Path) -> None:
    """The whole verdict path on RECORDED stamps: c1 (RTC_6d1f29be_*) stands in for C1S and TDR (RTF_03c3e472_*) for
    c31000 at es5-8 (T kinds at es5-6 only: T_LES patched to (1, 2)); a second launch = the same stamp again, so the
    pairs are BITWISE-EQUAL and the pooled lines must equal gates_fresh.json's single-launch lines. Module globals are
    patched for this check only and restored; the doc goes to --scratch."""
    rec = json.loads(GATES_FRESH.read_text())
    c1, tdr = rec["policies"]["new"], rec["policies"]["tdr"]
    saved = {k: getattr(R, k) for k in ("EMAP", "REAL_OK", "T_LES", "unit_name")}
    pfx = {"C1S": "RTC_6d1f29be_", "c1": "RTC_6d1f29be_", "c31000": "RTF_03c3e472_", "TDR": "RTF_03c3e472_"}

    def unit_name(u, pols):
        if u["tag"] == "_R1":
            return "NO_SUCH_ROW_R1"
        return pfx[u["pol"]] + u["rid"].replace("__omni", "")

    bad = []
    try:
        R.EMAP, R.REAL_OK, R.T_LES, R.unit_name = dict(EMAP_FRESH), RF.REAL_FRESH, (1, 2), unit_name
        pols = {"C1S": {"sha": c1, "ck": "-", "role": "teacher", "label": "C1S(=c1)", "placeholder": False},
                "c31000": {"sha": tdr, "ck": "-", "role": "teacher", "label": "c31000(=TDR)", "placeholder": False},
                "c1": {"sha": c1, "ck": "-", "role": "teacher", "label": "c1", "placeholder": False},
                "TDR": {"sha": tdr, "ck": "-", "role": "teacher", "label": "TDR", "placeholder": False}}
        Rr = R.P2ReadRows([R.RT_OUT / "lr", R.P2_OUT / "lr"], {c1, tdr}, set(), emap=EMAP_FRESH,
                          real_ok=RF.REAL_FRESH)
        pin = P3.read_json(R.P2_PIN)
        cache = R.PairCache(scratch / "pairs_e2e.json")
        doc = R.verdict_doc(Rr, pols, pin, {"sha": c1}, cache)
        cache.save()
        M.write_json(scratch / "read_e2e.json", doc)
    finally:
        for k, v in saved.items():
            setattr(R, k, v)
    G, Rc, Rs = rec["G"]["kinds"], rec["R"]["completion"], rec["R"]["solved"]
    for kind in R.T_KINDS:
        got = doc["T"]["kinds"][kind]
        for k in ("new", "tdr", "diff", "se"):
            if abs(got.get(k, 9) - G[kind][k]) > TOL:
                bad.append(f"T {kind} {k} {got.get(k)} != {G[kind][k]}")
        if got.get("gain") != G[kind]["gain"]:
            bad.append(f"T {kind} gain")
    for kind in R.TREG_KINDS:
        got = doc["T_reg"]["kinds"][kind]
        for k in ("diff", "se"):
            if abs(got.get(k, 9) - G[kind][k]) > TOL:
                bad.append(f"T-reg {kind} {k}")
        if got.get("regress") != G[kind]["regress"]:
            bad.append(f"T-reg {kind} regress")
    for lname, (rid, _l, _n) in R.N_ROWS.items():
        got, want = doc["N"]["rows"][lname], Rc[rid]
        for k in ("head", "ref", "se_diff", "margin", "line"):
            if abs(got.get(k, 9) - want[k]) > TOL:
                bad.append(f"N {lname} {k} {got.get(k)} != {want[k]}")
        if got.get("pass") != want["pass"]:
            bad.append(f"N {lname} pass")
    for kind in R.REPORT_KINDS:
        got, want = doc["report"]["kinds"][kind], Rs[kind]
        if abs(got.get("new", 9) - want["head"]) > TOL or abs(got.get("tdr", 9) - want["ref"]) > TOL:
            bad.append(f"report {kind} shares")
    pairs = doc["pairs"]
    if len(pairs) != 12 or not all(p["ok"] and p["verdict"] == "BITWISE-EQUAL" for p in pairs):
        bad.append(f"pairs {len(pairs)} {[p['verdict'] for p in pairs][:4]}")
    q3 = doc["report"]["q3_split"]
    ierr = [abs(q3[k].get("identity_error", 9)) for k in R.T_KINDS]
    if max(ierr) > 1e-12:
        bad.append(f"Q3 identity error {ierr}")
    tr = doc["Tr"]
    if tr.get("status") != "PASS" or tr.get("n_lines") != 18 or (tr.get("E") or {}).get("flags"):
        bad.append(f"Tr {tr.get('status')} {tr.get('n_lines')} {(tr.get('E') or {}).get('flags')}")
    lr = doc["T"]["kinds"]["slopes"].get("launches", {}).get("new") or []
    if not lr or any(not x.get("step0_fp") for x in lr):
        bad.append("per-launch shares / step-0 fingerprints missing beside the T line")
    if doc["status"] != "COMPUTED" or doc["decision"]["route"] != "RECOMMEND_YES":
        bad.append(f"status {doc['status']} route {doc['decision']}")
    vs = doc["T"]["kinds"]["slopes"].get("variant_split") or {}
    ld = doc["T"]["kinds"]["ledge_down"].get("variant_split") or {}
    report("6", not bad, f"end-to-end verdict on recorded stamps (c1 as C1S, TDR as c31000, es5-8): status "
                         f"{doc['status']} route={doc['decision']['route']} T={doc['T']['status']} "
                         f"T-reg={doc['T_reg']['status']} N={doc['N']['status']} Tr={tr.get('status')} "
                         f"pairs={len(pairs)} BITWISE-EQUAL; T / T-reg / N / report lines == gates_fresh.json to 1e-12; "
                         f"Q3 identity err {max(ierr):.1e}; slopes down/up "
                         f"{ {k: (round(v['new'], 3), round(v['ref'], 3)) for k, v in vs.items() if isinstance(v, dict)} }"
                         f"; ledge_down 0.14 m split "
                         f"{ {k: (v['n'], round(v['new'], 3), round(v['ref'], 3)) for k, v in ld.items() if isinstance(v, dict)} }"
                         f"; doc {scratch / 'read_e2e.json'}" + (f" -- {bad}" if bad else ""))
    R.drop_raw_cache()


# ============================================================================== 7
def check7() -> None:
    """The diag label's selection tracking WATCH: read from sel.json's real location
    (blocks.<pick.cand>.watch.tracking.rows), never an absent key; a missing watch is explicit."""
    bad, shown = [], []
    rsd_sel = P3.RSD_SCRIPTS.parent / "results" / "sel.json"
    sd = json.loads(rsd_sel.read_text())
    tw = R.selection_tracking_watch(sd)
    if not isinstance(tw, list) or len(tw) != 4:
        bad.append(f"RSD sel.json: want 4 diag rows, got {tw if not isinstance(tw, list) else len(tw)}")
    else:
        stats = sorted((r["statistic"], r["row"]) for r in tw)
        want = sorted((f"{s} diag", f"suite es{e}") for s in ("ratio", "|ratio-1|") for e in (1, 2))
        if stats != want:
            bad.append(f"rows {stats} != {want}")
        mx = max(r["worse_by"] for r in tw)
        if abs(mx - 0.0091) > 5e-5:
            bad.append(f"max worse_by {mx:.6f} != 0.0091")
        if any(set(r) != set(R.SEL_TW_KEYS) for r in tw):
            bad.append("row keys differ from SEL_TW_KEYS")
        shown.append(f"RSD sel.json -> {len(tw)} diag rows, max worse_by {mx:.4f}")

    def line(stat, wb, fails=True, es=1):
        return {"statistic": stat, "row": f"suite es{es}", "category": "diag", "worse_by": wb, "fails": fails}

    ok_lines = [line("track_err diag", 0.0, fails=False, es=e) for e in (1, 2)]
    # beyond the bound -> student loss, non-empty watch
    lab = R.diag_label({"status": "FAIL", "lines": [line("track_err diag", 0.036)] + ok_lines[1:]}, sd)
    if not (lab and lab.get("applies") and lab["label"].startswith("student")
            and isinstance(lab.get("selection_tracking_watch"), list) and lab["selection_tracking_watch"]):
        bad.append(f"worse_by 0.036 -> {lab}")
    else:
        shown.append("0.036 -> student loss + watch rows")
    # within the bound -> predicted cost, the watch still shown (report only)
    lab = R.diag_label({"status": "FAIL", "lines": [line("track_err diag", 0.035)] + ok_lines[1:]}, sd)
    if not (lab and lab["label"].startswith("predicted") and isinstance(lab.get("selection_tracking_watch"), list)
            and len(lab["selection_tracking_watch"]) == 4):
        bad.append(f"worse_by 0.035 -> {lab}")
    else:
        shown.append("0.035 -> predicted cost + watch rows")
    # a survival diag fail -> student loss
    lab = R.diag_label({"status": "FAIL", "lines": [line("survival diag", 0.01)] + ok_lines}, sd)
    if not (lab and lab["label"].startswith("student")):
        bad.append(f"survival diag -> {lab}")
    # a missing watch is explicit, never []
    for doc, frag in ((None, "no sel.json"), ({"pick": {"cand": "cX"}, "blocks": {}}, "no blocks.cX"),
                      ({"pick": {"cand": "c31000"}, "blocks": {"c31000": {"watch": {}}}}, "has no diag row")):
        tw = R.selection_tracking_watch(doc)
        if not (isinstance(tw, dict) and tw.get("status") == "MISSING" and frag in tw.get("why", "")):
            bad.append(f"missing case {frag!r} -> {tw}")
        lab = R.diag_label({"status": "FAIL", "lines": [line("track_err diag", 0.05)] + ok_lines[1:]}, doc)
        if not (isinstance(lab.get("selection_tracking_watch"), dict)
                and lab["selection_tracking_watch"].get("status") == "MISSING"):
            bad.append(f"label with missing watch {frag!r} -> {lab.get('selection_tracking_watch')}")
    shown.append("3 MISSING cases explicit")
    report("7", not bad, "; ".join(shown) + ("" if not bad else " | BAD: " + "; ".join(bad)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--only", action="append", default=None)
    a = ap.parse_args(argv)
    td = None
    scratch = a.scratch
    if scratch is None:
        td = tempfile.TemporaryDirectory(prefix="p2_read_check_")
        scratch = Path(td.name)
    scratch.mkdir(parents=True, exist_ok=True)
    todo = a.only or ["1", "1b", "2", "3", "4", "5", "6", "7"]
    fns = {"1": check1, "1b": check1b, "2": check2, "3": check3, "4": lambda: check4(scratch),
           "5": lambda: check5(scratch), "6": lambda: check6(scratch), "7": check7}
    for n in todo:
        try:
            fns[n]()
        except Exception as e:  # noqa: BLE001
            import traceback  # noqa: PLC0415
            traceback.print_exc()
            report(n, False, f"crashed: {type(e).__name__}: {e}")
    ok = all(RESULTS.get(n) for n in todo)
    print(f"P2_READ_CHECK {'PASS' if ok else 'FAIL'} " + " ".join(f"{n}={'PASS' if RESULTS.get(n) else 'FAIL'}"
                                                             for n in todo))
    if td:
        td.cleanup()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
