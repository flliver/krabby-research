#!/usr/bin/env python3
"""Amendment A (short arm) SELECTION: ARM step 2 part B, run at step 3 (--check) and step 6 (CPU; saved stamps only).

    select.py [--out JSON]          the pick rule on c1-c4 -> results/short_arm/selection.json (rt_short_selection_v1)
    select.py --dry-run             the same table, printed, nothing written (c3 / c4 may be absent)
    select.py --check               (i) the sweep's 29500 numbers, (ii) step 9's es1 + es2 suite lines: PASS / FAIL

POOL (ARM Design, pre-registered): c1 / c2 = RT_s19 model_29500 / model_30000 (6d1f29be / a346185c, seed 19); c3 / c4 =
RT_s20s model_29500 / model_30000 (results/runs/RT_s20s.json, seed 20). +509 / +1009 its after TDR (510 / 1010 updates).

SCREENING READ (selection seeds es1-2, already spent; never es5+): ``readout_fresh.ScreenRows`` (``readout.Rows`` with
only ``found()`` overridden: the identity map on es1-2, a logical es3 / es4 raises) on a scratch symlink root of every
lr name dir EXCEPT ``RTF_*`` (never listed) and the confirm slots, es5+ (any name dir with a ``seedNNN`` dir at NNN >=
5, directory names only: rt_short.py names the selection rows RTA_* and the confirm rows RTC_*, the ARM text says
"RTA_* confirm slots"; excluding by seed is right under either). Asserted after the read: no served stamp at es > 2,
every served build == results/pin.json.
* G lines: slopes b2 and ledge_down b2 through readout's own pair_tables + g_line with the thresholds file's frozen G
  rule (readout_fresh.g_kinds: readout.gates()'s G loop), gain / regress exactly as gates();
* suite lines: es1 and es2 through gate_read.zero_kg_check (readout_fresh.suite_only: readout.zero_kg's call, drift
  0), the 22 decisive suite lines (rows starting 'suite', not watch); res['verdict'] / res['fails'] never read;
* z_line = (margin - worse_by) / se_diff with the line's own margin['margin'] and se_diff from that call; se_diff == 0
  -> +inf if worse_by <= margin else -inf; a missing line -> -inf; z = min over the 22; the binding line recorded.

PICK RULE (mechanical; ARM "Pick rule"): Q = slopes gains AND ledge_down gains AND neither regresses; E = Q and z > 0.
  tier 1: E members with z >= 1 -> the largest ledge_down diff; ties |delta| <= 0.010 + 1e-9 (<= 2 episodes of 200)
          -> fewer iterations (+509 before +1009), then seed 19;
  tier 2: E non-empty -> the largest z; ties |delta z| <= 0.1 -> fewer iterations, then seed 19;
  tier 3: Q non-empty -> the largest z in Q; same ties; flag "R tracking predicted marginal";
  Q empty -> REFUSE (stop and ask).
A tie set is every candidate within the tolerance of the best value (+inf ties only +inf).

FAIL-CLOSED (refuse loudly, non-zero exit, nothing written): the thresholds sha256 / frozen scripts / R-tracking inputs
guards of readout_fresh (before ``Rows``); c1 / c2 sha256s; every candidate's checkpoint ``iter`` == its file
iteration; RT_s20s's run record (name, ok, rc 0, seed 20, 1010 its, resumed from TDR, final model_30000 = c4); every
screening row read OK (an INCOMPLETE / INVALID row is re-run at the same seed, not skipped); no non-RTF row at
es >= 5 exists (no re-pick after any es5+ number of a candidate); an existing COMMITTED selection.json is never
overwritten.

Sentinels: ``RT_SELECT PICK ...``, ``RT_SELECT REFUSED ...``, ``RT_SELECT CHECK (i)|(ii) PASS|FAIL ...``.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import select as _stdlib_select  # noqa: E402,F401  (the stdlib module, bound BEFORE short_arm joins sys.path: this
#                                   file's name must never shadow it; it is built in to the venv's interpreter anyway.
#                                   Load this file by path (importlib), never as ``import select``.)

HERE = Path(__file__).resolve().parent                    # 2026-11-10_rough_terrain/short_arm
sys.path.insert(0, str(HERE))
import readout_fresh as F  # noqa: E402  (imports scripts/readout.py and scripts/rt_spec.py)

readout, S = F.readout, F.S

# ============================================================================== pre-registered constants (ARM)
RUN19 = Path("/home/nickmagus/krabby/krabby-research/.claude/worktrees/rough-terrain/parkour/logs/rsl_rl/"
             "crab_hex_omni_teacher/2026-11-10_23-43-01_RT_s19")
C1_SHA = "6d1f29be8bf4504fd53fd20018dbab38ff60c6ccb32d7bd885f3ab15eaaed3c1"
C2_SHA = "a346185cd00119a385cbc45654d7ff9e4bf97a2bfbaaa9bfd91242ab8ec71faa"
STEP9_NEW = "d58ceb347f63ab3b51128a471c65be64b5c2a704e27e33e798230aeaabd042f7"   # RT_s19 model_34990
RT_S20S_RECORD = F.RESULTS / "runs" / "RT_s20s.json"
TDR_ITER = 28991
SCREEN_KINDS = ("slopes", "ledge_down")
SCREEN_BAND = "b2"
SCREEN_ES = (1, 2)
TIE_LD = 0.010 + 1e-9
TIE_Z = 0.1 + 1e-12
Z_TIER1 = 1.0
SWEEP = F.RESULTS / "sweep" / "sweep.json"
RULE_TEXT = (
    "ARM 'Pick rule' (pre-registered, commit cb04f86a). G lines from readout.g_line with the frozen G rule (slopes b2, "
    "ledge_down b2, es1-2 vs TDR's RT9 rows). Suite lines from gate_read.zero_kg_check on es1 and es2, drift 0 (vs "
    "TDR's RT6 R_suite rows). z(c) = min over the 22 decisive suite lines (11 per seed) of (margin - worse_by) / "
    "se_diff, each line's own se_diff from the same call (se_diff 0: +inf if worse_by <= margin else -inf; a missing "
    "line: -inf). Q = slopes gains and ledge_down gains and neither regresses. E = Q and z > 0. Tier 1: E members "
    "with z >= 1 -> largest ledge_down diff. Tier 2: E non-empty -> largest z. Tier 3: Q non-empty -> largest z in Q, "
    "flag 'R tracking predicted marginal'. Ties: ledge_down diff within 0.010 (tier 1) or z within 0.1 (tiers 2-3) -> "
    "fewer iterations, then seed 19. Q empty -> stop and ask.")


class SelectRefused(Exception):
    pass


class ScreenRows(F.FreshRows):
    """Selection seeds only: identity on es1-2 (a logical es3 / es4 raises)."""

    EMAP = {1: 1, 2: 2}
    REAL_OK = (1, 2)


# ============================================================================== candidates
def ckpt_iter(path: Path):
    import torch  # noqa: PLC0415

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    return ck.get("iter") if isinstance(ck, dict) else None


def candidates(need_all: bool) -> tuple:
    """[(candidate dict)], [problems]. c3 / c4 absent is a problem only when ``need_all``."""
    cands, probs = [], []
    for cid, it, sha in (("c1", 29500, C1_SHA), ("c2", 30000, C2_SHA)):
        p = RUN19 / f"model_{it}.pt"
        cands.append({"id": cid, "run": "RT_s19", "seed": 19, "file_iter": it, "its_after_tdr": it - TDR_ITER,
                      "updates": it - TDR_ITER + 1, "ckpt": str(p), "want_sha256": sha})
    rec = readout.read_json(RT_S20S_RECORD)
    if rec is None:                                         # refused unless --dry-run (need_all False)
        probs.append(f"{RT_S20S_RECORD} missing: c3 / c4 do not exist yet")
    else:
        cli = rec.get("cli") or {}
        fin = rec.get("final") or {}
        rp = []
        if rec.get("name") != "RT_s20s":
            rp.append(f"name {rec.get('name')!r}")
        if rec.get("status") != "ok" or rec.get("rc") != 0:
            rp.append(f"status {rec.get('status')!r} rc {rec.get('rc')!r}")
        if cli.get("seed") != 20 or cli.get("iterations") != 1010:
            rp.append(f"cli seed {cli.get('seed')} iterations {cli.get('iterations')} (want 20 / 1010)")
        tdr_ck = S.OMNI / "policy" / "rsd_9_p3b_rodspeed_dr" / "model_28991.pt"
        if not cli.get("checkpoint") or Path(cli["checkpoint"]).resolve() != tdr_ck.resolve():
            rp.append(f"resumed from {cli.get('checkpoint')!r}, not TDR model_28991")
        if fin.get("iter") != 30000:
            rp.append(f"final iter {fin.get('iter')} != 30000")
        run_dir = Path(rec.get("run_dir") or "/nonexistent")
        if not run_dir.is_dir():
            rp.append(f"run_dir {run_dir} missing")
        if rp:
            probs.append(f"RT_s20s run record: {rp}")
        else:
            for cid, it in (("c3", 29500), ("c4", 30000)):
                cands.append({"id": cid, "run": "RT_s20s", "seed": 20, "file_iter": it,
                              "its_after_tdr": it - TDR_ITER, "updates": it - TDR_ITER + 1,
                              "ckpt": str(run_dir / f"model_{it}.pt"),
                              "want_sha256": fin.get("sha256") if it == 30000 else None})
    ok = []
    for c in cands:
        p = Path(c["ckpt"])
        if not p.is_file():
            probs.append(f"{c['id']}: {p} missing")
            continue
        c["sha256"] = F.sha256(p)
        if c["want_sha256"] and c["sha256"] != c["want_sha256"]:
            probs.append(f"{c['id']}: sha256 {c['sha256'][:8]} != {c['want_sha256'][:8]}")
            continue
        c["iter"] = ckpt_iter(p)
        if c["iter"] != c["file_iter"]:
            probs.append(f"{c['id']}: checkpoint iter {c['iter']} != {c['file_iter']}")
            continue
        ok.append(c)
    shas = [c["sha256"] for c in ok]
    if len(set(shas)) != len(shas) or readout.TDR_SHA in shas:
        probs.append(f"candidate sha256s not distinct from each other and from TDR: {[s[:8] for s in shas]}")
    return ok, probs


# ============================================================================== per-candidate read
def z_of(lines: list, expected: list) -> dict:
    per, have = [], {(x.get("row"), x.get("statistic")): x for x in lines}
    for key in expected:
        x = have.get(key)
        if x is None or x.get("worse_by") is None or not isinstance(x.get("margin"), dict):
            per.append({"row": key[0], "statistic": key[1], "z": -math.inf, "why": "missing line"})
            continue
        m, w, se = float(x["margin"]["margin"]), float(x["worse_by"]), float(x.get("se_diff") or 0.0)
        if not all(math.isfinite(v) for v in (m, w, se)):   # a NaN would make min() order-dependent: refuse
            raise ValueError(f"non-finite suite line {key}: margin {m} worse_by {w} se_diff {se}")
        if se == 0.0:
            z = math.inf if w <= m else -math.inf
        else:
            z = (m - w) / se
        per.append({"row": key[0], "statistic": key[1], "z": z, "margin": m, "worse_by": w, "se_diff": se,
                    "slack": m - w, "fails": bool(x.get("fails"))})
    extra = sorted(set(have) - set(expected))
    for key in extra:                                       # an unexpected decisive line still binds (fail closed)
        x = have[key]
        m, w, se = float(x["margin"]["margin"]), float(x["worse_by"]), float(x.get("se_diff") or 0.0)
        if not all(math.isfinite(v) for v in (m, w, se)):
            raise ValueError(f"non-finite suite line {key}: margin {m} worse_by {w} se_diff {se}")
        z = (math.inf if w <= m else -math.inf) if se == 0.0 else (m - w) / se
        per.append({"row": key[0], "statistic": key[1], "z": z, "margin": m, "worse_by": w, "se_diff": se,
                    "slack": m - w, "fails": bool(x.get("fails")), "unexpected": True})
    zmin = min(p["z"] for p in per) if per else -math.inf
    bind = next((p for p in per if p["z"] == zmin), None)
    return {"z": zmin, "binding": {k: bind[k] for k in ("row", "statistic")} if bind else None, "lines": per,
            "n_lines": len(per), "unexpected": [list(k) for k in extra]}


def read_candidate(R, c: dict, thr: dict) -> dict:
    G = F.g_kinds(R, c["sha256"], readout.TDR_SHA, thr, kinds=SCREEN_KINDS)
    for k in SCREEN_KINDS:
        if thr["kinds"][k]["status"] != "gated" or thr["kinds"][k]["gate_band"] != SCREEN_BAND:
            raise SelectRefused(f"thresholds kind {k}: status {thr['kinds'][k]['status']} band "
                                f"{thr['kinds'][k].get('gate_band')} (the arm screens gated {SCREEN_BAND})")
    trk = F.suite_only(R, c["sha256"], readout.TDR_SHA, SCREEN_ES)
    expected = [(f"suite es{es}", st) for es in SCREEN_ES for st in F.SUITE_STATS]
    zz = z_of(trk.get("lines") or [], expected)
    lines = {k: G["kinds"].get(k) or {"status": "MISSING"} for k in SCREEN_KINDS}
    ok = all(v.get("status") == "OK" for v in lines.values()) and trk["status"] in ("PASS", "FAIL")
    sl, ld = lines["slopes"], lines["ledge_down"]
    q = bool(ok and sl.get("gain") and ld.get("gain") and not sl.get("regress") and not ld.get("regress"))
    return {**{k: c[k] for k in ("id", "run", "seed", "file_iter", "its_after_tdr", "updates", "ckpt", "sha256",
                                 "iter")},
            "read_ok": ok, "G": lines, "suite": trk, "z": zz["z"], "z_binding": zz["binding"], "z_lines": zz["lines"],
            "z_unexpected_lines": zz["unexpected"], "ledge_down_diff": ld.get("diff"), "slopes_diff": sl.get("diff"),
            "Q": q, "E": bool(q and zz["z"] > 0)}


# ============================================================================== the pick
def _order(c: dict) -> tuple:
    return (c["its_after_tdr"], 0 if c["seed"] == 19 else 1, c["id"])


def _ties(pool: list, key: str, tol: float) -> list:
    best = max(c[key] for c in pool)
    if math.isinf(best):
        return [c for c in pool if c[key] == best]
    return [c for c in pool if c[key] >= best - tol]


def pick(rows: list) -> dict:
    Q = [c for c in rows if c["Q"]]
    E = [c for c in Q if c["E"]]
    t1 = [c for c in E if c["z"] >= Z_TIER1]
    if t1:
        tier, pool, key, tol = 1, t1, "ledge_down_diff", TIE_LD
    elif E:
        tier, pool, key, tol = 2, E, "z", TIE_Z
    elif Q:
        tier, pool, key, tol = 3, Q, "z", TIE_Z
    else:
        return {"tier": None, "Q": [], "E": [], "refuse": "Q is empty: no candidate gains on both slopes and "
                                                           "ledge_down without a regression -- stop and ask"}
    tied = _ties(pool, key, tol)
    win = sorted(tied, key=_order)[0]
    reason = (f"tier {tier}: " + {1: f"E members with z >= {Z_TIER1:g}: {[c['id'] for c in t1]}; largest ledge_down "
                                     f"diff", 2: f"E = {[c['id'] for c in E]}, none with z >= {Z_TIER1:g}; largest z",
                                  3: f"E empty, Q = {[c['id'] for c in Q]}; largest z in Q"}[tier]
              + f"; best {key} {max(c[key] for c in pool):.6g}; tie set (within {tol:g}) "
              f"{[c['id'] for c in tied]} -> {win['id']} (fewer iterations, then seed 19)")
    return {"tier": tier, "Q": [c["id"] for c in Q], "E": [c["id"] for c in E], "tier1": [c["id"] for c in t1],
            "tie_key": key, "tie_tolerance": tol, "tie_set": [c["id"] for c in tied], "winner": win,
            "reason": reason, "flag": "R tracking predicted marginal" if tier == 3 else None}


# ============================================================================== shared run
def screen(roots, cands: list, thr: dict, pin: dict) -> tuple:
    shas = {c["sha256"] for c in cands} | {readout.TDR_SHA}
    R = ScreenRows(roots, dup="earliest", shas=shas)
    rows = [read_candidate(R, c, thr) for c in cands]
    sc = F.served_checks(R, pin, tdr_from_rtf=False)
    hi = [s for s in sc["served"] if s["real_es"] > 2]
    if hi:
        sc["ok"] = False
        sc["problems"].append(f"served stamps at es > 2: {hi[:3]}")
    excl = [s for s in sc["served"] if s.get("name_dir") and s["name_dir"].startswith(F.EXCLUDED_PREFIXES)]
    if excl:
        sc["ok"] = False
        sc["problems"].append(f"served an excluded (RTF) row: {excl[:3]}")
    return rows, sc


def fmt(x, n=4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    return f"{x:+.{n}f}" if isinstance(x, float) else str(x)


def table(rows: list) -> str:
    out = ["id   run      +its  slopes diff (gain/reg)   ledge_down diff (gain/reg)   z       binding line          "
           "Q E read"]
    for r in rows:
        sl, ld = r["G"]["slopes"], r["G"]["ledge_down"]
        b = r["z_binding"] or {}
        out.append(f"{r['id']:4s} {r['run']:8s} {r['its_after_tdr']:+5d}  {fmt(sl.get('diff'), 3):>7s} "
                   f"({str(sl.get('gain'))[0]}/{str(sl.get('regress'))[0]})          {fmt(ld.get('diff'), 3):>7s} "
                   f"({str(ld.get('gain'))[0]}/{str(ld.get('regress'))[0]})              {fmt(r['z'], 3):>7s} "
                   f"{(b.get('statistic', '-') + ' ' + b.get('row', '')).strip():22s} {int(r['Q'])} {int(r['E'])} "
                   f"{'OK' if r['read_ok'] else 'INCOMPLETE ' + str([sl.get('status'), ld.get('status'), r['suite']['status']])}")
    return "\n".join(out)


def _json_safe(x):
    """+-inf -> the strings '+inf' / '-inf' (selection.json stays strict JSON)."""
    if isinstance(x, float) and math.isinf(x):
        return "+inf" if x > 0 else "-inf"
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x


def select(out_path: Path, dry: bool, scratch: Path | None) -> int:
    if not dry and out_path.exists():
        cc = F.committed_clean(out_path) if out_path.resolve() == F.SELECTION.resolve() else {"tracked": False}
        if cc.get("tracked"):
            raise SelectRefused(f"{out_path} is committed: a selection is never re-picked or overwritten")
    conf = F.fresh_seed_name_dirs()                          # non-RTF name dirs at es >= 5: the confirm slots
    if conf:
        msg = (f"rows at env seed >= {F.FRESH_SEED_MIN} outside RTF exist ({conf[:4]}): es5+ numbers of a candidate "
               "exist, no re-pick (ARM 'Never')")
        if not dry:
            raise SelectRefused(msg)
        print(f"RT_SELECT WARNING {msg}")
    g = {"thresholds": F.guard_thresholds(), "r_inputs": F.guard_r_inputs()}
    if not all(v["ok"] for v in g.values()):
        raise SelectRefused("guards: " + "; ".join(p for v in g.values() for p in v.get("problems") or []))
    pin = F.load_pin()
    g["pin"] = {"ok": True, "code_sha": pin["code_sha"], "eval_sha": pin["eval_sha"]}
    thr = readout.load_thresholds(readout.THRESHOLDS)
    cands, cprobs = candidates(need_all=not dry)
    if cprobs and not dry:
        raise SelectRefused("candidates: " + "; ".join(cprobs))
    for p in cprobs:
        print(f"RT_SELECT DRY note: {p}")
    td, root, linked, skipped = F.masked_root(scratch)
    try:
        rows, sc = screen([root], cands, thr, pin)
        rows = F.subst_prefix(rows, str(root), str(readout.RT_LR))      # record the lr paths, not the scratch links
        for s in sc["served"]:
            if s.get("stamp_as_read"):
                s["stamp_as_read"] = s["stamp_as_read"].replace(str(root), str(readout.RT_LR))
    finally:
        td.cleanup()
    print(table(rows))
    if not sc["ok"]:
        raise SelectRefused("served stamps: " + "; ".join(sc["problems"][:6]))
    incomplete = [r["id"] for r in rows if not r["read_ok"]]
    if incomplete and not dry:
        raise SelectRefused(f"screening rows not all OK for {incomplete}: re-run the failed slots at the same seeds")
    if dry and (incomplete or cprobs):
        print(f"RT_SELECT DRY pick n/a: pool incomplete (absent / incomplete: {incomplete} {cprobs[:2]})")
        return 0
    pk = pick(rows)
    if pk["tier"] is None:
        raise SelectRefused(pk["refuse"])
    w = pk["winner"]
    print(f"RT_SELECT PICK {w['id']} tier {pk['tier']} sha {w['sha256'][:8]} z {fmt(w['z'], 3)} "
          f"ledge_down {fmt(w['ledge_down_diff'], 3)} Q={pk['Q']} E={pk['E']}"
          + (f" FLAG '{pk['flag']}'" if pk["flag"] else "") + f" :: {pk['reason']}")
    if dry:
        print("RT_SELECT DRY nothing written")
        return 0
    doc = {"schema": F.SELECTION_SCHEMA, "created": datetime.datetime.now().isoformat(timespec="seconds"),
           "arm": "short_arm/ARM.md (Amendment A, commit cb04f86a)", "rule": RULE_TEXT,
           "tie_tolerances": {"ledge_down_diff": TIE_LD, "z": TIE_Z, "z_tier1": Z_TIER1},
           "tdr": {"sha256": readout.TDR_SHA}, "screen": {"kinds": [f"{k} {SCREEN_BAND}" for k in SCREEN_KINDS],
                                                         "env_seeds": list(SCREEN_ES), "suite_env_seeds": list(SCREEN_ES)},
           "candidates": rows, "Q": pk["Q"], "E": pk["E"], "tier1_members": pk["tier1"],
           "pick": {"id": w["id"], "ckpt": w["ckpt"], "sha256": w["sha256"], "run": w["run"], "seed": w["seed"],
                    "file_iter": w["file_iter"], "its_after_tdr": w["its_after_tdr"], "tier": pk["tier"],
                    "reason": pk["reason"], "flag": pk["flag"], "tie_set": pk["tie_set"]},
           "served_stamps": [{"path": s["stamp"], "sha256": s["sha256"], "es": s["real_es"], "rid": s["rid"],
                              "status": s["status"]} for s in sc["served"]],
           "served_checks": {k: v for k, v in sc.items() if k != "served"},
           "root": {"excluded_prefixes": list(F.EXCLUDED_PREFIXES), "excluded_es_min": F.FRESH_SEED_MIN,
                    "linked": len(linked), "excluded": len(skipped), "excluded_names": skipped},
           "guards": g,
           "sources": {"select.py": F.sha256(Path(__file__)), **F.sources()}}
    readout.write_json(out_path, _json_safe(readout._norm(doc)))
    print(f"RT_SELECT wrote {out_path}")
    return 0


# ============================================================================== --check
def _dp(s: str) -> int:
    return len(s.split(".", 1)[1]) if "." in s else 0


def _at_precision(stored: str, got: float, slack: float = 0.0) -> tuple:
    """``got`` equals the stored decimal string to the stored precision (|got - stored| <= half its last digit +
    ``slack``)."""
    tol = 0.5 * 10 ** -_dp(stored) + slack + 1e-12
    return abs(float(got) - float(stored)) <= tol, tol


def check(scratch: Path | None) -> int:
    sw = json.loads(SWEEP.read_text())
    r1, r0 = sw["rows"][1], sw["rows"][0]
    if str(r1.get("ckpt")) != "29500" or str(r0.get("ckpt")) != "TDR":
        print(f"RT_SELECT CHECK (i) FAIL sweep rows[1] / rows[0] are {r1.get('ckpt')} / {r0.get('ckpt')}")
        return 1
    txt, rec_txt = sw["paired_vs_tdr"], sw["reconcile_agree"]
    raw = json.loads(SWEEP.read_text(), parse_float=str)["rows"]   # stored decimal strings (precision)
    pats = {
        "slopes": r"SLOPES b2: 29500 \+([0-9.]+) \(SE ([0-9.]+), lb ([0-9.]+), (\d+) wins / (\d+) losses\) (GAIN)",
        "ledge_down": r"LEDGE_DOWN b2: 29500 \+([0-9.]+) \(SE ([0-9.]+), lb ([0-9.]+), (\d+)/(\d+)\) (GAIN)",
        "back": r"Back track_err \(R ([0-9.]+)\): 29500 \+([0-9.]+) \(SE",
        "yaw": r"Turn yaw_progress \(R ([0-9.]+)\): 29500 \+([0-9.]+) \(SE",
        "back6": r"29500 back worse_by is ([0-9.]+)",
        "yaw6": r"29500 yaw worse_by is ([0-9.]+)"}
    mt = {}
    for k, p in pats.items():
        m = re.search(p, rec_txt if k.endswith("6") else txt)
        if not m:
            print(f"RT_SELECT CHECK (i) FAIL sweep.json text has no '{k}' 29500 entry (pattern {p})")
            return 1
        mt[k] = m.groups()
    g = {"thresholds": F.guard_thresholds(), "r_inputs": F.guard_r_inputs()}
    if not all(v["ok"] for v in g.values()):
        print("RT_SELECT CHECK guards FAIL: " + "; ".join(p for v in g.values() for p in v.get("problems") or []))
        return 1
    pin = F.load_pin()
    thr = readout.load_thresholds(readout.THRESHOLDS)
    c1 = RUN19 / "model_29500.pt"
    if F.sha256(c1) != C1_SHA:
        print("RT_SELECT CHECK (i) FAIL c1 sha256")
        return 1
    td, root, _, _ = F.masked_root(scratch)
    try:
        R = ScreenRows([root], dup="earliest", shas={C1_SHA, STEP9_NEW, readout.TDR_SHA})
        G = F.g_kinds(R, C1_SHA, readout.TDR_SHA, thr, kinds=SCREEN_KINDS)
        s1 = F.suite_only(R, C1_SHA, readout.TDR_SHA, (1,))
        s9 = F.suite_only(R, STEP9_NEW, readout.TDR_SHA, (1, 2))
        sc = F.served_checks(R, pin, tdr_from_rtf=False)
    finally:
        td.cleanup()
    items = []

    def item(name, stored, got, slack=0.0, exact=False):
        if exact:
            ok = stored == got
            items.append((name, ok, f"stored {stored!r} got {got!r}"))
        else:
            ok, tol = _at_precision(stored, got, slack)
            items.append((name, ok, f"stored {stored} got {got!r} tol {tol:.2g}"))

    for kind, (d, se, lb, w, l_, gtxt) in (("slopes", mt["slopes"]), ("ledge_down", mt["ledge_down"])):
        ln = G["kinds"][kind]
        if ln.get("status") != "OK":
            items.append((f"{kind} read", False, f"status {ln.get('status')} {ln.get('why')}"))
            continue
        key = "slopes_success" if kind == "slopes" else "ledge_down_success"
        item(f"{kind} new success (rows[1].{key})", raw[1][key], ln["new"])
        item(f"{kind} TDR success (rows[0].{key})", raw[0][key], ln["tdr"])
        item(f"{kind} diff", d, ln["diff"])
        # the sweep's SE / lb are a paired BOOTSTRAP (4000 resamples, seed 12345; its text: "the analytic paired SE
        # agrees to within 0.0005"); readout's SE is thresholds.paired_completion_se. Held to the stored 3 dp anyway
        # (no slack): a miss here is reported, and that bootstrap-vs-analytic note is the first thing to read
        item(f"{kind} SE (stored bootstrap, 3 dp)", se, ln["se"])
        item(f"{kind} CI lower bound (stored bootstrap, 3 dp)", lb, ln["ci_lb"])
        item(f"{kind} wins/losses", f"{w}/{l_}", f"{ln['paired_wins_losses']['new_only']}/"
                                                  f"{ln['paired_wins_losses']['tdr_only']}", exact=True)
        item(f"{kind} gain/regress", (gtxt == "GAIN", False), (bool(ln["gain"]), bool(ln["regress"])), exact=True)
    by = {(x["row"], x["statistic"]): x for x in s1.get("lines") or []}
    for nm, stat, (Rv, wb4), wb6, hk, pk in (
            ("back", "track_err back", mt["back"], mt["back6"][0], "track_err_back", None),
            ("yaw", "yaw_progress turn", mt["yaw"], mt["yaw6"][0], None, "yaw_progress_turn")):
        x = by.get(("suite es1", stat))
        if x is None:
            items.append((f"suite es1 {stat}", False, f"line missing (suite status {s1.get('status')})"))
            continue
        item(f"suite es1 {stat} worse_by (6 dp)", wb6, x["worse_by"])
        item(f"suite es1 {stat} worse_by (4 dp)", wb4, x["worse_by"])
        item(f"suite es1 {stat} R", Rv, x["margin"]["margin"])
        if hk:
            item(f"suite es1 {stat} head |r-1| (rows[1].{hk})", raw[1][hk], x["head"])
            item(f"suite es1 {stat} TDR |r-1| (rows[0].{hk})", raw[0][hk], x["p2a"])
        if pk:
            item(f"suite es1 {stat} head (rows[1].{pk})", raw[1][pk], x["head"])
            item(f"suite es1 {stat} TDR (rows[0].{pk})", raw[0][pk], x["p2a"])
    z1 = z_of(s1.get("lines") or [], [("suite es1", st) for st in F.SUITE_STATS])
    print(f"  (i) report: c1 es1-only z {fmt(z1['z'], 3)} binding {z1['binding']} (ARM: c1 in Q minus E, z ~ -0.4)")
    ok_i = all(o for _, o, _ in items) and sc["ok"]
    for name, o, det in items:
        print(f"  (i) {'ok ' if o else 'BAD'} {name}: {det}")
    print(f"RT_SELECT CHECK (i) {'PASS' if ok_i else 'FAIL'} sweep 29500 vs recomputed: {sum(o for _, o, _ in items)}"
          f"/{len(items)} items" + ("" if sc["ok"] else f"; served problems {sc['problems'][:3]}"))
    rec = json.loads(F.STEP9_GATES.read_text())
    if rec["policies"]["new"] != STEP9_NEW:
        print("RT_SELECT CHECK (ii) FAIL gates.json policies.new is not d58ceb34")
        return 1
    want = rec["R"]["tracking"]["suite_lines"]
    got = readout._norm(s9.get("lines") or [])
    wk = {(x["row"], x["statistic"]): x for x in want}
    gk = {(x["row"], x["statistic"]): x for x in got}
    bad = []
    if set(wk) != set(gk) or len(got) != len(want):
        bad.append(f"line sets differ: {sorted(set(wk) ^ set(gk))[:6]} ({len(got)} vs {len(want)})")
    for k in sorted(set(wk) & set(gk)):
        if wk[k] != gk[k]:
            dk = sorted(f for f in set(wk[k]) | set(gk[k]) if wk[k].get(f) != gk[k].get(f))
            bad.append(f"{k}: fields differ {dk}")
    zz = z_of(got, [(f"suite es{es}", st) for es in (1, 2) for st in F.SUITE_STATS])
    ok_ii = not bad and s9.get("status") in ("PASS", "FAIL") and len(got) == 22
    for b in bad[:12]:
        print(f"  (ii) BAD {b}")
    print(f"RT_SELECT CHECK (ii) {'PASS' if ok_ii else 'FAIL'} step 9 suite lines es1+es2 (d58ceb34 vs TDR): "
          f"{len(got)} recomputed vs {len(want)} recorded, exact on every field; suite status {s9.get('status')} "
          f"fails {s9.get('fails')}; z {fmt(zz['z'], 3)} binding {zz['binding']}")
    return 0 if (ok_i and ok_ii) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--check", action="store_true")
    m.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=F.SELECTION)
    ap.add_argument("--scratch", type=Path, default=None, help="where the masked symlink lr root is made")
    a = ap.parse_args(argv)
    try:
        if a.check:
            return check(a.scratch)
        return select(a.out, a.dry_run, a.scratch)
    except (SelectRefused, F.FreshRefused, F.FreshSeedError, readout.KeySetError) as e:
        print(f"RT_SELECT REFUSED {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
