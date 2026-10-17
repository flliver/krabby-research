#!/usr/bin/env python3
"""PLAN step 8 (Phase C preflight) READER (CPU): every check's pass rule (PL/preflight/STEPC_RULES.md), the STOP-C
decision and the values step 8 feeds into thresholds v1 / pipeline.yaml.

    t4_stepC_read.py --check NAME      one check (the queue's gate): writes SC/checks/NAME.json and exits
                                       0 PASS | 2 REPORT (report-only finding; the queue continues) | 3 FAIL | 4 INCOMPLETE
    t4_stepC_read.py [--json OUT] [--md OUT]
                                       every check -> PL/preflight/stepC_read.{json,md}: per-check verdicts, STOP-C
                                       (yes / no / pending), the derived values (N_p, x_rule, the manifest-shape split,
                                       the live-rule validations, the pilot-based payload rates, the C2 re-anchor list,
                                       the sweep anchors, the measured costs) and the patches they imply.
Read-only on every record; it never launches anything. Statistics are the TP tools' own (gate_read / payload_read /
live_watch / replay_read / thresholds), reached through t4_stepC_lib.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t4_stepC_lib as C  # noqa: E402

PASS, REPORT, FAIL, INCOMPLETE = "PASS", "REPORT", "FAIL", "INCOMPLETE"
RC = {PASS: 0, REPORT: 2, FAIL: 3, INCOMPLETE: 4}
CHECKS_DIR = C.SC / "checks"
#: pinned tolerances (STEPC_RULES)
TOL = {"mass_kg": 1e-3, "com_m": 1e-4, "eig_rel": 1e-3, "axis_deg": 1.0, "cover_share": 0.95, "sees_share": 0.95,
       "sum_force": 0.03, "h_per_1k": 4.0, "depth_floor": 0.05, "field_floor": 0.05}
STOP_CHECKS = ("C1", "C3_readback", "C3_settle", "C3_S1_curve", "C3_pilot_live", "C3_np", "C3_gate", "C4_camera",
               "C4_readback", "C4_parity", "C4_timing_check", "C4_depth", "C4_live", "C4_distill_payload_check",
               "C6_new", "C7")
REPORT_CHECKS = ("C2", "C4_det", "C5", "C6_det")


class Incomplete(Exception):
    pass


def need(x, what):
    if x is None:
        raise Incomplete(f"missing: {what}")
    return x


def rstamp(name, fam_or_sc, es, ckpt=None):
    sc = fam_or_sc if "__" in fam_or_sc else C.ctx().spec.d["rows"][fam_or_sc]["scenario"]
    return C.row_stamp(name, sc, es, ckpt)


# ================================================================================================ C1
def chk_C1() -> dict:
    out, bad, inc = {}, [], []
    # 1. cfg dumps vs the M1a reference (root-normalised)
    ref, new = C.SC / "dump_ref", C.OUT / "dumps"
    d = {"ref": str(ref), "new": str(new), "targets": {}}
    if not (new / ".complete").is_file():
        inc.append("C1_dumps")
    elif not ref.is_dir():
        d["note"] = "no reference dumps (SC/dump_ref missing): compare skipped -- REPORT"
    else:
        src = (ref / "SOURCE.txt").read_text() if (ref / "SOURCE.txt").is_file() else ""
        old_root = src.split("dump root ")[-1].strip() if "dump root " in src else None
        for f in sorted(ref.glob("*.json")):
            a = f.read_text()
            b = (new / f.name).read_text() if (new / f.name).is_file() else None
            if old_root:
                a = a.replace(old_root, str(C.R))
            same = b is not None and json.loads(a) == json.loads(b)
            pay = b is not None and "payload_reset" in b
            d["targets"][f.stem] = {"identical": same, "payload_term_present": pay}
            if not same or pay:
                bad.append(f"dump {f.stem}: {'differs' if not same else 'carries payload_reset'}")
    out["dumps"] = d
    # 2. smokes: P-config x6, P-early (s1 vs X43), bitwise vs A2, armed 0:0 vs off
    runs = {st: C.driver_head(f"PC_{st}_s18") for st in C.STAGES6}
    pc = {}
    for st, h in runs.items():
        if h is None:
            inc.append(f"C1_PC_{st}")
            continue
        r = C.RR().pconfig(Path(h["ckpt"]).parent, C.LINEAGE_RUN[st])
        pc[st] = {"pass": r["pass"], "not_allowed": [x["key"] for f in r["files"].values() for x in f["not_allowed"]][:12]}
        if not r["pass"]:
            bad.append(f"P-config {st}: {pc[st]['not_allowed'][:4]}")
    out["pconfig"] = pc
    if runs.get("s1"):
        s1 = Path(runs["s1"]["ckpt"]).parent
        pe = C.RR().pearly(s1, C.X43_RUN, range(20))
        out["pearly_vs_X43"] = {k: pe[k] for k in ("pass", "design_bitwise", "differ", "shared", "bitwise")}
        if not pe["pass"]:
            bad.append("P-early vs X43")
        a2 = C.a2_smoke_runs().get("s1")
        if a2 is None:
            inc.append("A2 PF_s1 run dir (preflight/orchestrator.log)")
        else:
            ser = C.RR().pearly_series(C.RR().scalars(a2), C.RR().scalars(s1), list(range(20)))
            out["s1_vs_A2"] = {"pass": ser["pass"], "strict_bitwise": not ser["differ"], "differ": ser["differ"],
                               "design_bitwise": ser["design_bitwise"]}
            if not ser["pass"]:
                bad.append("post-build s1 smoke vs A2")
        hp = C.driver_head("PC_s1pay00_s18")
        if hp is None:
            inc.append("C1_PC_s1pay00")
        else:
            pr = Path(hp["ckpt"]).parent
            sa, sb = C.RR().scalars(s1), C.RR().scalars(pr)
            ser = C.RR().pearly_series(sa, sb, list(range(20)))
            kg = sb.get("Payload/kg_mean") or {}
            env_yaml = (pr / "params/env.yaml").read_text() if (pr / "params/env.yaml").is_file() else ""
            out["armed00_vs_off"] = {"pass": ser["pass"], "strict_bitwise": not ser["differ"], "differ": ser["differ"],
                                     "only_armed_tags": ser["only_new"], "payload_term_in_env_yaml":
                                     "payload_reset" in env_yaml, "kg_mean_max": max(kg.values()) if kg else None}
            if not ser["pass"]:
                bad.append("armed 0:0 smoke vs off (shared tags)")
            if "payload_reset" not in env_yaml:
                bad.append("the armed 0:0 smoke carries no payload_reset term (not armed?)")
            if kg and max(abs(v) for v in kg.values()) > 0:
                bad.append("Payload/kg_mean != 0 on the 0:0 smoke")
    else:
        inc.append("C1_PC_s1")
    # 3. rows, bitwise
    pairs = [("C1_S1_suite", "T4C1_S1", "suite", "REFS_suite_es1", None),
             ("C1_S1_field_hard", "T4C1_S1", "field_hard", "REF_field_hard_es1", None),
             ("C1_S1_suite_body", "T4C1_S1body", "suite_body", "ANC_suite_body_es1", None),
             ("C1_X43_suite", "T4C1_X43", "suite", "X43_suite_es1", None),
             ("C1_X60_suite", "T4C1_X60", "suite", "X60_suite_es1", None),
             ("C1_X10_suite", "T4C1_X10", "suite", "X10_suite_es1", None),
             ("C1_S1_suite_L000", "T4C1_S1L000", "suite", "REFS_suite_es1", 0),
             ("C1_fwd3a_old", "T4C1_fwd3a_old", "slow__A15pB", "FWD3A_slow_es1", None)]
    rows = {}
    for jn, nm, fam, refk, lvl in pairs:
        st = rstamp(nm, fam, 1)
        if st is None:
            inc.append(jn)
            continue
        fwd = jn == "C1_fwd3a_old"
        b = C.bitwise(C.REF[refk], st, allow_new_keys=(C.FWD_NEW_RAW_KEYS if fwd else ()), ref_keys_only=fwd)
        rows[jn] = {"stamp": str(st), **{k: b.get(k) for k in ("bitwise", "raw_bitwise", "metrics_equal_whole",
                                                               "only_metadata_differs", "n_value_mismatch",
                                                               "aggregate_keys_differ", "new_only_aggregate_keys",
                                                               "allowed_new_raw_keys")}}
        if lvl is not None:
            pp = C.GR().payload_problems(C.read_json(st / "run_meta.json", {}), float(lvl))
            rows[jn]["payload_problems"] = pp
            if pp:
                bad.append(f"{jn}: realised payload {pp[:2]}")
        if not b["bitwise"]:
            bad.append(f"{jn} not bitwise vs {refk}")
    cur = rstamp("T4C1_fwd3a_cur", "slow__A15pB", 1)
    if cur is not None:
        a = C.read_json(cur / "scenario_metrics.json", {}).get("aggregate") or {}
        rows["C1_fwd3a_cur"] = {"stamp": str(cur), "completion": a.get("schedule_completion_rate"),
                                "n": a.get("n_episodes"), "expected": "0.75 / 25 falls (FWD policy README)",
                                "split_vs_recorded_0.79": (None if a.get("schedule_completion_rate") is None
                                                           else abs(0.79 - a["schedule_completion_rate"]))}
    out["rows"] = rows
    return verdict(out, bad, inc)


# ================================================================================================ C2
C2_REF = {"suite@1": "REFS_suite_es1", "side22@1": "P2_side22_es1", "field_hard@1": "REF_field_hard_es1",
          "field_light@1": "REF_field_light_es1", "ring_obst@1": "REF_ring_obst_es1",
          "field_turn@1": "REF_field_turn_es1", "field_suite@1": "REFS_field_suite_es1",
          "suite_body@1": "ANC_suite_body_es1", "turnlevels_body@1": "ANC_turnlevels_body_es1",
          "grid_body@1": "ANC_grid_body_es1"}


def chk_C2() -> dict:
    out, differ, inc = {"families": {}}, [], []
    for key, refk in C2_REF.items():
        fam, es = key.split("@")
        st = rstamp("T4C2_S1", fam, int(es), C.S1_CK)
        if st is None:
            inc.append(f"C2_S1_{key}")
            continue
        b = C.bitwise(C.REF[refk], st)
        out["families"][fam] = {"ref": refk, "stamp": str(st), **{k: b.get(k) for k in (
            "bitwise", "raw_bitwise", "only_metadata_differs", "n_value_mismatch", "aggregate_keys_differ")}}
        if not b["bitwise"]:
            differ.append(fam)
    st = rstamp("T4C2_X60", "turn_band", 1, C.X60_C6X_CK)
    if st is None:
        inc.append("C2_X60_turn_band@1")
    else:
        b = C.bitwise(C.REF["TB_c6x_es1"], st)
        out["families"]["turn_band"] = {"ref": "TB_c6x_es1 (X60 c6x)", "stamp": str(st), "bitwise": b["bitwise"],
                                        "aggregate_keys_differ": b.get("aggregate_keys_differ")}
        if not b["bitwise"]:
            differ.append("turn_band")
    for key, refk in (("suite@1", "REFS_suite_es1"), ("walkstop@1", "REFS_walkstop_es1")):
        fam, es = key.split("@")
        sc = C.ctx().spec.d["rows"][fam]["scenario"]
        st = C.p2_stamp("T4C2_REFS_S1", sc, int(es), C.S1_SHA)
        if st is None:
            inc.append(f"C2_REFS_{key}")
            continue
        b = C.bitwise(C.REF[refk], st)
        out["families"][f"REFS:{fam}"] = {"ref": refk, "stamp": str(st), "bitwise": b["bitwise"]}
        if not b["bitwise"]:
            differ.append(f"REFS:{fam}")
    for nm, sc, refk in (("T4C2_SH_T17", "hwalkstop__omni", "SH_T17_hwalkstop_es1"),
                         ("T4C2_SH_T17", "hraise__omni", "SH_T17_hraise_es1"),
                         ("T4C2_SH_T17z", "walkstop__omni", "SH_T17z_walkstop_es1")):
        st = C.p2_stamp(nm, sc, 1)
        if st is None:
            inc.append(f"{nm} {sc}")
            continue
        b = C.bitwise(C.REF[refk], st, raw=False)
        fam = {"hwalkstop__omni": "hwalkstop", "hraise__omni": "hraise", "walkstop__omni": "walkstop(SH zero)"}[sc]
        out["families"][fam] = {"ref": refk, "stamp": str(st), "aggregate_equal": b["metrics_equal_stripped"],
                                "aggregate_keys_differ": b.get("aggregate_keys_differ")}
        if not b["metrics_equal_stripped"]:
            differ.append(fam)
    out["reanchor"] = sorted(set(differ))
    out["reanchor_procedure"] = "STEPC_RULES C2.3 (before the v1 freeze, at the pin)"
    out["dirsweep_body"] = "no recorded stamp: anchored by C5 (report-only)"
    if inc:
        out["incomplete"] = inc
        return {"verdict": INCOMPLETE if not out["families"] else REPORT, **out}
    return {"verdict": REPORT if differ else PASS, **out}


# ================================================================================================ C3
def readback_rules(js: dict, mode: str, level_kg=None) -> list:
    bad = []
    if not js.get("term_present"):
        return ["payload term absent (the knobs did not arm the event)"]
    w = js.get("worst") or {}
    if w.get("mass_err_kg", 1) > TOL["mass_kg"]:
        bad.append(f"mass error {w.get('mass_err_kg')} kg > {TOL['mass_kg']}")
    if w.get("com_err_m", 1) > TOL["com_m"]:
        bad.append(f"CoM error {w.get('com_err_m')} m > {TOL['com_m']}")
    if w.get("eig_rel_err", 1) > TOL["eig_rel"]:
        bad.append(f"principal-moment rel. error {w.get('eig_rel_err')} > {TOL['eig_rel']}")
    if not js.get("zero_kg_exact"):
        bad.append("a 0-kg env is not bitwise its captured base")
    if not js.get("untouched_envs_bitwise"):
        bad.append("a partial reset changed an env outside the reset subset")
    if js.get("out_of_range_draws"):
        bad.append(f"{js['out_of_range_draws']} draws outside kg_range")
    if mode == "train":
        t = js.get("transitions") or {}
        if not (t.get("loaded_to_zero") and t.get("zero_to_loaded")):
            bad.append(f"partial resets did not exercise loaded->0 and 0->loaded ({t})")
    if mode == "level":
        lv = js.get("level") or {}
        if not (lv.get("every_env_at_L") and lv.get("com_exact")):
            bad.append(f"level row env not every env at {level_kg} kg / CoM (0, 0, z): {lv}")
    return bad


def chk_C3_readback() -> dict:
    out, bad, inc = {}, [], []
    for nm, mode, lvl in (("train", "train", None), ("level90", "level", 90)):
        js = C.read_json(C.OUT / f"readback/{nm}.json", {}) or {}
        if js.get("complete") is not True:
            inc.append(f"readback/{nm}.json")
            continue
        b = readback_rules(js, mode, lvl)
        out[nm] = {"worst": js.get("worst"), "transitions": js.get("transitions"), "zero_share": js.get("zero_share"),
                   "base_mass_spread_kg": js.get("base_mass_spread_kg"), "problems": b}
        bad += [f"{nm}: {x}" for x in b]
    return verdict(out, bad, inc)


def settle_read(st: Path, kg: int) -> dict:
    sm = C.read_json(st / "scenario_metrics.json", {})
    meta = C.read_json(st / "run_meta.json", {})
    eps = sm.get("episodes") or []
    term = [e for e in eps if e.get("termination_reason") != "schedule_complete"]
    act = C.PR().actuator_load(st, float(kg))
    tilt, sag = [], []
    import numpy as np  # noqa: PLC0415

    for z in sorted((st / "raw").glob("episode_*.npz")):
        a = np.load(z)
        qq = a["root_quat_w"].astype(float)
        up = 1.0 - 2.0 * (qq[:, 1] ** 2 + qq[:, 2] ** 2)
        tilt.append(float(np.degrees(np.arccos(np.clip(up, -1, 1))).max()))
        h = a["root_pos_w"][:, 2].astype(float)
        k = max(1, len(h) // 10)
        sag.append(float(h[:k].mean() - h[-k:].mean()))
    return {"stamp": str(st), "n": len(eps), "terminated": len(term), "tips": len(term) > 0,
            "termination_reasons": (sm.get("aggregate") or {}).get("termination_reasons"),
            "sum_foot_force_ratio": act.get("sum_foot_force_ratio"), "max_tilt_deg": max(tilt) if tilt else None,
            "mean_sag_m": (sum(sag) / len(sag)) if sag else None,
            "payload_problems": C.GR().payload_problems(meta, float(kg))}


def chk_C3_settle() -> dict:
    out, bad, inc = {"levels": {}}, [], []
    for kg in C.LEVELS:
        st = C.adhoc_stamp(f"T4C3_settle_L{kg:03d}")
        if st is None:
            inc.append(f"C3_settle_L{kg:03d}")
            continue
        r = settle_read(st, kg)
        out["levels"][kg] = r
        if r["payload_problems"]:
            bad.append(f"{kg} kg: realised payload {r['payload_problems'][:2]}")
        if r["tips"] and kg <= 90:
            bad.append(f"settle TIPS at {kg} kg (<= 90): {r['terminated']}/{r['n']} episodes not schedule_complete")
    # CHARTER step-8 pin (orchestrator, 2026-10-16): Sigma foot force is read RELATIVE to the 0-kg settle's own ratio.
    # A3 measured a steady -2.4..-2.7 % offset at 0-180 kg (foot-contact sensing), so an absolute 1 +- 0.03 band
    # would stop on the sensor offset, not on a missing payload; the realised mass is checked directly above.
    f0 = (out["levels"].get(0) or {}).get("sum_foot_force_ratio")
    out["sum_force_ref_0kg"] = f0
    for kg, r in out["levels"].items():
        f = r["sum_foot_force_ratio"]
        rel = (f / f0) if (f is not None and f0) else None
        r["sum_foot_force_rel_0kg"] = rel
        if not r["tips"] and (rel is None or abs(rel - 1.0) > TOL["sum_force"]):
            bad.append(f"{kg} kg: sum foot force ratio {f} / 0-kg ratio {f0} = {rel} (outside 1 +- {TOL['sum_force']})")
    out["tips_above_90"] = [kg for kg, r in out["levels"].items() if r["tips"] and kg > 90]
    return verdict(out, bad, inc)


def curve_read(prefix: str, ckpt, stage="p2a") -> dict:
    curve, reads = {}, {}
    for kg in C.LEVELS:
        nm = C.curve_names(prefix, kg)
        stamps = {k: rstamp(nm, k.split("@")[0], int(k.split("@")[1]), ckpt) for k in C.LEVEL_ROWS}
        if any(v is None for v in stamps.values()):
            break
        r = C.level_read(stamps)
        pp = {k: C.GR().payload_problems(C.read_json(v / "run_meta.json", {}), float(kg)) for k, v in stamps.items()}
        reads[kg] = {"stable": r["stable"], "failed": r["failed"], "payload_problems": {k: v for k, v in pp.items() if v},
                     "lines": {x["id"]: x["value"] for x in r["lines"]}, "stamps": {k: str(v) for k, v in stamps.items()}}
        curve[kg] = r["stable"]
        if not r["stable"]:
            break
    return {"curve": curve, "max_stable": C.PR().max_stable(curve), "reads": reads,
            "first_unstable": next((k for k in sorted(curve) if not curve[k]), None),
            "complete": bool(curve) and (not all(curve.values()) or len(curve) == len(C.LEVELS))}


def chk_C3_S1_curve() -> dict:
    cr = curve_read("T4C3_S1", C.S1_CK)
    out = {"S1_curve": cr, "label": C.PR().LABEL}
    bad, inc = [], []
    if not cr["complete"]:
        inc.append("S1 curve rows")
    if 0 in cr["curve"] and not cr["curve"][0]:
        bad.append(f"S1 at 0 kg (armed 0:0) FAILS the a-priori stable-at-L lines: {cr['reads'][0]['failed'][:6]}")
    for kg, r in cr["reads"].items():
        if r["payload_problems"]:
            bad.append(f"S1 curve {kg} kg: realised payload {r['payload_problems']}")
    return verdict(out, bad, inc)


def live_driver_replay(run_dir: Path, source_run: Path, stage="p3a") -> dict:
    LW = C.LW()
    ser, src = LW.read_series(run_dir), LW.read_series(source_run)
    cfg = C.ctx().spec.d.get("live") or {}
    rep = LW.replay("driver", ser, every=50, stage=stage, source=src, cfg=cfg)
    first = LW.first_step(ser)
    tail = LW.tail_mean(src.get(LW.T_EPLEN))
    el = LW._since(ser.get(LW.T_EPLEN), first + 300) if first is not None else []
    cf = LW._since(ser.get(LW.T_CRAB), first + 300) if first is not None else []
    roll = int((cfg.get("collapse") or {}).get("roll", 200))

    def rolls(s):
        v = [x for _, x in s]
        return [sum(v[max(0, i - roll + 1):i + 1]) / len(v[max(0, i - roll + 1):i + 1]) for i in range(len(v))]

    rl, rc = rolls(el), rolls(cf)
    return {"replay": rep, "first": first, "last": LW.last_step(ser), "source_tail_ep_len": tail,
            "min_ep_len_over_kill_line": (min(rl) / (0.5 * tail)) if (rl and tail) else None,
            "max_crab_failure_roll": max(rc) if rc else None,
            "resume_read": LW.judge_resume(ser, src, cfg.get("resume"))}


def chk_C3_pilot_live() -> dict:
    rd = need(C.pilot_run(), "the pilot run (PP_p3a_s18)")
    r = live_driver_replay(rd, C.S1_RUN)
    bad = []
    if r["replay"].get("killed_at") is not None:
        bad.append(f"the live rules KILL the healthy payload pilot at +{r['replay'].get('rel')}: {r['replay'].get('fails')}")
    if r["resume_read"].get("rc") == C.LW().RC_KILL:
        bad.append(f"resume rule fails on the pilot: {r['resume_read'].get('fails')}")
    return verdict({"pilot_run": str(rd), **r, "note": "calibration under load: margins reported (unvalidated rule)"},
                   bad, [])


def chk_C3_np() -> dict:
    rd = need(C.pilot_run(), "the pilot run")
    reads, inc, bad, per = {}, [], [], {}
    for off in C.PILOT_OFFSETS:
        c = C.pilot_ckpt_at(off)
        nm = f"T4C3_pilot_c{c['iter']}_L090"
        st = [rstamp(nm, "suite", es, c["ckpt"]) for es in (1, 2)]
        if any(s is None for s in st):
            inc.append(f"{nm} suite es1-2")
            continue
        for s in st:
            pp = C.GR().payload_problems(C.read_json(s / "run_meta.json", {}), 90.0)
            if pp:
                bad.append(f"{nm}: realised payload {pp[:2]}")
        reads[off] = C.suite90_read(st)
        per[off] = {"iter": c["iter"], "ckpt": str(c["ckpt"]), "read": reads[off]}
    if inc:
        return verdict({"reads": per}, bad, inc)
    rule = C.np_rule(reads, C.th_doc()["R_used"])
    out = {"N_p": rule["N_p"], "rule": rule, "reads": per, "pilot_run": str(rd),
           "budget_bound_at_3000": C.PR().budget_bound(C.LW().read_series(rd, ("Episode_Termination/crab_failure",
                                                                              "Metrics/base_parkour/track_ratio_mean")))}
    if rule["N_p"] is None:
        bad.append("N_p undetermined")
    return verdict(out, bad, [])


def gate_values(prefix: str, ckpt, its_past: int) -> dict:
    """The full p3a gate (pipeline.gate_payload's reads, on an explicit stamp set) of a payload head ``prefix``."""
    th = C.thresholds()
    trained = C.ctx().plan["stages"]["p3a"]["trained"]
    tt = [c for c in trained if c in C.TRANSLATE]
    head = {k: rstamp(prefix, k.split("@")[0], int(k.split("@")[1]), ckpt) for k in C.ZERO_KG_ROWS}
    s1 = {k: rstamp("T4C2_S1", k.split("@")[0], int(k.split("@")[1]), C.S1_CK) for k in C.ZERO_KG_ROWS}
    miss = [f"head {k}" for k, v in head.items() if v is None] + [f"S1 {k}" for k, v in s1.items() if v is None]
    if miss:
        raise Incomplete(f"missing rows: {miss[:8]}")
    pp = {k: C.GR().payload_problems(C.read_json(v / "run_meta.json", {}), None) for k, v in head.items()}
    floors = C.GR().walker_decisive("p3a", {k: head[k] for k in ("suite@1", "suite@2", "side22@1")}, trained, th,
                                    th_stage="p3a")

    def pack(d):
        o = {"suite": {1: d["suite@1"], 2: d["suite@2"]}, "side22": d["side22@1"]}
        for row in ("field_hard", "field_light", "ring_obst", "field_turn"):
            o[row] = [d[f"{row}@1"], d[f"{row}@2"]]
        return o

    rf = list((C.TL().load_yaml(C.PL / "threshold_inputs.yaml").get("resolution_files") or []))
    zk = C.GR().zero_kg_check(pack(head), pack(s1), tt, th, its_past, rf)
    cr = curve_read(prefix, ckpt, "p3a")
    return {"floors": floors, "zero_kg": zk, "curve": cr, "payload_problems_0kg": {k: v for k, v in pp.items() if v},
            "trained": trained, "its_past_p2a": its_past}


def apriori_rates(cr: dict, its_note: str) -> dict:
    """PLAN Rule 'payload a-priori lines': each p3a.stable90.* line's chance rate with the PILOT's 90-kg value as the
    mean and variance 2 sigma_hat(p2a)^2 + SE^2 (thresholds.payload_apriori_rate)."""
    TH = C.TH()
    r90 = (cr.get("reads") or {}).get(90)
    if r90 is None:
        return {"available": False, "why": "no stable 90-kg read on the pilot"}
    lines = {x["id"]: x for x in C.th_doc()["lines"] if x["id"].startswith("p3a.stable90.")}
    st = r90["stamps"]
    cats = {es: TH.suite_categories(Path(st[f"suite@{es}"])) for es in (1, 2)}
    eps = {es: TH.suite_episodes(Path(st[f"suite@{es}"])) for es in (1, 2)}
    out, rates = {}, []
    for lid, ln in lines.items():
        mu = r90["lines"].get(lid)
        if mu is None:
            continue
        se = None
        parts = lid.split(".")
        if parts[2] == "survival":
            es = int(parts[-1][2:])
            n = (cats[es].get(parts[3]) or {}).get("n") or 100
            se = math.sqrt(max(mu * (1 - mu), 1e-6) / n)
        elif parts[2] in ("tracking", "turn_fc"):
            es = int(parts[-1][2:])
            cat = parts[3] if parts[2] == "tracking" else "turn"
            key = "ratio" if parts[2] == "tracking" else "facing_closed"
            se = C.boot_se_median([e[key] for e in eps[es] if e["cat"] == cat])
        elif lid.endswith("field_light_completion"):
            se = math.sqrt(max(mu * (1 - mu), 1e-6) / 200.0)
        else:
            se = None
        sig = ln.get("sigma_hat_p2a") or 0.0
        hib = ln["direction"] == ">="
        rate = TH.payload_apriori_rate(ln["line"], mu, sig, se if se is not None else sig, hib)
        out[lid] = {"line": ln["line"], "mu_pilot": mu, "se": se, "se_stand_in": se is None, "sigma_hat_p2a": sig,
                    "rate": rate}
        rates.append(rate)
    return {"available": True, "lines": out, "sum": TH.total(rates), "union": TH.union(rates), "note": its_note}


def chk_C3_gate() -> dict:
    npr = need((C.read_json(CHECKS_DIR / "C3_np.json", {}) or {}).get("N_p"), "N_p (C3_np)")
    c = C.pilot_ckpt_at(int(npr))
    gv = gate_values(f"T4C3_pilot_c{c['iter']}", c["ckpt"], c["iter"] - C.S1_ITER)
    cr = gv["curve"]
    if not cr["complete"]:
        raise Incomplete("the pilot's curve rows")
    bad = []
    fl = [d for d in gv["floors"] if not d.get("pass")]
    if fl:
        bad.append(f"p3a floors fail on the pilot: {[d['id'] for d in fl][:6]}")
    ms = cr["max_stable"]
    if ms is None or ms < 90:
        bad.append(f"pilot NOT stable at 90 kg (max_stable {ms}; first unstable {cr['first_unstable']}: "
                   f"{(cr['reads'].get(cr['first_unstable']) or {}).get('failed', [])[:6]})")
    if gv["zero_kg"].get("verdict") != "PASS":
        bad.append(f"the paired 0-kg check FAILS vs S1: {gv['zero_kg'].get('fails', [])[:6]}")
    if gv["payload_problems_0kg"]:
        bad.append(f"0-kg rows carry a payload: {gv['payload_problems_0kg']}")
    for kg, r in cr["reads"].items():
        if r["payload_problems"]:
            bad.append(f"level {kg}: realised payload {r['payload_problems']}")
    TH = C.TH()

    def mval(z):
        m = z.get("margin")
        return m.get("margin") if isinstance(m, dict) else m

    zrows = [z for z in (gv["zero_kg"].get("rows") or []) if not z.get("watch") and z.get("se_diff") is not None
             and mval(z) is not None]
    per = [{"statistic": z.get("statistic"), "row": z.get("row"), "se_diff": z["se_diff"], "margin": mval(z),
            "binds": (z.get("margin") or {}).get("binds") if isinstance(z.get("margin"), dict) else None,
            "margin_used": z.get("margin_used"), "rate": TH.paired_rate(mval(z), z["se_diff"])} for z in zrows]
    out = {"N_p": npr, "ckpt": str(c["ckpt"]), "iter": c["iter"], "floors": gv["floors"], "max_stable": ms,
           "curve": cr["curve"], "first_unstable": cr["first_unstable"], "zero_kg": gv["zero_kg"],
           "v1_rates": {"stable90": apriori_rates(cr, f"pilot model_{c['iter']} at N_p={npr}"),
                        "zero_kg_paired": {"per_statistic": per, "sum": TH.total(p["rate"] for p in per),
                                           "max_margin_used": gv["zero_kg"].get("max_margin_used")}},
           "label": C.PR().LABEL}
    if cr["first_unstable"] is not None:
        try:
            ser = C.LW().read_series(C.pilot_run(), ("Episode_Termination/crab_failure",
                                                     "Metrics/base_parkour/track_ratio_mean"))
            fu = cr["first_unstable"]
            lr = C.level_read({k: Path(v) for k, v in cr["reads"][fu]["stamps"].items()})
            out["limiting_factor"] = C.PR().limiting_factor(lr, budget=C.PR().budget_bound(ser))
        except Exception as exc:  # noqa: BLE001
            out["limiting_factor"] = {"unreadable": repr(exc)[:200]}
    return verdict(out, bad, [])


# ================================================================================================ C4
def chk_C4_camera() -> dict:
    out, bad, inc = {}, [], []
    for mode in ("flat", "field"):
        js = C.read_json(C.OUT / f"camera/{mode}.json", {}) or {}
        if js.get("complete") is not True:
            inc.append(f"camera/{mode}.json")
            continue
        out[mode] = js.get("cameras")
        for n, v in (js.get("cameras") or {}).items():
            for k in ("axis_quat_deg_max", "axis_rays_deg_max"):
                if v.get(k) is None or v[k] > TOL["axis_deg"]:
                    bad.append(f"{mode} {n}: {k} {v.get(k)} > {TOL['axis_deg']} deg")
            if mode == "flat":
                if (v.get("cover_share") or 0) < TOL["cover_share"]:
                    bad.append(f"{n}: ground band covers target {v.get('target_4b')} in only "
                               f"{v.get('cover_share')} of envs (median band {v.get('band_lo_median')}..{v.get('band_hi_median')})")
                if (v.get("sees_something_share") or 0) < TOL["sees_share"]:
                    bad.append(f"{n}: depth image empty (all at the clip) in {1 - (v.get('sees_something_share') or 0):.2f} of envs")
        if mode == "field":
            tot = sum((v.get("block_envs") or 0) for v in (js.get("cameras") or {}).values())
            out["field_block_envs_total"] = tot
            if tot < 1:
                bad.append("no ray of any camera hit a field block (>= 5 rays off the ground in one env)")
    return verdict(out, bad, inc)


def chk_C4_readback() -> dict:
    js = C.read_json(C.OUT / "readback/student.json", {}) or {}
    if js.get("complete") is not True:
        return verdict({}, [], ["readback/student.json"])
    b = readback_rules(js, "train")
    return verdict({"student_env": {"worst": js.get("worst"), "transitions": js.get("transitions"),
                                    "zero_share": js.get("zero_share"), "params": js.get("params")}},
                   b, [])


def suite_band(a: Path, b: Path, trained) -> dict:
    TH = C.TH()
    R = C.th_doc()["R_used"]
    ca, cb = TH.suite_categories(Path(a)), TH.suite_categories(Path(b))
    res, bad = {}, []
    for c in [x for x in trained if x in C.TRANSLATE] + ["turn", "move_turn"]:
        x, y = ca.get(c) or {}, cb.get(c) or {}
        if x.get("survival") is None or y.get("survival") is None:
            bad.append(f"{c}: survival missing")
            continue
        pbar = 0.5 * (x["survival"] + y["survival"])
        band = max(R.get(f"cat.{c}.survival", 0.0), TH.binomial_guard_floor(pbar, int(x.get("n") or 100)))
        d = abs(x["survival"] - y["survival"])
        res[f"{c}.survival"] = {"walk_play": x["survival"], "student_play": y["survival"], "delta": d, "band": band}
        if d > band:
            bad.append(f"{c} survival {x['survival']:.3f} vs {y['survival']:.3f} (band {band:.3f})")
        if c in C.TRANSLATE:
            if x.get("ratio") is None or y.get("ratio") is None:
                bad.append(f"{c}: ratio missing")
                continue
            d = abs(x["ratio"] - y["ratio"])
            band = R.get(f"cat.{c}.ratio")
            res[f"{c}.ratio"] = {"walk_play": x["ratio"], "student_play": y["ratio"], "delta": d, "band": band}
            if band is None or d > band:
                bad.append(f"{c} ratio {x['ratio']:.3f} vs {y['ratio']:.3f} (band {band})")
    x, y = (ca.get("turn") or {}).get("facing_closed"), (cb.get("turn") or {}).get("facing_closed")
    if x is not None and y is not None:
        band = R.get("cat.turn.facing_closed")
        res["turn.facing_closed"] = {"walk_play": x, "student_play": y, "delta": abs(x - y), "band": band}
        if abs(x - y) > band:
            bad.append(f"turn facing_closed {x:.3f} vs {y:.3f} (band {band:.3f})")
    return {"stats": res, "problems": bad}


def chk_C4_parity() -> dict:
    TH = C.TH()
    trained = C.ctx().plan["stages"]["p2a"]["trained"]
    refused = set(C.P().student_refused_keys())
    out, bad, inc = {}, [], []
    for key in ("suite@1", "side22@1", "field_hard@1", "field_hard@2", "field_light@1"):
        fam, es = key.split("@")
        a = rstamp("T4C2_S1", fam, int(es), C.S1_CK)
        b = rstamp("T4C4_S1_SP", fam, int(es), C.S1_CK)
        if a is None or b is None:
            inc.append(key)
            continue
        meta = C.read_json(b / "run_meta.json", {})
        rec = {"walk_play": str(a), "student_play": str(b), "task": meta.get("task"),
               "refused_keys_present": sorted(k for k in (meta.get("env_vars") or {}) if k in refused)}
        if meta.get("task") != "Isaac-Crab-Hex-Omni-Student-Play-v0":
            bad.append(f"{key}: Student-Play row ran task {meta.get('task')}")
        if rec["refused_keys_present"]:
            bad.append(f"{key}: refused runner knobs in the Student-Play env {rec['refused_keys_present']}")
        bw = C.bitwise(a, b)
        rec["bitwise"] = bw["bitwise"]
        if not bw["bitwise"]:
            if fam == "suite":
                s = suite_band(a, b, trained)
                rec.update(s)
                bad += [f"{key}: {x}" for x in s["problems"]]
            elif fam == "side22":
                ra, rb = C.GR().side22(a)["class"], C.GR().side22(b)["class"]
                rec["class"] = {"walk_play": ra, "student_play": rb}
                if list(ra) != list(rb):
                    bad.append(f"side22 class {ra} vs {rb}")
            else:
                ea, eb = TH.field_episodes(a), TH.field_episodes(b)
                pa, pb = sum(ea.values()) / len(ea), sum(eb.values()) / len(eb)
                se = TH.paired_completion_se(ea, eb)
                band = max(TOL["field_floor"], 2 * se)
                rec.update({"walk_play": pa, "student_play": pb, "delta": abs(pa - pb), "band": band})
                if abs(pa - pb) > band:
                    bad.append(f"{key}: completion {pa:.3f} vs {pb:.3f} (band {band:.3f})")
        out[key] = rec
    return verdict(out, bad, inc)


def chk_C4_timing_check() -> dict:
    rd = need(C.student_run("T4C4_timing"), "the timing distill run")
    t = C.s_per_it(rd, "Loss_depth/depth_actor_loss")
    if t.get("h_per_1k") is None:
        raise Incomplete(f"timing unreadable: {t}")
    bad = [] if t["h_per_1k"] <= TOL["h_per_1k"] else [f"{t['h_per_1k']:.2f} h per 1k its > {TOL['h_per_1k']} (STOP)"]
    t["projected_5000_its_h"] = t["h_per_1k"] * 5
    t["forward_K1_h_per_1k"] = 1.97
    return verdict({"run": str(rd), **t}, bad, [])


def chk_C4_det() -> dict:
    a, b = C.student_run("T4C4_det_a"), C.student_run("T4C4_det_b")
    if a is None or b is None:
        return {"verdict": INCOMPLETE, "why": "det runs missing"}
    tb = C.tf_bitwise(a, b)
    fa = C.student_final("T4C4_det_a", C.S1_ITER, 50)
    fb = C.student_final("T4C4_det_b", C.S1_ITER, 50)
    sha = (C.sha256(fa) == C.sha256(fb)) if (fa and fb) else None
    det = bool(tb["bitwise"] and sha)
    return {"verdict": PASS if det else REPORT, "deterministic": det, "tags": tb, "final_sha_equal": sha,
            "meaning": "bitwise -> a same-seed student rerun is a replay (the no-retry rationale holds for the "
                       "student too); else a rerun is a fresh draw (report-only)"}


def field_eps(stamps: list) -> dict:
    TH = C.TH()
    out = {}
    for i, s in enumerate(stamps):
        out.update({(i, k): v for k, v in TH.field_episodes(Path(s)).items()})
    return out


def chk_C4_depth() -> dict:
    TH = C.TH()
    dk = need(C.student_final("T4C4_pilot_depth", C.S1_ITER, 1000), "depth pilot head")
    zk = need(C.student_final("T4C4_pilot_zero", C.S1_ITER, 1000), "zero-depth pilot head")
    ds = [need(rstamp("T4C4_pdepth", "field_hard", es, dk), f"depth fh es{es}") for es in (1, 2)]
    zs = [need(rstamp("T4C4_pzero", "field_hard", es, zk), f"zero fh es{es}") for es in (1, 2)]
    a, b = field_eps(ds), field_eps(zs)
    pa, pb = sum(a.values()) / len(a), sum(b.values()) / len(b)
    se = TH.paired_completion_se(a, b)
    line = max(TOL["depth_floor"], 2 * se)
    out = {"depth_completion": pa, "zero_completion": pb, "diff": pa - pb, "se_diff": se, "required": line,
           "depth_helps": (pa - pb) >= line}
    bad = [] if out["depth_helps"] else [f"depth - zeroed field_hard es1-2 = {pa - pb:+.3f} < max(0.05, 2 SE) = {line:.3f}"]
    # the manifest-shape split (STEPC_RULES C4.5), on the depth pilot
    sp = {}
    fa = rstamp("T4C4_pdepth", "field_hard", 1, dk)
    fb = C.row_stamp("T4C4_pdepth_shape", "field_hard__omni_noblk", 1, dk)
    sa = rstamp("T4C4_pdepth", "suite", 1, dk)
    sb = C.row_stamp("T4C4_pdepth_shape", "suite__omni_blk", 1, dk)
    if fa and fb:
        x = C.read_json(fa / "scenario_metrics.json", {})["aggregate"]["schedule_completion_rate"]
        y = C.read_json(fb / "scenario_metrics.json", {})["aggregate"]["schedule_completion_rate"]
        sp["field_hard_es1"] = {"block": x, "no_block": y, "abs": abs(x - y), "bitwise": C.bitwise(fa, fb)["raw_bitwise"]}
    if sa and sb:
        x = C.read_json(sa / "scenario_metrics.json", {})["aggregate"].get("schedule_completion_rate")
        y = C.read_json(sb / "scenario_metrics.json", {})["aggregate"].get("schedule_completion_rate")
        if x is not None and y is not None:
            sp["suite_es1"] = {"no_block": x, "block": y, "abs": abs(x - y), "bitwise": C.bitwise(sa, sb)["raw_bitwise"]}
    if sp:
        v = max(r["abs"] for r in sp.values())
        out["manifest_shape_split"] = {"rows": sp, "value": math.ceil(v * 100 - 1e-9) / 100.0,
                                       "low_power": pa < 0.2,
                                       "binds": "never below the 0.05 floor" if v <= 0.05 else "widens the margin"}
    else:
        out["manifest_shape_split"] = {"value": None, "why": "shape rows missing"}
    return verdict(out, bad, [])


def student_replays(run_dir: Path, tstar_run: Path) -> dict:
    LW = C.LW()
    ser = LW.read_series(run_dir)
    ts = LW.read_series(tstar_run, (LW.T_EPLEN, LW.T_CRAB))
    tail = {"ep_len": LW.tail_mean(ts.get(LW.T_EPLEN)), "crab_failure": LW.tail_mean(ts.get(LW.T_CRAB))}
    base = dict((C.ctx().spec.d.get("live") or {}).get("student") or {})
    loss_only = LW.replay("student", ser, every=50, cfg={"student": base}, tstar_tail=None)
    no_loss = LW.replay("student", ser, every=50, cfg={"student": {**base, "loss_frac": 1e12}}, tstar_tail=tail)
    full = LW.replay("student", ser, every=50, cfg={"student": base}, tstar_tail=tail)
    return {"run": str(run_dir), "tstar_tail": tail, "loss_rule": loss_only, "ep_len_crab_rules": no_loss,
            "all_rules": full}


def chk_C4_live() -> dict:
    dr = need(C.student_run("T4C4_pilot_depth"), "depth pilot run")
    out = {"depth_pilot": student_replays(dr, C.S1_RUN)}
    zr = C.student_run("T4C4_pilot_zero")
    if zr:
        out["zero_pilot"] = student_replays(zr, C.S1_RUN)
    out["forward_3a"] = student_replays(C.FWD_3A_RUN, C.FWD_2C_RUN)
    bad = []
    if out["depth_pilot"]["ep_len_crab_rules"].get("killed_at") is not None:
        bad.append(f"the episode-length / crab_failure rules KILL the depth pilot at "
                   f"+{out['depth_pilot']['ep_len_crab_rules'].get('rel')}: {out['depth_pilot']['ep_len_crab_rules'].get('fails')}")
    loss_fails = [k for k in ("depth_pilot", "forward_3a") if out[k]["loss_rule"].get("killed_at") is not None]
    out["v1_student_loss_rule"] = "watch" if loss_fails else "kill"
    out["loss_rule_failed_on"] = loss_fails
    out["zero_pilot_as_known_bad"] = ((out.get("zero_pilot") or {}).get("all_rules") or {}).get("killed_at")
    return verdict(out, bad, [])


def chk_C4_distill_payload_check() -> dict:
    npr = need((C.read_json(CHECKS_DIR / "C3_np.json", {}) or {}).get("N_p"), "N_p")
    c = C.pilot_ckpt_at(int(npr))
    rd = need(C.student_run("T4C4_dpay"), "the payload distill run")
    fin = need(C.student_final("T4C4_dpay", c["iter"], 200), "its final checkpoint")
    ser = C.LW().read_series(rd, ("Payload/kg_mean", "Payload/loaded_frac"))
    kg = [v for _, v in ser.get("Payload/kg_mean") or []]
    lf = [v for _, v in ser.get("Payload/loaded_frac") or []]
    bad = []
    if not kg or not lf:
        bad.append("the student training env logs no Payload/* tags (payload not armed in the distill env)")
    p = C.ctx().spec.d["payload"]["training"]
    hi = float(str(C.ctx().spec.overlay("p3a")["KRABBY_PAYLOAD_KG"]).split(":")[1])
    exp_kg = (1 - float(p["KRABBY_PAYLOAD_ZERO_FRAC"])) * hi / 2.0
    exp_lf = 1 - float(p["KRABBY_PAYLOAD_ZERO_FRAC"])
    mk = sum(kg[-50:]) / len(kg[-50:]) if kg else None
    ml = sum(lf[-50:]) / len(lf[-50:]) if lf else None
    if mk is not None and abs(mk - exp_kg) > 0.25 * exp_kg:
        bad.append(f"Payload/kg_mean {mk:.1f} vs expected {exp_kg:.1f} (+-25 %)")
    if ml is not None and abs(ml - exp_lf) > 0.10:
        bad.append(f"Payload/loaded_frac {ml:.2f} vs expected {exp_lf:.2f} (+-0.10)")
    st = rstamp("T4C4_dpay_L090", "suite", 1, fin)
    row = None
    if st is None:
        bad.append("the 90-kg Student-Play row is missing")
    else:
        meta = C.read_json(st / "run_meta.json", {})
        pp = C.GR().payload_problems(meta, 90.0)
        if pp:
            bad.append(f"90-kg row realised payload: {pp[:2]}")
        if meta.get("task") != "Isaac-Crab-Hex-Omni-Student-Play-v0":
            bad.append(f"90-kg row task {meta.get('task')}")
        cs = C.TH().suite_categories(st)
        row = {"stamp": str(st), "survival": {k: (v or {}).get("survival") for k, v in cs.items()
                                               if k in C.TRANSLATE}}
    return verdict({"run": str(rd), "final": str(fin), "kg_mean_last50": mk, "loaded_frac_last50": ml,
                    "expected": {"kg_mean": exp_kg, "loaded_frac": exp_lf}, "row_90kg": row,
                    "timing": C.s_per_it(rd, "Loss_depth/depth_actor_loss")}, bad, [])


# ================================================================================================ C5 / C6 / C7
def chk_C5() -> dict:
    out, inc = {}, []
    sc = C.ctx().spec.d["rows"]["dirsweep_body"]["scenario"]
    for seed, ck in ((16, C.X10_CK), (17, C.X11_CK), (18, C.S1_CK)):
        for es in (1, 2):
            st = C.row_stamp(f"T4C5_s{seed}", sc, es, ck)
            if st is None:
                inc.append(f"s{seed} es{es}")
                continue
            out[f"s{seed}_es{es}"] = {"stamp": str(st),
                                      "watch": C.GR().watch_reads("p2a", {f"dirsweep_body@{es}": st},
                                                                  C.ctx().spec.d["rows"])}
    return {"verdict": REPORT if inc else PASS, "anchors": out, "incomplete": inc,
            "v1_threshold_inputs": {"sweep_anchors": {k: v["stamp"] for k, v in out.items()}}}


def smoke_ok(root: Path, name: str) -> dict:
    d = root / "runs" / name
    sm = C.read_json(d / "smoke.json", {}) or {}
    env = C.read_json(d / "params/krabby_env.json", {}) or {}
    ey = (d / "params/env.yaml").read_text() if (d / "params/env.yaml").is_file() else ""
    return {"run": str(d), "verdict": sm.get("verdict"), "payload_keys": sorted(k for k in env if k.startswith("KRABBY_PAYLOAD_")),
            "payload_term": "payload_reset" in ey, "final": (sm.get("run") or {}).get("final_checkpoint")}


def chk_C6_new() -> dict:
    import t4_stepC_jobs as J  # noqa: PLC0415

    emit = C.OUT / "stand/emit"
    out, bad, inc = {}, [], []
    for w in ("sn", "sh"):
        f = emit / f"{w}.txt"
        if not f.is_file():
            inc.append(f"emit {w}")
            continue
        t = f.read_text()
        ex = [ln for ln in t.splitlines() if ln.startswith("export KRABBY_")]
        pay = [ln for ln in ex if ln.startswith("export KRABBY_PAYLOAD_")]
        out[f"emit_{w}"] = {"n_exports": len(ex), "payload_exports": pay}
        if not ex:
            bad.append(f"{w}_dagger.sh --dry-run composed no env (refused?): {t.strip()[-300:]}")
        if pay:
            bad.append(f"{w}: the DAgger env carries {pay}")
    for w, base, job, root in (("sn", "T4C6_SN_new", "C6_sn_new", C.OUT / "stand_network"),
                               ("sh", "T4C6_SH_new", "C6_sh_new", C.OUT / "stand_height")):
        nm = J.stand_name(base, job)
        if not (root / "runs" / nm / "smoke.json").is_file():
            inc.append(nm)
            continue
        s = smoke_ok(root, nm)
        out[w] = s
        if s["verdict"] != "PASS":
            bad.append(f"{nm} smoke verdict {s['verdict']}")
        if s["payload_keys"] or s["payload_term"]:
            bad.append(f"{nm}: payload in the DAgger trainer env {s['payload_keys']} term={s['payload_term']}")
    return verdict(out, bad, inc)


def chk_C6_det() -> dict:
    import t4_stepC_jobs as J  # noqa: PLC0415

    out, inc = {}, []
    for w, root, ref_log in (("sn", C.OUT / "stand_network", C.R / "parkour/outputs/omni_plan/stand_network/runs/"
                                                                   "SN_Amask_s17/train_log.jsonl"),
                             ("sh", C.OUT / "stand_height", C.R / "parkour/outputs/omni_plan/stand_height/runs/"
                                                                 "SH_T17/train_log.jsonl")):
        na = J.stand_name(f"T4C6_{w.upper()}_det_a", f"C6_det_{w}_a")
        nb = J.stand_name(f"T4C6_{w.upper()}_det_b", f"C6_det_{w}_b")
        sa, sb = smoke_ok(root, na), smoke_ok(root, nb)
        if not (sa["final"] and sb["final"] and Path(sa["final"]).is_file() and Path(sb["final"]).is_file()):
            inc.append(w)
            continue
        la, lb = Path(sa["run"]) / "train_log.jsonl", Path(sb["run"]) / "train_log.jsonl"
        ca, cb = C.RR().loss_curve(la), C.RR().loss_curve(lb)
        sha = C.sha256(sa["final"]) == C.sha256(sb["final"])
        rec = C.RR().loss_curve(ref_log) if ref_log.is_file() else []
        out[w] = {"sha_equal": sha, "loss_curve_equal": ca == cb, "bitwise": sha and ca == cb,
                  "n_rows": len(ca), "matches_recorded_prefix": (rec[:len(ca)] == ca) if rec else None}
    if inc:
        return {"verdict": INCOMPLETE, "pairs": out, "incomplete": inc}
    rule = "bitwise" if all(v["bitwise"] for v in out.values()) else "report_only"
    return {"verdict": PASS, "pairs": out, "x_rule": rule,
            "meaning": "bitwise: X decides on file sha + loss curves (replay_read.x_read); report_only: X is read "
                       "against the K lines, never feeds STOP-R, never holds F19"}


def chk_C7() -> dict:
    v = C.read_json(C.SC / "c7/dry_verdict.json", {}) or {}
    if v.get("verdict") not in ("PASS", "FAIL"):
        return {"verdict": INCOMPLETE, "why": "no dry verdict"}
    return {"verdict": PASS if v["verdict"] == "PASS" else FAIL, "dry": v}


# ================================================================================================ plumbing
def verdict(out: dict, bad: list, inc: list) -> dict:
    if bad:
        return {"verdict": FAIL, "problems": bad, "incomplete": inc, **out}
    if inc:
        return {"verdict": INCOMPLETE, "incomplete": inc, **out}
    return {"verdict": PASS, **out}


CHECKS = {n: globals()[f"chk_{n}"] for n in STOP_CHECKS + REPORT_CHECKS}


def run_check(name: str) -> dict:
    try:
        res = CHECKS[name]()
    except Incomplete as exc:
        res = {"verdict": INCOMPLETE, "why": str(exc)}
    except Exception as exc:  # noqa: BLE001  (a reader crash is INCOMPLETE with its traceback, never a PASS)
        res = {"verdict": INCOMPLETE, "why": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
    res["check"] = name
    res["class"] = "stop" if name in STOP_CHECKS else "report"
    C.write_json(CHECKS_DIR / f"{name}.json", res)
    return res


def report(js_out: Path, md_out: Path) -> int:
    res = {n: run_check(n) for n in CHECKS}
    stops = [n for n in STOP_CHECKS if res[n]["verdict"] == FAIL]
    pend = [n for n in STOP_CHECKS if res[n]["verdict"] == INCOMPLETE]
    c3 = res.get("C3_gate") or {}
    v1 = {"N_p": (res.get("C3_np") or {}).get("N_p"),
          "x_rule": (res.get("C6_det") or {}).get("x_rule"),
          "manifest_shape_split": ((res.get("C4_depth") or {}).get("manifest_shape_split") or {}).get("value"),
          "live": {"student_loss_rule": (res.get("C4_live") or {}).get("v1_student_loss_rule"),
                   "collapse_under_load": {k: (res.get("C3_pilot_live") or {}).get(k) for k in
                                           ("min_ep_len_over_kill_line", "max_crab_failure_roll")}},
          "p3a_rates": c3.get("v1_rates"),
          "reanchor_families": (res.get("C2") or {}).get("reanchor"),
          "sweep_anchors": ((res.get("C5") or {}).get("v1_threshold_inputs") or {}).get("sweep_anchors"),
          "costs": {"student_h_per_1k": (res.get("C4_timing_check") or {}).get("h_per_1k"),
                    "payload_distill_s_per_it": ((res.get("C4_distill_payload_check") or {}).get("timing") or {})
                    .get("s_per_it")}}
    split = v1["manifest_shape_split"]
    patch = {"pipeline.yaml": {"N_p": v1["N_p"], "x_rule": v1["x_rule"],
                               "stages[sel].completion_margin.manifest_shape_split": split,
                               "live.student.loss_rule (TP change, STEPC_RULES V.3)": v1["live"]["student_loss_rule"]},
             "threshold_inputs v1 (TP change, STEPC_RULES V.2)": {"p3a_pilot": c3.get("ckpt"),
                                                                  "sweep_anchors": v1["sweep_anchors"],
                                                                  "reanchor": v1["reanchor_families"]}}
    doc = {"stop_c": "YES" if stops else ("PENDING" if pend else "no"), "stop_c_checks": stops, "pending": pend,
           "checks": {n: {"verdict": r["verdict"], "problems": r.get("problems"), "why": r.get("why"),
                          "incomplete": r.get("incomplete")} for n, r in res.items()},
           "v1": v1, "patch": patch, "detail_dir": str(CHECKS_DIR)}
    C.write_json(js_out, doc)
    L = ["# Step 8 (Phase C) read", "", f"**STOP-C: {doc['stop_c']}**" + (f" -- {stops}" if stops else "")
         + (f"; pending: {pend}" if pend else ""), "", "| check | class | verdict | first problem |", "|---|---|---|---|"]
    for n, r in res.items():
        first = (r.get("problems") or [r.get("why") or ""])[0] if (r.get("problems") or r.get("why")) else ""
        L.append(f"| {n} | {r['class']} | {r['verdict']} | {str(first)[:110]} |")
    L += ["", "## Values for thresholds v1 / pipeline.yaml", "", "```json", json.dumps(v1, indent=1, default=str)[:4000],
          "```", "", f"Details per check: `{CHECKS_DIR}`."]
    Path(md_out).write_text("\n".join(L) + "\n")
    print("\n".join(L[:len(res) + 6]))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", default=None, choices=sorted(CHECKS))
    ap.add_argument("--json", default=str(C.PL / "preflight/stepC_read.json"))
    ap.add_argument("--md", default=str(C.PL / "preflight/stepC_read.md"))
    a = ap.parse_args(argv)
    if a.check:
        r = run_check(a.check)
        print(json.dumps({k: r.get(k) for k in ("check", "verdict", "problems", "why", "incomplete")}, default=str))
        return RC[r["verdict"]]
    return report(Path(a.json), Path(a.md))


if __name__ == "__main__":
    sys.exit(main())
