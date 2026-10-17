#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Task 4 payload reader (CPU, pure numpy on the raw npz): level stability, max_stable, the limiting factor, and the
REPORT-ONLY actuator load. Every payload result is labelled "sim plant, force side not enforced" (PLAN Payload).

STABLE AT L (PLAN Payload; DESIGN 3.3 minus the lines PLAN retired: support_feet / swing_apex) -- the a-priori lines
are READ FROM thresholds.json (``p3a.stable90.*``: survival per trained translate category >= 0.90, translate
|ratio - 1| <= 0.25, turn and move_turn survival >= 0.885, turn facing_closed >= 0.80, median h_meas >= 0.85 m on
suite and field_light, roll / pitch RMS <= 0.10 rad on suite and field_light, field_light completion >= 0.85), each
on suite + field_light at es1-2 of the level rows (``KG=L:L, ZERO_FRAC=0, COM_XY=0, COM_Z=0.309``). The same lines
apply at every level L (S1 at 0 kg passes all of them -- the threshold self-check).

    max_stable(curve) = the largest L such that every level <= L is stable (the curve runs 0..270 in 45-kg steps and
                        stops at the first unstable level).

LIMITING FACTOR at the first unstable level (DESIGN 3.4 + PLAN "Budget-bound"; first match wins):
  0 budget-bound     the stage's TRAINING tail: survival (1 - Episode_Termination/crab_failure) AND tracking
                     (Metrics/base_parkour/track_ratio_mean) both still rising at N_p. Defined on the POOLED tail
                     (PLAN left-for-implementation item 11): the last 20 % of the stage's logged iterations split in
                     two halves; rising iff mean(second) - mean(first) > 2 x the pooled within-half SE, for both;
  1 training instability  a live kill / NaN on the stage;
  2 actuator         effort-cap share at L >= 2 x its 0-kg share AND median h_meas sag (0 kg - L) >= 0.05 m;
  3 stability        a survival / orientation / height line failed;
  4 tracking         only tracking / facing / completion lines failed.
ACTUATOR LOAD (report only; DESIGN 3.3): hip / knee effort-cap share (|applied_torque| >= 0.98 x the holding-level cap,
1000 / 400 N m), rod force p50 / p90 against rated (2000 / 500 N) and self-lock (2800 / 4500 N) via ``sh_kin.py``'s
coupled rod math (the spec's ``paths.sh_kin``, loaded by path, no bytecode), h_meas sag against 0 kg, sum of foot force against (230 + L) g
(WATCH on level rows, never on field_hard).

    payload_read.py --spec PL/pipeline.yaml level --thresholds T.json --stamp suite@1=DIR --stamp suite@2=DIR --stamp field_light@1=DIR ...
                          [--level 90] [--trained fwd,side,back,diag]
    payload_read.py --spec PL/pipeline.yaml actuator --stamp DIR [--level 90]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def _lib():
    import importlib.util  # noqa: PLC0415

    if "tp_lib" in sys.modules:
        return sys.modules["tp_lib"]
    spec = importlib.util.spec_from_file_location("tp_lib", str(HERE / "tp_lib.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tp_lib"] = mod
    spec.loader.exec_module(mod)
    return mod


L = _lib()
LABEL = "sim plant, force side not enforced"
ROBOT_KG = 230.06
G = 9.81
CAP_NM = {"hip": 1000.0, "knee": 400.0}
RATED_N = {"hip": 2000.0, "knee": 500.0}
SELF_LOCK_N = {"hip": 2800.0, "knee": 4500.0}
CAP_FRAC = 0.98
SAG_MIN_M = 0.05
TAIL_FRAC = 0.20
LIMITING = ("budget_bound", "training_instability", "actuator", "stability", "tracking")
_STABLE_RE = re.compile(r"^p3a\.stable90\.(?P<stat>[a-z_]+)\.(?P<scope>[a-z_]+)\.es(?P<es>\d+)$")


def sh_kin():
    return L.load_by_path(L.path("sh_kin"), "tp_sh_kin")


# ============================================================================== the a-priori lines
def stable_lines(th_doc: dict) -> list[dict]:
    """The ``p3a.stable90.*`` lines of a thresholds.json as ``{id, stat, scope, es, line, direction, class}``.

    ``stat``: survival | tracking | turn_fc | h_meas | roll_rms | pitch_rms | field_light_completion; ``scope``: a
    suite category (survival / tracking), ``suite`` / ``field_light`` (h_meas / orientation), or None. Refuses an id
    this reader does not know (a new line must be read here before it can gate)."""
    out = []
    for x in th_doc["lines"]:
        lid = x["id"]
        if not lid.startswith("p3a.stable90."):
            continue
        if lid == "p3a.stable90.field_light_completion":
            out.append({"id": lid, "stat": "field_light_completion", "scope": None, "es": None, "line": x["line"],
                        "direction": x["direction"]})
            continue
        m = _STABLE_RE.match(lid)
        if m is None:
            m2 = re.match(r"^p3a\.stable90\.turn_fc\.es(\d+)$", lid)
            if m2:
                out.append({"id": lid, "stat": "turn_fc", "scope": "turn", "es": int(m2.group(1)), "line": x["line"],
                            "direction": x["direction"]})
                continue
            raise ValueError(f"payload_read: unknown stable-at-L line id {lid!r}")
        out.append({"id": lid, "stat": m.group("stat"), "scope": m.group("scope"), "es": int(m.group("es")),
                    "line": x["line"], "direction": x["direction"]})
    if not out:
        raise ValueError("thresholds.json carries no p3a.stable90.* lines")
    return out


def _cmp(v, line, direction) -> bool:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return False
    return v >= line - 1e-12 if direction == ">=" else v <= line + 1e-12


def level_reads(stamps: dict, th_doc: dict, *, trained_translate=None) -> dict:
    """Stable-at-L on one level's rows ``{"suite@1": dir, "suite@2": dir, "field_light@1": dir, "field_light@2": dir}``.

    ``trained_translate``: the stage's trained translate categories (lines for other categories are skipped and
    listed). Returns ``{stable, lines: [...], failed: [...], skipped: [...]}``."""
    TH = L.thresholds_mod()
    cats, horient = {}, {}
    for es in (1, 2):
        st = stamps.get(f"suite@{es}")
        if st is not None:
            cats[es] = TH.suite_categories(Path(st))
            horient[("suite", es)] = TH.h_meas_and_orientation(Path(st))
        fl = stamps.get(f"field_light@{es}")
        if fl is not None:
            horient[("field_light", es)] = TH.h_meas_and_orientation(Path(fl))
    fls = [Path(stamps[f"field_light@{es}"]) for es in (1, 2) if f"field_light@{es}" in stamps]
    flc = TH.field_completion(fls) if fls else None
    lines, failed, skipped = [], [], []
    for ln in stable_lines(th_doc):
        stat, scope, es = ln["stat"], ln["scope"], ln["es"]
        if stat in ("survival", "tracking") and scope in ("fwd", "side", "back", "diag") and \
                trained_translate is not None and scope not in trained_translate:
            skipped.append(ln["id"])
            continue
        v = None
        if stat == "survival":
            v = ((cats.get(es) or {}).get(scope) or {}).get("survival")
        elif stat == "tracking":
            r = ((cats.get(es) or {}).get(scope) or {}).get("ratio")
            v = None if r is None else abs(r - 1.0)
        elif stat == "turn_fc":
            v = ((cats.get(es) or {}).get("turn") or {}).get("facing_closed")
        elif stat in ("h_meas",):
            v = (horient.get((scope, es)) or {}).get("h_meas_median")
        elif stat in ("roll_rms", "pitch_rms"):
            v = (horient.get((scope, es)) or {}).get(stat)
        elif stat == "field_light_completion":
            v = None if flc is None else flc["rate"]
        ok = _cmp(v, ln["line"], ln["direction"])
        rec = {**ln, "value": v, "pass": ok}
        lines.append(rec)
        if not ok:
            failed.append(ln["id"])
    return {"stable": not failed and bool(lines), "lines": lines, "failed": failed, "skipped": skipped,
            "label": LABEL}


def max_stable(curve: dict):
    """``curve`` {L: stable bool} -> the largest L with every level <= L stable (None when level 0 is unstable or
    absent). Levels are read ascending; a gap (an unread level below a stable one) stops the count."""
    best = None
    for lv in sorted(curve):
        if not curve[lv]:
            break
        best = lv
    return best


# ============================================================================== training tail (budget-bound)
def tail_rising(steps, values, frac: float = TAIL_FRAC) -> dict:
    """Pooled-tail rise test (PLAN item 11): the last ``frac`` of the logged iterations in two halves; rising iff
    mean(second) - mean(first) > 2 sqrt((var1 + var2) / n_half) (the pooled within-half SE of the difference)."""
    v = np.asarray(values, float)
    n = len(v)
    k = max(4, int(round(n * frac)))
    if n < 8:
        return {"rising": None, "why": f"only {n} logged points"}
    t = v[-k:]
    h = len(t) // 2
    a, b = t[:h], t[h:2 * h]
    se = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / max(h, 1)) if h > 1 else float("inf")
    d = float(b.mean() - a.mean())
    return {"rising": bool(d > 2.0 * se), "diff": d, "se": se, "n_half": h,
            "steps": [int(steps[-k]), int(steps[-1])] if len(steps) else None}


def budget_bound(series: dict) -> dict:
    """``series``: {tag: [(step, value)]} of the stage's training log (Episode_Termination/crab_failure,
    Metrics/base_parkour/track_ratio_mean)."""
    cf = series.get("Episode_Termination/crab_failure") or []
    tr = series.get("Metrics/base_parkour/track_ratio_mean") or []
    if not cf or not tr:
        return {"budget_bound": None, "why": "survival / tracking telemetry missing"}
    s = tail_rising([x for x, _ in cf], [1.0 - y for _, y in cf])
    t = tail_rising([x for x, _ in tr], [y for _, y in tr])
    bb = None if (s["rising"] is None or t["rising"] is None) else bool(s["rising"] and t["rising"])
    return {"budget_bound": bb, "survival": s, "tracking": t}


# ============================================================================== actuator load (report only)
def _joint_index(names, pat):
    return [i for i, n in enumerate(names) if pat in n]


def actuator_load(stamp: Path, level_kg: float = 0.0) -> dict:
    """Report-only actuator metrics of one stamp's raw npz (DESIGN 3.3). ``applied_torque`` is the implicit-actuator
    estimate (caveat in the output)."""
    stamp = Path(stamp)
    meta = json.loads((stamp / "run_meta.json").read_text())
    names = meta.get("joint_names") or []
    hip = _joint_index(names, "_Hip_Femur_")
    knee = _joint_index(names, "_Femur_Tibia_")
    files = sorted(glob.glob(str(stamp / "raw" / "episode_*.npz")))
    if not files or len(hip) != 6 or len(knee) != 6:
        return {"available": False, "why": f"{len(files)} raw episodes, {len(hip)} hip / {len(knee)} knee joints",
                "label": LABEL}
    kin = sh_kin()
    cap_steps = {"hip": 0, "knee": 0}
    steps = 0
    fh_all, fk_all, sumf = [], [], []
    for f in files:
        z = np.load(f)
        done = z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(done)
        tau = z["applied_torque"][:S].astype(float)
        q = z["joint_pos"][:S].astype(float)
        steps += S
        cap_steps["hip"] += int((np.abs(tau[:, hip]) >= CAP_FRAC * CAP_NM["hip"]).any(axis=1).sum())
        cap_steps["knee"] += int((np.abs(tau[:, knee]) >= CAP_FRAC * CAP_NM["knee"]).any(axis=1).sum())
        side = np.asarray(kin.SIDE)
        th, tk = q[:, hip], q[:, knee] * side[None, :]
        th_tau, tk_tau = tau[:, hip], tau[:, knee] * side[None, :]
        try:
            fh, fk = kin.rod_forces(th, tk, th_tau, tk_tau)
            fh_all.append(np.abs(np.asarray(fh)).ravel())
            fk_all.append(np.abs(np.asarray(fk)).ravel())
        except Exception:  # noqa: BLE001  (report-only; an unreadable pose range is recorded, never decides)
            pass
        if "foot_force_norm" in z.files:
            sumf.append(float(np.median(z["foot_force_norm"][:S].astype(float).sum(axis=1))))
    fh = np.concatenate(fh_all) if fh_all else np.array([])
    fk = np.concatenate(fk_all) if fk_all else np.array([])
    weight = (ROBOT_KG + float(level_kg)) * G

    def pct(x, p):
        return float(np.percentile(x, p)) if x.size else None

    return {"available": True, "label": LABEL, "steps": steps,
            "torque_note": "applied_torque is the implicit-actuator ESTIMATE (not a measured force)",
            "cap_share": {k: (cap_steps[k] / steps if steps else None) for k in cap_steps},
            "rod_force_n": {"hip": {"p50": pct(fh, 50), "p90": pct(fh, 90), "rated": RATED_N["hip"],
                                    "self_lock": SELF_LOCK_N["hip"]},
                            "knee": {"p50": pct(fk, 50), "p90": pct(fk, 90), "rated": RATED_N["knee"],
                                     "self_lock": SELF_LOCK_N["knee"]}},
            "sum_foot_force_ratio": (float(np.median(sumf)) / weight) if sumf else None,
            "sum_foot_force_note": "median over episodes of the per-episode median sum |F_foot| / ((230 + L) g); "
                                   "WATCH on level rows (hard only at the step-8 zero-action settle)"}


def limiting_factor(level_read: dict, *, live_killed: bool = False, nan: bool = False, budget: dict | None = None,
                    actuator_L: dict | None = None, actuator_0: dict | None = None, h_sag: float | None = None) -> dict:
    """The first matching class at the first unstable level (module docstring)."""
    if budget and budget.get("budget_bound"):
        return {"class": 0, "name": LIMITING[0], "why": "training tail still rising at N_p"}
    if live_killed or nan:
        return {"class": 1, "name": LIMITING[1], "why": "live kill" if live_killed else "NaN"}
    if actuator_L and actuator_0 and actuator_L.get("available") and actuator_0.get("available"):
        for j in ("hip", "knee"):
            a, b = (actuator_L["cap_share"] or {}).get(j), (actuator_0["cap_share"] or {}).get(j)
            if a is not None and b is not None and a >= 2.0 * max(b, 1e-9) and (h_sag or 0.0) >= SAG_MIN_M:
                return {"class": 2, "name": LIMITING[2], "why": f"{j} cap share {a:.3f} >= 2 x {b:.3f} and sag "
                                                                 f"{h_sag:.3f} m"}
    failed = set(level_read.get("failed") or [])
    stab = [f for f in failed if any(s in f for s in (".survival.", ".h_meas.", ".roll_rms.", ".pitch_rms."))]
    if stab:
        return {"class": 3, "name": LIMITING[3], "why": sorted(stab)}
    return {"class": 4, "name": LIMITING[4], "why": sorted(failed)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ap.add_argument("--spec", type=Path, required=True, help="the pipeline spec (its paths block)")
    a1 = sub.add_parser("level")
    a1.add_argument("--thresholds", required=True, type=Path)
    a1.add_argument("--stamp", action="append", default=[], metavar="ROW@ES=DIR")
    a1.add_argument("--trained", default=None)
    a2 = sub.add_parser("actuator")
    a2.add_argument("--stamp", required=True, type=Path)
    a2.add_argument("--level", type=float, default=0.0)
    a = ap.parse_args(argv)
    L.configure_from(a.spec)
    if a.cmd == "level":
        th = json.loads(a.thresholds.read_text())
        res = level_reads(dict(s.split("=", 1) for s in a.stamp), th,
                          trained_translate=a.trained.split(",") if a.trained else None)
    else:
        res = actuator_load(a.stamp, a.level)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
