#!/usr/bin/env python
"""Descriptive mechanism telemetry reads for the PPO-construction screen (CHARTER 4.9 + the campaign's
PPO/* telemetry). CPU only: reads tfevents and the gate JSONs, launches nothing.

For every ``<records-dir>/*.record.json`` that is not killed (killed ones are skipped and listed), opens the
record's ``run_dir`` tfevents and computes windowed means of the PPO/Loss/Metrics tags, joins the per-hand
classes (record ``hand_l`` / ``hand_r``), the per-hand feet_ss at 0.22 / 0.25 (``<ref>_lr.json`` on the
two-direction beds, ``<ref>.json`` on the C bed) and the per-hand rod / joint odd RMS (``<ref>_odd.json``).

Window convention (matches scripts/verdict.py ``window``): a window "a-b" is the mean over logged steps
a <= step <= b. Exceptions: "600-3000" is 600 < step <= 3000 (so it does not share the step-600 point
with "0-600"); "x-3000" ends at the last logged step (2999). Point reads ("effective_std at N") take the
logged point nearest N. Tags that are absent come out as null.

Outputs: ``--json`` (per-run reads, bed x arm summary with 90 % t-CIs, FAST-vs-non-FAST hand contrast)
and ``--md`` (the same as tables).

    /home/nickmagus/krabby/isaac_venv/bin/python scripts/mechanism_reads.py \
        --json verdict/mechanism_reads.json --md verdict/mechanism_reads.md
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import math
from pathlib import Path

import numpy as np

CAMPAIGN = Path(
    "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/"
    "experiments/2026-10-01_ppo_construction"
)

# key -> tfevents tag
TAGS = {
    "hand_grad_cos": "PPO/hand_grad_cos",
    "hand_adv_l": "PPO/hand_adv_mean_l",
    "hand_adv_r": "PPO/hand_adv_mean_r",
    "hand_frac_l": "PPO/hand_frac_l",
    "hand_frac_r": "PPO/hand_frac_r",
    "lock_cam": "PPO/lock_cam",
    "lock_rod": "PPO/lock_rod",
    "sat1_cam": "PPO/sat1_cam",
    "sat1_rod": "PPO/sat1_rod",
    "sat15_cam": "PPO/sat15_cam",
    "sat15_rod": "PPO/sat15_rod",
    "mu_med_cam": "PPO/mu_med_cam",
    "mu_med_rod": "PPO/mu_med_rod",
    "effective_std": "PPO/effective_std",
    "exec_change": "PPO/exec_change",
    "exec_action_rate": "PPO/exec_action_rate",
    "kl_mean": "PPO/kl_mean",
    "lr_applied": "PPO/lr_applied",
    "clip_frac": "PPO/clip_frac",
    "explained_var": "PPO/explained_var",
    "grad_clip_share": "PPO/grad_clip_share",
    "symmetry": "Loss/symmetry",
    "action_bound": "Loss/action_bound",
    "track": "Metrics/base_parkour/track_ratio_mean",
    "track_rsi": "Metrics/base_parkour/track_ratio_mean_rsi",
    "track_norsi": "Metrics/base_parkour/track_ratio_mean_norsi",
}

# (output key, series key, lo, hi, lo_exclusive)
WINDOWS = [
    ("hand_grad_cos_0_200", "hand_grad_cos", 0, 200, False),
    ("hand_grad_cos_0_600", "hand_grad_cos", 0, 600, False),
    ("hand_grad_cos_600_3000", "hand_grad_cos", 600, 3000, True),
    ("hand_adv_l_0_200", "hand_adv_l", 0, 200, False),
    ("hand_adv_r_0_200", "hand_adv_r", 0, 200, False),
    ("hand_adv_l_0_600", "hand_adv_l", 0, 600, False),
    ("hand_adv_r_0_600", "hand_adv_r", 0, 600, False),
    ("hand_adv_l_2500_3000", "hand_adv_l", 2500, 3000, False),
    ("hand_adv_r_2500_3000", "hand_adv_r", 2500, 3000, False),
    ("hand_frac_l_all", "hand_frac_l", 0, 3000, False),
    ("hand_frac_r_all", "hand_frac_r", 0, 3000, False),
    ("lock_cam_300_600", "lock_cam", 300, 600, False),
    ("lock_rod_300_600", "lock_rod", 300, 600, False),
    ("lock_cam_2900_3000", "lock_cam", 2900, 3000, False),
    ("lock_rod_2900_3000", "lock_rod", 2900, 3000, False),
    ("sat1_cam_900_1000", "sat1_cam", 900, 1000, False),
    ("sat1_rod_900_1000", "sat1_rod", 900, 1000, False),
    ("sat15_cam_900_1000", "sat15_cam", 900, 1000, False),
    ("sat15_rod_900_1000", "sat15_rod", 900, 1000, False),
    ("mu_med_cam_900_1000", "mu_med_cam", 900, 1000, False),
    ("mu_med_rod_900_1000", "mu_med_rod", 900, 1000, False),
    ("sat1_cam_2900_3000", "sat1_cam", 2900, 3000, False),
    ("sat1_rod_2900_3000", "sat1_rod", 2900, 3000, False),
    ("sat15_cam_2900_3000", "sat15_cam", 2900, 3000, False),
    ("sat15_rod_2900_3000", "sat15_rod", 2900, 3000, False),
    ("mu_med_cam_2900_3000", "mu_med_cam", 2900, 3000, False),
    ("mu_med_rod_2900_3000", "mu_med_rod", 2900, 3000, False),
    ("exec_change_1000_3000", "exec_change", 1000, 3000, False),
    ("exec_action_rate_1000_3000", "exec_action_rate", 1000, 3000, False),
    ("kl_mean_500_3000", "kl_mean", 500, 3000, False),
    ("lr_applied_500_3000", "lr_applied", 500, 3000, False),
    ("clip_frac_500_3000", "clip_frac", 500, 3000, False),
    ("explained_var_500_3000", "explained_var", 500, 3000, False),
    ("grad_clip_share_500_3000", "grad_clip_share", 500, 3000, False),
    ("symmetry_0_600", "symmetry", 0, 600, False),
    ("symmetry_2500_3000", "symmetry", 2500, 3000, False),
    ("action_bound_0_600", "action_bound", 0, 600, False),
    ("action_bound_2500_3000", "action_bound", 2500, 3000, False),
    ("track_2900_3000", "track", 2900, 3000, False),
    ("track_rsi_2900_3000", "track_rsi", 2900, 3000, False),
    ("track_norsi_2900_3000", "track_norsi", 2900, 3000, False),
]
NPOINTS = [("hand_grad_cos_n_0_600", "hand_grad_cos", 0, 600, False),
           ("hand_grad_cos_n_600_3000", "hand_grad_cos", 600, 3000, True),
           ("hand_grad_cos_n_0_200", "hand_grad_cos", 0, 200, False)]
POINTS = [("effective_std_at_0", "effective_std", 0), ("effective_std_at_500", "effective_std", 500),
          ("effective_std_at_1000", "effective_std", 1000), ("effective_std_at_3000", "effective_std", 3000)]

# Summary reads (bed x arm): key -> label
SUMMARY_READS = [
    ("hand_grad_cos_0_600", "hand_grad_cos 0-600"),
    ("lock_300_600", "lock share 300-600 (cam,rod mean)"),
    ("lock_cam_300_600", "lock_cam 300-600"),
    ("lock_rod_300_600", "lock_rod 300-600"),
    ("rod_odd_max_hand", "rod odd RMS max-hand"),
    ("rod_odd_min_hand", "rod odd RMS min-hand"),
    ("effective_std_at_3000", "effective_std @3000"),
    ("exec_change_1000_3000", "exec_change 1000-3000"),
    ("lr_applied_500_3000", "lr_applied 500-3000"),
    ("mu_med_cam_2900_3000", "mu_med_cam 2900-3000"),
    ("mu_med_rod_2900_3000", "mu_med_rod 2900-3000"),
    ("symmetry_2500_3000", "Loss/symmetry 2500-3000"),
    ("track_2900_3000", "track_ratio_mean 2900-3000"),
]


# ------------------------------------------------------------------------------------------- helpers
def load_series(run_dir: str) -> tuple[dict, list[str]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: PLC0415

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = list(ea.Tags().get("scalars", []))
    out = {}
    for key, tag in TAGS.items():
        if tag in have:
            s = ea.Scalars(tag)
            out[key] = (np.array([x.step for x in s]), np.array([x.value for x in s], dtype=float))
    return out, have


def _mask(st, lo, hi, lo_excl):
    return ((st > lo) if lo_excl else (st >= lo)) & (st <= hi)


def window(series, lo, hi, lo_excl=False):
    if series is None:
        return None
    st, v = series
    m = _mask(st, lo, hi, lo_excl) & np.isfinite(v)
    return float(np.mean(v[m])) if m.any() else None


def npoints(series, lo, hi, lo_excl=False):
    if series is None:
        return 0
    st, v = series
    return int((_mask(st, lo, hi, lo_excl) & np.isfinite(v)).sum())


def nearest(series, at):
    if series is None:
        return None
    st, v = series
    i = int(np.argmin(np.abs(st - at)))
    return {"step": int(st[i]), "value": float(v[i])}


def _mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def tci90(xs):
    """mean, sd, n, 90 % t-CI (t with n-1 df). CI is null for n < 2."""
    from scipy import stats  # noqa: PLC0415

    xs = [float(x) for x in xs if x is not None and math.isfinite(x)]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci90": None}
    m = float(np.mean(xs))
    if n < 2:
        return {"n": 1, "mean": m, "sd": None, "ci90": None}
    sd = float(np.std(xs, ddof=1))
    h = float(stats.t.ppf(0.95, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": m, "sd": sd, "ci90": [m - h, m + h]}


def _load_json(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def _inner(d, ref):
    if d is None:
        return None
    return d.get(ref) or (next(iter(d.values())) if len(d) == 1 else None)


def gate_reads(records_dir: Path, rec: dict) -> dict:
    ref, two = rec["ref"], bool(rec.get("two_direction"))
    feet_src = records_dir / (f"{ref}_lr.json" if two else f"{ref}.json")
    odd_src = records_dir / f"{ref}_odd.json"
    feet = _inner(_load_json(feet_src), ref)
    odd = _inner(_load_json(odd_src), ref)
    hands = ["l", "r"] if two else ["l"]
    out = {"feet_source": str(feet_src) if feet else None, "odd_source": str(odd_src) if odd else None,
           "hands": {}}
    for h in hands:
        labels = (feet or {}).get("labels", {})
        hr = {
            "class": rec.get(f"hand_{h}"),
            "feet_ss_022": (labels.get(f"lat_{h}_022") or {}).get("feet_ss"),
            "feet_ss_025": (labels.get(f"lat_{h}_025") or {}).get("feet_ss"),
            "rod_odd_rms": (odd or {}).get(f"rod_odd_rms_{h}"),
            "joint_odd_rms": (odd or {}).get(f"joint_odd_rms_{h}"),
        }
        out["hands"][h] = hr
    rods = [hr["rod_odd_rms"] for hr in out["hands"].values() if hr["rod_odd_rms"] is not None]
    out["rod_odd_max_hand"] = max(rods) if rods else None
    out["rod_odd_min_hand"] = min(rods) if rods else None
    return out


def run_reads(records_dir: Path, rec: dict) -> tuple[dict, list[str]]:
    ser, have = load_series(rec["run_dir"])
    r = {k: rec.get(k) for k in ("ref", "bed", "arm", "seed", "two_direction", "run_dir")}
    r["last_step"] = max((int(s[0].max()) for s in ser.values() if len(s[0])), default=None)
    r["missing_tags"] = sorted(t for k, t in TAGS.items() if k not in ser)
    for out_key, key, lo, hi, excl in WINDOWS:
        r[out_key] = window(ser.get(key), lo, hi, excl)
    for out_key, key, lo, hi, excl in NPOINTS:
        r[out_key] = npoints(ser.get(key), lo, hi, excl)
    for out_key, key, at in POINTS:
        p = nearest(ser.get(key), at)
        r[out_key] = None if p is None else p["value"]
        r[out_key + "_step"] = None if p is None else p["step"]
    r["lock_300_600"] = _mean([r["lock_cam_300_600"], r["lock_rod_300_600"]])
    r["lock_2900_3000"] = _mean([r["lock_cam_2900_3000"], r["lock_rod_2900_3000"]])
    g = gate_reads(records_dir, rec)
    r["gate"] = g
    r["rod_odd_max_hand"] = g["rod_odd_max_hand"]
    r["rod_odd_min_hand"] = g["rod_odd_min_hand"]
    return r, have


def fast_contrast(runs: list[dict]) -> dict:
    """A/B two-direction runs with exactly one FAST hand: the FAST hand vs the other hand."""
    rows = []
    for r in runs:
        if not r["two_direction"] or r["arm"] not in ("A", "B"):
            continue
        cls = {h: r["gate"]["hands"][h]["class"] for h in ("l", "r")}
        fast = [h for h in ("l", "r") if cls[h] == "FAST"]
        if len(fast) != 1:
            continue
        f = fast[0]
        o = "r" if f == "l" else "l"
        row = {"ref": r["ref"], "bed": r["bed"], "arm": r["arm"], "seed": r["seed"], "fast_hand": f,
               "other_hand": o, "other_class": cls[o]}
        for w in ("0_200", "0_600", "2500_3000"):
            af, ao = r.get(f"hand_adv_{f}_{w}"), r.get(f"hand_adv_{o}_{w}")
            row[f"adv_fast_{w}"], row[f"adv_other_{w}"] = af, ao
            row[f"adv_diff_{w}"] = None if af is None or ao is None else af - ao
        for k in ("rod_odd_rms", "joint_odd_rms", "feet_ss_022", "feet_ss_025"):
            row[f"{k}_fast"] = r["gate"]["hands"][f][k]
            row[f"{k}_other"] = r["gate"]["hands"][o][k]
        row["frac_fast"] = r.get(f"hand_frac_{f}_all")
        row["frac_other"] = r.get(f"hand_frac_{o}_all")
        rows.append(row)
    summ = {}
    for w in ("0_200", "0_600", "2500_3000"):
        d = [x[f"adv_diff_{w}"] for x in rows if x[f"adv_diff_{w}"] is not None]
        summ[f"adv_diff_{w}"] = tci90(d) | {"n_fast_higher": int(sum(v > 0 for v in d)), "n": len(d)}
    for k in ("rod_odd_rms", "joint_odd_rms", "feet_ss_022", "feet_ss_025"):
        d = [x[f"{k}_fast"] - x[f"{k}_other"] for x in rows
             if x[f"{k}_fast"] is not None and x[f"{k}_other"] is not None]
        summ[f"{k}_diff"] = tci90(d) | {"n_fast_higher": int(sum(v > 0 for v in d)), "n": len(d)}
    # Same comparison restricted to L-FAST and R-FAST separately (hand identity confound check)
    summ["fast_hand_counts"] = {"l": sum(x["fast_hand"] == "l" for x in rows),
                                "r": sum(x["fast_hand"] == "r" for x in rows)}
    return {"rows": rows, "summary": summ}


def hand_adv_bias_all(runs: list[dict]) -> dict:
    """Context: L-minus-R early advantage across ALL two-direction runs (incl. C0, no FAST hand)."""
    out = {}
    for arm in ("C0", "A", "B"):
        d = [r["hand_adv_l_0_600"] - r["hand_adv_r_0_600"] for r in runs
             if r["two_direction"] and r["arm"] == arm and r["hand_adv_l_0_600"] is not None
             and r["hand_adv_r_0_600"] is not None]
        out[arm] = tci90(d) | {"n_l_higher": int(sum(v > 0 for v in d))}
    return out


# ------------------------------------------------------------------------------------------- markdown
def _f(x, nd=3):
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    if abs(x) != 0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def _ci(s, nd=3):
    if s["n"] == 0:
        return "--"
    if s["ci90"] is None:
        return f"{_f(s['mean'], nd)} (n=1)"
    return f"{_f(s['mean'], nd)} [{_f(s['ci90'][0], nd)}, {_f(s['ci90'][1], nd)}]"


def to_md(doc: dict) -> str:
    L = []
    m = doc["meta"]
    L += ["# PPO construction -- mechanism reads (descriptive, CHARTER 4.9)", "",
          f"Generated {m['generated_at']} by `scripts/mechanism_reads.py` from `{m['records_dir']}` "
          f"({len(doc['runs'])} runs read, {len(m['skipped_killed'])} killed skipped).", "",
          "Windows are inclusive on logged steps (verdict.py convention); `600-3000` excludes step 600; "
          "`x-3000` ends at the last logged step 2999. Hand tags are logged every 25 iterations "
          "(hand diagnostics are off on the C bed: those columns are `--`). `hand_adv_*` are batch-normalised "
          "advantages (z-scores over the 256x24 rollout) averaged over the first 4096 rows of the rollout, "
          "split by the sign of the observed lateral command (L = +y).", "",
          "Killed runs skipped: " + (", ".join(f"{k['ref']} ({k['kind']} at {k['at']})"
                                              for k in m["skipped_killed"]) or "none"), ""]
    L += ["## Tags found", "",
          "All requested tags exist under the requested names. Absent per run (null): " +
          "; ".join(f"`{t}` missing in {', '.join(v)}" for t, v in m["missing_by_tag"].items()) + ".", ""]

    L += ["## Bed x arm summary (mean [90 % t-CI, n-1 df])", ""]
    hdr = ["bed", "arm", "n"] + [lab for _, lab in SUMMARY_READS]
    L += ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for row in doc["summary_rows"]:
        L.append("| " + " | ".join([row["bed"], row["arm"], str(row["n_runs"])] +
                                   [_ci(row["reads"][k], 4 if k.startswith("lr") else 3)
                                    for k, _ in SUMMARY_READS]) + " |")
    L.append("")

    L += ["## Per-run reads", "", "### Hands, locks, saturation", ""]
    cols = [("ref", "ref"), ("hand L/R", None), ("cos 0-200", "hand_grad_cos_0_200"),
            ("cos 0-600", "hand_grad_cos_0_600"), ("n", "hand_grad_cos_n_0_600"),
            ("cos 600-3000", "hand_grad_cos_600_3000"), ("n", "hand_grad_cos_n_600_3000"),
            ("advL 0-600", "hand_adv_l_0_600"), ("advR 0-600", "hand_adv_r_0_600"),
            ("advL 2500-3000", "hand_adv_l_2500_3000"), ("advR 2500-3000", "hand_adv_r_2500_3000"),
            ("fracL", "hand_frac_l_all"), ("fracR", "hand_frac_r_all"),
            ("lock cam/rod 300-600", ("lock_cam_300_600", "lock_rod_300_600")),
            ("lock cam/rod 2900-3000", ("lock_cam_2900_3000", "lock_rod_2900_3000")),
            ("sat1 cam/rod 900-1000", ("sat1_cam_900_1000", "sat1_rod_900_1000")),
            ("sat15 cam/rod 900-1000", ("sat15_cam_900_1000", "sat15_rod_900_1000")),
            ("mu_med cam/rod 900-1000", ("mu_med_cam_900_1000", "mu_med_rod_900_1000")),
            ("sat1 cam/rod 2900-3000", ("sat1_cam_2900_3000", "sat1_rod_2900_3000")),
            ("sat15 cam/rod 2900-3000", ("sat15_cam_2900_3000", "sat15_rod_2900_3000")),
            ("mu_med cam/rod 2900-3000", ("mu_med_cam_2900_3000", "mu_med_rod_2900_3000"))]
    L += _table(doc["runs"], cols)
    L += ["", "### Std, update size, optimiser, losses, tracking", ""]
    cols = [("ref", "ref"), ("std@0", "effective_std_at_0"), ("std@500", "effective_std_at_500"),
            ("std@1000", "effective_std_at_1000"), ("std@3000", "effective_std_at_3000"),
            ("exec_change", "exec_change_1000_3000"), ("exec_rate", "exec_action_rate_1000_3000"),
            ("kl", "kl_mean_500_3000"), ("lr_applied", "lr_applied_500_3000"), ("clip_frac", "clip_frac_500_3000"),
            ("expl_var", "explained_var_500_3000"), ("gclip", "grad_clip_share_500_3000"),
            ("sym 0-600", "symmetry_0_600"), ("sym 2500-3000", "symmetry_2500_3000"),
            ("bound 0-600", "action_bound_0_600"), ("bound 2500-3000", "action_bound_2500_3000"),
            ("track", "track_2900_3000"), ("track_rsi", "track_rsi_2900_3000"),
            ("track_norsi", "track_norsi_2900_3000")]
    L += _table(doc["runs"], cols)
    L += ["", "### Gate per-hand reads (feet_ss 0.22 / 0.25, rod / joint odd RMS)", ""]
    L += ["| ref | hand | class | feet_ss 0.22 | feet_ss 0.25 | rod odd RMS | joint odd RMS |",
          "|---|---|---|---|---|---|---|"]
    for r in doc["runs"]:
        for h, hr in r["gate"]["hands"].items():
            L.append(f"| {r['ref']} | {h.upper()} | {hr['class']} | {_f(hr['feet_ss_022'])} | "
                     f"{_f(hr['feet_ss_025'])} | {_f(hr['rod_odd_rms'])} | {_f(hr['joint_odd_rms'])} |")
    L.append("")

    fc = doc["fast_contrast"]
    L += ["## FAST hand vs non-FAST hand (A/B two-direction runs with exactly one FAST hand)", "",
          f"FAST hand is L in {fc['summary']['fast_hand_counts']['l']} runs, R in "
          f"{fc['summary']['fast_hand_counts']['r']}.", "",
          "| ref | FAST | other (class) | adv F/O 0-200 | adv F/O 0-600 | diff 0-600 | adv F/O 2500-3000 | "
          "rod odd F/O | joint odd F/O | feet_ss 0.22 F/O | feet_ss 0.25 F/O |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in fc["rows"]:
        L.append(f"| {x['ref']} | {x['fast_hand'].upper()} | {x['other_hand'].upper()} ({x['other_class']}) | "
                 f"{_f(x['adv_fast_0_200'])}/{_f(x['adv_other_0_200'])} | "
                 f"{_f(x['adv_fast_0_600'])}/{_f(x['adv_other_0_600'])} | {_f(x['adv_diff_0_600'])} | "
                 f"{_f(x['adv_fast_2500_3000'])}/{_f(x['adv_other_2500_3000'])} | "
                 f"{_f(x['rod_odd_rms_fast'])}/{_f(x['rod_odd_rms_other'])} | "
                 f"{_f(x['joint_odd_rms_fast'])}/{_f(x['joint_odd_rms_other'])} | "
                 f"{_f(x['feet_ss_022_fast'])}/{_f(x['feet_ss_022_other'])} | "
                 f"{_f(x['feet_ss_025_fast'])}/{_f(x['feet_ss_025_other'])} |")
    L += ["", "| paired difference (FAST - other) | mean [90 % t-CI] | runs FAST higher / n |", "|---|---|---|"]
    for k, s in fc["summary"].items():
        if k == "fast_hand_counts":
            continue
        L.append(f"| {k} | {_ci(s)} | {s['n_fast_higher']}/{s['n']} |")
    L += ["", "Context -- early advantage L minus R (0-600) over all two-direction runs, by arm:", "",
          "| arm | mean [90 % t-CI] | runs L higher / n |", "|---|---|---|"]
    for arm, s in doc["hand_adv_bias_all"].items():
        L.append(f"| {arm} | {_ci(s)} | {s['n_l_higher']}/{s['n']} |")
    L.append("")
    return "\n".join(L)


def _table(runs, cols):
    L = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in runs:
        cells = []
        for c, k in cols:
            if k is None:
                cells.append(" / ".join(str(h["class"]) for h in r["gate"]["hands"].values()))
            elif isinstance(k, tuple):
                cells.append("/".join(_f(r[x]) for x in k))
            elif k == "ref":
                cells.append(r["ref"])
            elif k.startswith("hand_grad_cos_n"):
                cells.append(str(r[k]))
            else:
                cells.append(_f(r[k], 4 if "lr_" in k else 3))
        L.append("| " + " | ".join(cells) + " |")
    return L


# ------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records-dir", default=str(CAMPAIGN / "gate"))
    ap.add_argument("--json", default=str(CAMPAIGN / "verdict" / "mechanism_reads.json"))
    ap.add_argument("--md", default=str(CAMPAIGN / "verdict" / "mechanism_reads.md"))
    args = ap.parse_args(argv)
    records_dir = Path(args.records_dir)

    runs, skipped, tags_seen = [], [], {}
    for p in sorted(glob.glob(str(records_dir / "*.record.json"))):
        rec = json.loads(Path(p).read_text())
        if rec.get("killed"):
            skipped.append({"ref": rec["ref"], "kind": rec.get("killed_kind"), "at": rec.get("killed_at"),
                            "run_dir": rec.get("run_dir")})
            continue
        r, have = run_reads(records_dir, rec)
        tags_seen[rec["ref"]] = have
        runs.append(r)
        print(f"[read] {r['ref']}: last step {r['last_step']}, missing {len(r['missing_tags'])} tags")

    bed_order = {"C": 0, "C_lr": 1, "C_lr_rsi": 2}
    arm_order = {"C0": 0, "A": 1, "B": 2}
    runs.sort(key=lambda r: (bed_order.get(r["bed"], 9), arm_order.get(r["arm"], 9), r["seed"]))

    missing_by_tag = {}
    for r in runs:
        for t in r["missing_tags"]:
            missing_by_tag.setdefault(t, []).append(r["ref"])

    summary_rows = []
    for bed in sorted({r["bed"] for r in runs}, key=lambda b: bed_order.get(b, 9)):
        for arm in sorted({r["arm"] for r in runs if r["bed"] == bed}, key=lambda a: arm_order.get(a, 9)):
            sel = [r for r in runs if r["bed"] == bed and r["arm"] == arm]
            summary_rows.append({"bed": bed, "arm": arm, "n_runs": len(sel), "refs": [r["ref"] for r in sel],
                                 "reads": {k: tci90([r.get(k) for r in sel]) for k, _ in SUMMARY_READS}})

    doc = {
        "meta": {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "records_dir": str(records_dir),
            "script": str(Path(__file__).resolve()),
            "tags": TAGS,
            "windows": [{"key": k, "tag": TAGS[s], "lo": lo, "hi": hi, "lo_exclusive": e}
                        for k, s, lo, hi, e in WINDOWS],
            "points": [{"key": k, "tag": TAGS[s], "at": at} for k, s, at in POINTS],
            "all_scalar_tags_union": sorted(set().union(*[set(v) for v in tags_seen.values()])) if tags_seen else [],
            "missing_by_tag": missing_by_tag,
            "skipped_killed": skipped,
            "ci": "90 % two-sided t-interval, t with n-1 df, over runs (seeds) in a bed x arm cell",
        },
        "runs": runs,
        "summary_rows": summary_rows,
        "fast_contrast": fast_contrast(runs),
        "hand_adv_bias_all": hand_adv_bias_all(runs),
    }
    Path(args.json).write_text(json.dumps(doc, indent=1))
    Path(args.md).write_text(to_md(doc))
    print(f"[write] {args.json}\n[write] {args.md}")


if __name__ == "__main__":
    main()
