"""Evidence table for the side bisect. hold_stats/why_excluded/yaw imported verbatim from the classifier;
cadence uses the same window [a4, hi) and the same F>50 N mask as hold_stats' sweep loop."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
spec = importlib.util.spec_from_file_location("sc", "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/scripts/side22_classify.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
hold_stats, why_excluded = sc.hold_stats, sc.why_excluded

def cadence(z, sid, S, minlen=0, strict=True):
    # same index logic as hold_stats: t0 = first hold step, hi = min(t0+500, tend), a4 = t0+200
    seg = z["segment_id"]; idx = np.where((seg == sid) & (np.arange(len(seg)) < S))[0]
    t0 = idx[0]; tend = idx[-1]; hi = min(t0 + 500, tend); a4 = t0 + 200
    F = z["foot_force_norm"].astype(float) > 50
    n = 0
    for f in range(6):
        L = np.zeros(len(seg), bool); L[a4:hi] = F[a4:hi, f]
        d = np.diff(np.r_[0, L.astype(int), 0]); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
        n += sum(1 for a, b in zip(st, en) if b - a >= minlen and (a > a4 or not strict))
    return n / 6 / ((hi - a4) * sc.DT)

def read(d):
    d = Path(d); meta = json.loads((d / "run_meta.json").read_text())
    labels = meta["hold_labels"]; want = [l for l in labels if l.startswith("lat_")]
    tr = meta["termination_reason"]
    if meta.get("suite"):
        files = [(d / "raw" / f"episode_{e['env_index']:02d}.npz", e["label"], tr[e["env_index"]]) for e in meta["env_index_map"] if e["label"] in want]
    else:
        files = [(p, None, tr[int(p.stem.split('_')[1])]) for p in sorted((d / "raw").glob("episode_*.npz"))]
    out = {l: dict(rows=[], cad=[], cad5=[], excl={}, ep=0, fall_hold=0, fall_any=0) for l in want}
    for p, lab, reason in files:
        z = np.load(p); done = z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(z["segment_id"])
        seg = z["segment_id"]
        for l in ([lab] if lab else want):
            sid = labels.index(l); o = out[l]
            idx = np.where(seg == sid)[0]
            if lab is None and (len(idx) == 0 or idx[0] >= S):   # schedule stamp: episode never reached this hold
                continue
            o["ep"] += 1
            if reason == "fall":
                o["fall_any"] += 1
                if S < len(seg) and seg[S] == sid:
                    o["fall_hold"] += 1
            r = hold_stats(z, sid, S)
            if r:
                o["rows"].append(r); o["cad"].append(cadence(z, sid, S)); o["cad5"].append(cadence(z, sid, S, 0, False))
            else:
                w = why_excluded(z, sid, S)
                if w != "absent":
                    o["excl"][w] = o["excl"].get(w, 0) + 1
    res = {"checkpoint": meta.get("checkpoint"), "labels": {}}
    for l, o in out.items():
        a = {k: np.array([r[k] for r in o["rows"]], float) for k in ("feet_ss", "lean_abs", "sweep", "lean_lvl", "body_ss")}
        n = len(o["rows"]); st = o["excl"].get("static", 0)
        fs = a["feet_ss"]; la = a["lean_abs"]
        res["labels"][l] = dict(
            ep=o["ep"], n=n, excl=o["excl"],
            feet_ss=float(np.median(fs)), q25=float(np.percentile(fs, 25)), q75=float(np.percentile(fs, 75)),
            lean_abs=float(np.median(la)), lean_absabs=float(np.median(np.abs(la))),
            frac_lean015=float(np.mean(la > 0.15)), frac_abslean015=float(np.mean(np.abs(la) > 0.15)),
            frac_neg015=float(np.mean(la < -0.15)),
            sweep=float(np.nanmedian(a["sweep"])), cad=float(np.median(o["cad"])), cad5=float(np.median(o["cad5"])),
            stall=(int(np.sum(fs < 0.03)) + st) / (n + st), stall_n=f"{int(np.sum(fs < 0.03)) + st}/{n + st}",
            fall_hold=o["fall_hold"] / o["ep"], fall_hold_n=f"{o['fall_hold']}/{o['ep']}", fall_any=o["fall_any"] / o["ep"],
            walkers_feet_ss=float(np.median(fs[fs >= 0.03])) if (fs >= 0.03).any() else None,
            lean_sweep_rho=float(np.corrcoef(la, a["sweep"])[0, 1]) if n > 3 else None,
            feet_lean_rho=float(np.corrcoef(la, fs)[0, 1]) if n > 3 else None,
        )
    return res

G = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect"
O = "/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
arms = json.load(open(sys.argv[1]))
out = {}
for name, d in arms.items():
    out[name] = read(d); out[name]["stamp"] = d
    for l, r in out[name]["labels"].items():
        print(f"{name:14s} {l:10s} ep {r['ep']:3d} n {r['n']:3d} {r['excl']}  fss {r['feet_ss']:.3f} [{r['q25']:.3f},{r['q75']:.3f}] lean {r['lean_abs']:+.3f} |l| {r['lean_absabs']:.3f} f>.15 {r['frac_lean015']:.2f}/{r['frac_abslean015']:.2f} neg {r['frac_neg015']:.2f} sw {r['sweep']:.3f} cad {r['cad']:.3f}/{r['cad5']:.3f} stall {r['stall_n']} fallH {r['fall_hold_n']} fallAny {r['fall_any']:.2f}", flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
