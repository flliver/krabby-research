"""Side-speed bisection gate: feet_ss on the side22_25 rows (CPU, read-only).

Usage: side22_classify.py [--thresholds JSON] [--json OUT] NAME=STAMP_DIR [NAME=STAMP_DIR ...]

Each STAMP_DIR is one eval stamp: side22_25__omni (category_suite) or side22_25__side (schedule), or a
GPU-0 sidesweep stamp (labels lat_l/lat_r). For every hold label present it reports n, the median of the
per-episode feet_ss (the class statistic), body_ss, lean04, lean_lvl and sweep (labels only), plus the
excluded episodes (died before 8 s into the hold, or moved < 0.05 m over it). The class applies the
pre-registered rule (CHARTER "Gate") to the lat_l_022 / lat_l_025 medians.

hold_stats is copied VERBATIM from the scoping script (scope_bisect/gate_stats.py) so the statistic
the thresholds came from is the statistic read here.
"""
import argparse
import json
from pathlib import Path

import numpy as np

DT = 0.02


def yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def hold_stats(z, sid, S):
    seg = z["segment_id"]
    idx = np.where((seg == sid) & (np.arange(len(seg)) < S))[0]
    if len(idx) < 400:          # need >= 8 s alive
        return None
    t0 = idx[0]; tend = idx[-1]
    rp = z["root_pos_w"][:, :2].astype(float); fc = z["foot_pos_w"][:, :, :2].astype(float).mean(1)
    u = rp[tend] - rp[t0]; n = np.linalg.norm(u)
    if n < 0.05:
        return None
    u /= n
    hi = min(t0 + 500, tend)
    a4 = t0 + 200
    T = (hi - a4) * DT
    feet_ss = (fc[hi] - fc[a4]) @ u / T
    body_ss = (rp[hi] - rp[a4]) @ u / T
    h10 = min(t0 + 500, tend)
    body_hold = (rp[h10] - rp[t0]) @ u / ((h10 - t0) * DT)
    lean04 = ((rp[a4] - rp[t0]) - (fc[a4] - fc[t0])) @ u
    off = (rp - fc) @ u
    lean_lvl = np.median(off[a4:hi]) - off[t0]
    lean_abs = np.median(off[a4:hi])
    psi = yaw(z["root_quat_w"].astype(float))
    fp = z["foot_pos_w"][:, :, :2].astype(float) - rp[:, None, :]
    ub_ang = np.arctan2(u[1], u[0]) - psi
    c, s = np.cos(psi), np.sin(psi)
    fbx = c[:, None] * fp[..., 0] + s[:, None] * fp[..., 1]
    fby = -s[:, None] * fp[..., 0] + c[:, None] * fp[..., 1]
    ub = np.array([np.cos(np.median(ub_ang[t0:tend])), np.sin(np.median(ub_ang[t0:tend]))])
    fa = fbx * ub[0] + fby * ub[1]
    F = z["foot_force_norm"].astype(float) > 50
    sw = []
    for f in range(6):
        L = np.zeros(len(seg), bool); L[a4:hi] = F[a4:hi, f]
        d = np.diff(np.r_[0, L.astype(int), 0]); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
        sw += [abs(fa[b - 1, f] - fa[a, f]) for a, b in zip(st, en) if b - a >= 5]
    sweep = float(np.median(sw)) if sw else np.nan
    return dict(feet_ss=feet_ss, body_ss=body_ss, body_hold=body_hold, lean04=lean04, lean_lvl=lean_lvl,
                lean_abs=lean_abs, sweep=sweep)


def why_excluded(z, sid, S):
    seg = z["segment_id"]
    idx = np.where((seg == sid) & (np.arange(len(seg)) < S))[0]
    if len(idx) == 0:
        return "absent"
    if len(idx) < 400:
        return "short"
    rp = z["root_pos_w"][:, :2].astype(float)
    return "static" if np.linalg.norm(rp[idx[-1]] - rp[idx[0]]) < 0.05 else "?"


def read_stamp(d: Path):
    meta = json.loads((d / "run_meta.json").read_text())
    labels = meta["hold_labels"]
    want = [l for l in labels if l.startswith("lat_")]
    rows = {l: [] for l in want}
    excl = {l: {} for l in want}
    if meta.get("suite"):
        files = [(d / "raw" / f"episode_{e['env_index']:02d}.npz", e["label"]) for e in meta["env_index_map"]
                 if e["label"] in want]
    else:
        files = [(p, None) for p in sorted((d / "raw").glob("episode_*.npz"))]
    for p, lab in files:
        z = np.load(p)
        done = z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(z["segment_id"])
        for l in ([lab] if lab else want):
            r = hold_stats(z, labels.index(l), S)
            if r:
                rows[l].append(r)
            else:
                w = why_excluded(z, labels.index(l), S)
                if w != "absent":
                    excl[l][w] = excl[l].get(w, 0) + 1
    return meta, rows, excl


GATE = {"fast": (0.155, 0.175), "slow": (0.13, 0.145), "dead": 0.05}


def classify(m22, m25, gate):
    if m22 is None or m25 is None or not np.isfinite(m22) or not np.isfinite(m25):
        return "DEAD?(n=0/NaN)"
    if m22 < gate["dead"] or m25 < gate["dead"]:
        return "DEAD"
    if m22 >= gate["fast"][0] and m25 >= gate["fast"][1]:
        return "FAST"
    if m22 <= gate["slow"][0] and m25 <= gate["slow"][1]:
        return "SLOW"
    return "INTERMEDIATE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", default=None, help="JSON {fast:[a,b], slow:[a,b], dead:x}")
    ap.add_argument("--json", default=None)
    ap.add_argument("stamps", nargs="+")
    a = ap.parse_args()
    gate = dict(GATE)
    if a.thresholds:
        gate.update(json.loads(Path(a.thresholds).read_text()))
    out = {}
    for spec in a.stamps:
        name, d = spec.split("=", 1)
        d = Path(d)
        meta, rows, excl = read_stamp(d)
        res = {"stamp": str(d), "checkpoint": meta.get("checkpoint"),
               "env_vars": meta.get("env_vars"), "labels": {}}
        for l, rs in rows.items():
            arr = {k: np.array([r[k] for r in rs], float) for k in
                   ("feet_ss", "body_ss", "body_hold", "lean04", "lean_lvl", "lean_abs", "sweep")}
            med = {k: (float(np.nanmedian(v)) if len(v) else None) for k, v in arr.items()}
            res["labels"][l] = {"n": len(rs), "excluded": excl[l], **med,
                                "feet_ss_q25": float(np.nanpercentile(arr["feet_ss"], 25)) if len(rs) else None,
                                "feet_ss_q75": float(np.nanpercentile(arr["feet_ss"], 75)) if len(rs) else None,
                                "feet_ss_all": arr["feet_ss"].tolist()}
        L = res["labels"]
        if "lat_l_022" in L and "lat_l_025" in L:
            res["class"] = classify(L["lat_l_022"]["feet_ss"], L["lat_l_025"]["feet_ss"], gate)
        out[name] = res
        print(f"== {name}  {d}")
        print(f"   ckpt {meta.get('checkpoint')}")
        for l, r in L.items():
            f = lambda x: "  nan" if x is None else f"{x:+.3f}"
            print(f"   {l:10s} n {r['n']:3d} excl {r['excluded']}  feet_ss {f(r['feet_ss'])} "
                  f"[{f(r['feet_ss_q25'])},{f(r['feet_ss_q75'])}]  body_ss {f(r['body_ss'])}  "
                  f"lean04 {f(r['lean04'])}  lean_lvl {f(r['lean_lvl'])}  sweep {f(r['sweep'])}")
        if "class" in res:
            print(f"   CLASS {res['class']}  (gate {gate})")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
