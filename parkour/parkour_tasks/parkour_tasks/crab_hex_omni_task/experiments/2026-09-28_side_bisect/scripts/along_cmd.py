"""Direction check for the side22 gate (CPU, read-only). Review finding CC-1, 2026-09-30.

feet_ss (side22_classify.py) is the feet-centroid speed along the robot's OWN net displacement over the hold, so a
wrong-way walker scores as fast. This script recomputes, per lat_* hold, the speed along the COMMANDED direction
(world travel_yaw, circular mean over the hold), with the classifier's own inclusion rules, and also:
  - wrong_way: included episodes whose net displacement is > 90 deg from the command;
  - static: episodes alive >= 8 s in the hold that moved < 0.05 m (the classifier drops them); along_med_s0 scores
    them 0 (finding EL-2);
  - class_along / class_along_r: the CHARTER gate applied to along_med_s0 (lat_l / lat_r medians).
Usage: along_cmd.py [--json OUT] NAME=STAMP_DIR [...]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("s22", Path(__file__).with_name("side22_classify.py"))
s22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s22)
DT = s22.DT


def hold_along(z, sid, S):
    seg = z["segment_id"]
    idx = np.where((seg == sid) & (np.arange(len(seg)) < S))[0]
    if len(idx) < 400:
        return None, "short" if len(idx) else "absent"
    t0, tend = idx[0], idx[-1]
    rp = z["root_pos_w"][:, :2].astype(float)
    fc = z["foot_pos_w"][:, :, :2].astype(float).mean(1)
    ty = z["travel_yaw"][t0:tend + 1].astype(float)
    c = np.array([np.cos(ty).mean(), np.sin(ty).mean()])
    c /= np.linalg.norm(c)
    hi = min(t0 + 500, tend)
    a4 = t0 + 200
    T = (hi - a4) * DT
    u = rp[tend] - rp[t0]
    n = np.linalg.norm(u)
    if n < 0.05:
        return 0.0, "static"
    along = float((fc[hi] - fc[a4]) @ c / T)
    wrong = bool((u / n) @ c < 0)
    return along, ("wrong" if wrong else "ok")


def read(d):
    meta = json.loads((d / "run_meta.json").read_text())
    labels = meta["hold_labels"]
    want = [l for l in labels if l.startswith("lat_")]
    if meta.get("suite"):
        files = [(d / "raw" / f"episode_{e['env_index']:02d}.npz", e["label"]) for e in meta["env_index_map"]
                 if e["label"] in want]
    else:
        files = [(p, None) for p in sorted((d / "raw").glob("episode_*.npz"))]
    acc = {l: {"along": [], "static": 0, "wrong": 0, "short": 0} for l in want}
    for p, lab in files:
        z = np.load(p)
        done = z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(z["segment_id"])
        for l in ([lab] if lab else want):
            v, why = hold_along(z, labels.index(l), S)
            if why in ("short", "absent"):
                if why == "short":
                    acc[l]["short"] += 1
                continue
            acc[l]["along"].append(v)
            if why == "static":
                acc[l]["static"] += 1
            elif why == "wrong":
                acc[l]["wrong"] += 1
    out = {}
    for l, a in acc.items():
        arr = np.array(a["along"], float)
        moving = arr[[i for i in range(len(arr))]] if len(arr) else arr
        out[l] = {"n_s0": len(arr), "static": a["static"], "wrong_way": a["wrong"], "short": a["short"],
                  "along_med_s0": float(np.median(arr)) if len(arr) else None,
                  "along_med_moving": float(np.median(arr[arr != 0.0])) if (arr != 0.0).any() else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("stamps", nargs="+")
    a = ap.parse_args()
    res = {}
    for spec in a.stamps:
        name, d = spec.split("=", 1)
        L = read(Path(d))
        r = {"stamp": d, "labels": L}
        for hand, key in (("l", "class_along"), ("r", "class_along_r")):
            k22, k25 = f"lat_{hand}_022", f"lat_{hand}_025"
            if k22 in L and k25 in L:
                r[key] = s22.classify(L[k22]["along_med_s0"], L[k25]["along_med_s0"], s22.GATE)
        res[name] = r
        print(f"== {name}")
        for l, v in L.items():
            f = lambda x: "  nan" if x is None else f"{x:+.3f}"
            print(f"   {l:10s} along(static=0) {f(v['along_med_s0'])}  along(moving) {f(v['along_med_moving'])}  "
                  f"n {v['n_s0']:3d}  static {v['static']:2d}  wrong_way {v['wrong_way']:2d}  short {v['short']:2d}")
        print(f"   CLASS_ALONG {r.get('class_along')}  CLASS_ALONG_R {r.get('class_along_r')}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
