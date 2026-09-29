"""CPU-1 (REPORT.md): the onset time course of side holds across the omni lineage.

Per 2-s window from the hold start: body speed and feet-centroid speed along travel, and the lean build (their
difference). Reads raw eval npz for the early formation stamps (slow__omni, per-episode segment layout; lat_l row) and
the arm/control suite stamps (suite__omni; lat_l and lat_r rows). The fwd row of 1a is included as a reference, since 1a
trained fwd only.
"""
import json, sys, numpy as np
from pathlib import Path
WIN, DT = 100, 0.02
G = Path("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases")
def newest(p): return sorted([q for q in p.iterdir() if q.is_dir()])[-1]
F = G / "2026-09-16_1805_omni_formation"
early = {f"1a_001": newest(F / "seed3_1a_001_A15pB/slow__omni/seed001"),
         f"1b_002": newest(F / "seed3_1b_002_A15pB/slow__omni/seed001"),
         f"1c_003": newest(F / "seed3_1c_003_A15pB/slow__omni/seed001"),
         f"1c_004": newest(F / "seed3_1c_004_A15pB/slow__omni/seed001")}
suites = {}
for camp in ("2026-09-26_stroke_posture", "2026-09-26_stroke_formation", "2026-09-27_stroke_cat_W", "2026-09-27_stroke_cat_t3"):
    for x in sorted((G / camp).iterdir()):
        s = x / "suite__omni" / "seed001"
        if s.is_dir(): suites[f"{camp[11:]}:{x.name.split('_A15pB')[0]}"] = newest(s)
suites["W"] = G / "2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29"
suites["X08_t3"] = G / "2026-09-23_clock_deprecation/step0_E3_X08-t3_visible/suite__omni/seed001/2026-09-23_21-38-11"

def windows(d, want):
    meta = json.loads((d / "run_meta.json").read_text()); labels = meta["hold_labels"]
    ids = [labels.index(w) for w in want if w in labels]
    if meta.get("suite"):
        files = [d / "raw" / f"episode_{e['env_index']:02d}.npz" for e in meta["env_index_map"] if e["label"] in want]
    else:
        files = sorted((d / "raw").glob("episode_*.npz"))
    res = {}
    for p in files:
        z = np.load(p); seg = z["segment_id"]; done = z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(seg) - 1
        for sid in ids:
            idx = np.where((seg == sid) & (np.arange(len(seg)) < S))[0]
            if len(idx) < 300: continue
            rp = z["root_pos_w"][:, :2].astype(float); fc = z["foot_pos_w"][:, :, :2].astype(float).mean(1)
            u = rp[idx[-1]] - rp[idx[0]]; n = np.linalg.norm(u)
            if n < 1e-6: continue
            u = u / n
            for w in range(len(idx) // WIN):
                a, b = idx[w * WIN], idx[(w + 1) * WIN - 1]; T = (b - a) * DT
                r = res.setdefault(w, {"body": [], "feet": []})
                r["body"].append((rp[b] - rp[a]) @ u / T); r["feet"].append((fc[b] - fc[a]) @ u / T)
    return res

def show(tag, d, want):
    res = windows(d, want)
    rows = []
    for w in sorted(res)[:5]:
        r = res[w]
        if len(r["body"]) < 5: continue
        b, f = np.median(r["body"]), np.median(r["feet"]); rows.append((w, b, f, b - f, len(r["body"])))
    cum_lean = sum(max(x[3], 0) * 2 for x in rows[:2])  # metres of lean built in the first 4 s (2-s windows)
    line = " | ".join(f"{b:.3f}/{f:.3f}" for _, b, f, _, _ in rows)
    steady = np.mean([f for w, _, f, _, _ in rows if w >= 2]) if len(rows) > 2 else float("nan")
    print(f"{tag:34s} {'+'.join(want):11s} body/feet per 2-s window: {line} | lean built 0-4 s ~{cum_lean:.2f} m | steady feet {steady:.3f} (n {rows[0][4] if rows else 0})")

print("== early formation (slow__omni, seed 3): 1a trained fwd only; 1b side-only fork of 1a; 1c_003 from 1b; 1c_004 from scratch")
for tag, d in early.items():
    show(tag, d, ["lat_l"])
show("1a_001 (fwd reference)", early["1a_001"], ["fwd"])
print("== suite stamps (lat_l + lat_r)")
for tag, d in suites.items():
    show(tag, d, ["lat_l", "lat_r"])
