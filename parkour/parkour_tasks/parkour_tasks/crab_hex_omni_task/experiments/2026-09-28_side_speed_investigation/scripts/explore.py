import sys, numpy as np, math
sys.path.insert(0, "/tmp/claude-1000/sidespeed")
import importlib.util
spec = importlib.util.spec_from_file_location("k3", "/tmp/claude-1000/sidespeed/kin3.py")
src = open("/tmp/claude-1000/sidespeed/kin3.py").read().split("res = {}")[0]
g = {}; exec(src, g)
from scipy.stats import norm
SIG = 2.0
def offrail(mu):
    return norm.cdf((1-mu)/SIG) - norm.cdf((-1-mu)/SIG)
for tag, label in [("W", "lat_l"), ("t3", "lat_l"), ("SH", "side"), ("W", "fwd"), ("1b_002", "lat_l")]:
    acc = {}
    for z, meta, ids in g["eps"](tag, label):
        seg = z["segment_id"].astype(int); done = z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(seg)-1
        m = np.isin(seg, ids) & z["steady_mask"].astype(bool) & (np.arange(len(seg)) < S)
        if m.sum() < 50: continue
        cmd = z["cmd_applied"][:, :2].astype(float)
        if label == "fwd": lead = np.ones(6, bool)
        else:
            sgn = 1.0 if label == "side" else np.sign(np.median(cmd[m, 1]))
            lead = ~g["LEFT"] if sgn > 0 else g["LEFT"]
        a = z["actions"].astype(float)
        ks = np.where(g["LEFT"], 1.0, -1.0)
        for f in range(6):
            row = "lead" if lead[f] else "trail"
            for ch, col, sg in [("cam", f, 1.0), ("hip", 6+f, 1.0), ("knee", 12+f, ks[f])]:
                mu = a[m, col]*sg
                acc.setdefault(f"{row}.{ch}.mu_p50", []).append(np.median(mu))
                acc.setdefault(f"{row}.{ch}.absmu_p50", []).append(np.median(np.abs(mu)))
                acc.setdefault(f"{row}.{ch}.P_offrail_mean", []).append(offrail(mu).mean())
                acc.setdefault(f"{row}.{ch}.frac_P_lt1pct", []).append((offrail(mu) < 0.01).mean())
    print("==", tag, label)
    for k in sorted(acc): print(f"  {k:28s} {np.median(acc[k]):8.3f}")
