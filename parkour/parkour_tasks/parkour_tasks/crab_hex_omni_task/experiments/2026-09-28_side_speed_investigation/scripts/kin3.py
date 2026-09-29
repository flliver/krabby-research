import sys, math, json, glob
import numpy as np, torch
sys.path.insert(0, "/tmp/claude-1000/sidespeed")
from kin import L, LEGS, LEFT, heading, jidx, STAMPS, F_LOAD
from pathlib import Path
L1, L2 = L.FEMUR_LEN_M, 0.8255
FORM = Path("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-16_1805_omni_formation")
def stamp_of(tag):
    if tag in STAMPS: return STAMPS[tag]
    return Path(glob.glob(str(FORM / f"seed3_{tag}_A15pB/slow__omni/seed001/*"))[0])

def eps(tag, label):
    d = stamp_of(tag); meta = json.loads((d / "run_meta.json").read_text()); labels = meta["hold_labels"]
    if meta.get("suite"):
        for ent in meta["env_index_map"]:
            if ent["label"] == label:
                yield dict(np.load(d / "raw" / f"episode_{ent['env_index']:02d}.npz")), meta, [labels.index(label)]
    else:
        ids = [labels.index("creep"), labels.index("low")] if label == "side" else [labels.index(label)]
        for p in sorted((d / "raw").glob("episode_*.npz")):
            yield dict(np.load(p)), meta, ids

def run(tag, label):
    acc = {}
    def add(k, v): acc.setdefault(k, []).append(v)
    for z, meta, ids in eps(tag, label):
        seg = z["segment_id"].astype(int); done = z["done"].astype(bool)
        S = int(np.argmax(done)) if done.any() else len(seg) - 1
        m = np.isin(seg, ids) & z["steady_mask"].astype(bool) & (np.arange(len(seg)) < S)
        if m.sum() < 50: continue
        dt = 0.02
        cmd = z["cmd_applied"][:, :2].astype(np.float64)
        if not meta.get("suite") and label == "side":
            ty = meta["travel_yaw"]; cmd = np.stack([cmd[:, 0]*math.cos(ty)-cmd[:, 1]*math.sin(ty), cmd[:, 0]*math.sin(ty)+cmd[:, 1]*math.cos(ty)], -1)
        u = cmd / np.maximum(np.linalg.norm(cmd, axis=-1, keepdims=True), 1e-9)
        vb = z["root_lin_vel_b"][:, :2].astype(np.float64); vs = np.zeros_like(vb); vs[:-1] = vb[1:]
        add("speed", float((vs*u)[m].sum(-1).mean()))
        lateral = abs(np.median(u[m, 1])) > 0.7
        # travel-side legs: R legs are at +y (FL at -y in this USD)
        if lateral:
            sgn = np.sign(np.median(u[m, 1]))
            lead_leg = ~LEFT if sgn > 0 else LEFT
        else:
            lead_leg = None
        jp = z["joint_pos"].astype(np.float64); a = np.clip(z["actions"].astype(np.float64), -1, 1)
        ih = jidx(meta, "Hip_Femur_RevoluteJoint"); ik = jidx(meta, "Femur_Tibia_RevoluteJoint")
        ks = np.where(LEFT, 1.0, -1.0); th = jp[:, ih]; tk = jp[:, ik]*ks; phi = th+tk
        ut = L1*np.cos(th) - L2*np.sin(phi)
        dudh = -L1*np.sin(th) - L2*np.cos(phi); dudk = -L2*np.cos(phi)
        ak = a[:, 12:18]*ks; ah = a[:, 6:12]
        F = z["foot_force_norm"].astype(np.float64); loaded = F > F_LOAD
        idx = np.where(m)[0]; lo, hi = idx.min(), idx.max(); dur = (hi-lo+1)*dt
        for f in range(6):
            row = ("lead" if lead_leg[f] else "trail") if lead_leg is not None else "all"
            add(f"{row}.u_p50", float(np.median(ut[m, f])))
            add(f"{row}.u_p5", float(np.percentile(ut[m, f], 5))); add(f"{row}.u_p95", float(np.percentile(ut[m, f], 95)))
            add(f"{row}.knee_rail_pos", float((ak[m, f] >= 0.98).mean())); add(f"{row}.knee_rail_neg", float((ak[m, f] <= -0.98).mean()))
            add(f"{row}.hip_rail", float((np.abs(ah[m, f]) >= 0.98).mean()))
            sk = np.sign(ak[lo:hi+1, f]); sk = sk[sk != 0]
            add(f"{row}.knee_flips_per_s", float((np.diff(sk) != 0).sum()/dur))
            sh = np.sign(ah[lo:hi+1, f]); sh = sh[sh != 0]
            add(f"{row}.hip_flips_per_s", float((np.diff(sh) != 0).sum()/dur))
            add(f"{row}.knee_deg_p5", float(np.degrees(np.percentile(tk[m, f], 5)))); add(f"{row}.knee_deg_p95", float(np.degrees(np.percentile(tk[m, f], 95))))
            # stance du decomposition
            ldw = loaded[lo:hi+1, f]; d = np.diff(np.concatenate([[0], ldw.astype(int), [0]]))
            st = np.where(d == 1)[0]; en = np.where(d == -1)[0]; hs = []; ks_ = []; tot = []
            for a0, a1 in zip(st, en):
                if a0 == 0 or a1 == len(ldw) or a1-a0 < 5: continue
                t0, t1 = lo+a0, lo+a1
                dth = np.diff(th[t0:t1, f]); dtk = np.diff(tk[t0:t1, f])
                hs.append(float((dudh[t0:t1-1, f]*dth).sum())); ks_.append(float((dudk[t0:t1-1, f]*dtk).sum())); tot.append(float(ut[t1-1, f]-ut[t0, f]))
            if tot:
                add(f"{row}.stance_du", float(np.median(tot))); add(f"{row}.stance_du_hip", float(np.median(hs))); add(f"{row}.stance_du_knee", float(np.median(ks_)))
    return {k: float(np.nanmedian(v)) for k, v in acc.items()}

res = {}
for tag, label in [("W", "lat_l"), ("W", "lat_r"), ("t3", "lat_l"), ("SH", "side"), ("1b_002", "lat_l"), ("1a_001", "lat_l"), ("1c_004", "lat_l"), ("2b_006", "lat_l"), ("W", "fwd"), ("1a_001", "fwd"), ("SH", "stand")] :
    try:
        if tag == "SH" and label == "stand": continue
        res[f"{tag}:{label}"] = run(tag, label)
    except Exception as e:
        print(tag, label, "ERR", e)
json.dump(res, open("/tmp/claude-1000/sidespeed/kin3.json", "w"), indent=1)
keys = sorted({k for r in res.values() for k in r})
cols = list(res)
print(f"{'metric':24s}" + "".join(f"{c[:11]:>12s}" for c in cols))
for k in keys: print(f"{k:24s}" + "".join(f"{res[c].get(k, np.nan):12.3f}" for c in cols))
