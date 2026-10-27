"""CPU probe: handedness vs direction-agnostic structure of side gaits (read-only)."""
import json, sys, glob
from pathlib import Path
import numpy as np

LOGS = Path("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect")
LR = Path("/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect/lr")
TAG = {"O":"seed3_X01","C":"seed3_X02","R":"seed3_X03","C_mo":"seed3_X04","C_band":"seed3_X05","C_rsi":"seed3_X06",
       "C_s2":"seed2_X07","C_mo_s2":"seed2_X08","C_mo_s1":"seed1_X09","LRfa":"seed3_X10","LR4":"seed3_X11",
       "LR4_dwell":"seed3_X12","C_lr":"seed3_X14","C_lr_10":"seed3_X15"}
LEGS = ["FL","FR","ML","MR","RL","RR"]
SWAP = {"FL":"FR","FR":"FL","ML":"MR","MR":"ML","RL":"RR","RR":"RL"}
def sgn(n):
    for t,s in (("_Body_CamShaft_",-1.),("_Body_Hip_",-1.),("_Hip_Femur_",1.),("_Femur_Tibia_",-1.)):
        if t in n: return s
def mymap(names):
    idx = {n:i for i,n in enumerate(names)}
    P = np.array([idx[SWAP[n[:2]]+n[2:]] for n in names]); S = np.array([sgn(n) for n in names])
    return P, S  # mirrored[i] = S[i]*x[P[i]]

def stamp(arm):
    if arm in ("LRfa","LR4","LR4_dwell","C_lr","C_lr_10"):
        return sorted(glob.glob(str(LR/arm/"side22_25lr__omni/seed001/*/")))[-1]
    return sorted(glob.glob(str(LOGS/(TAG[arm]+"_s1_A15pB")/"side22_25__omni/seed001/*/")))[-1]

def windows(d):
    d = Path(d); meta = json.loads((d/"run_meta.json").read_text()); labels = meta["hold_labels"]
    out = {}
    for e in meta["env_index_map"]:
        lab = e["label"]
        z = np.load(d/"raw"/f"episode_{e['env_index']:02d}.npz")
        done = z["done"].astype(bool); S = int(np.argmax(done)) if done.any() else len(done)
        sid = labels.index(lab); seg = z["segment_id"]
        idx = np.where((seg == sid) & (np.arange(len(seg)) < S))[0]
        if len(idx) < 500: continue
        t0 = idx[0]; a4, hi = t0+200, t0+500
        out.setdefault(lab, []).append(dict(a=z["actions"][a4:hi].astype(float), q=z["joint_pos"][a4:hi].astype(float),
                                           F=z["foot_force_norm"][a4:hi].astype(float)))
    return meta, out

def corrmat(X):
    X = X - X.mean(0); s = X.std(0); s[s < 1e-9] = 1e-9
    return (X/s).T @ (X/s) / len(X)

res = {}
for arm in sys.argv[1:]:
    meta, W = windows(stamp(arm))
    an = meta["action_joint_names"]; jn = meta["joint_names"]
    Pa, Sa = mymap(an); Pq, Sq = mymap(jn)
    r = {}
    for lab, eps in W.items():
        A = np.stack([e["a"] for e in eps]); Q = np.stack([e["q"] for e in eps])
        sat = (np.abs(A) >= 1.0).mean(axis=(0,1))
        r[lab] = dict(n=len(eps), sat_cam=float(sat[:6].mean()), sat_hip=float(sat[6:12].mean()), sat_knee=float(sat[12:].mean()),
                      med_abs_a=float(np.median(np.abs(A))),
                      duty=(np.sign(A) > 0).mean(axis=(0,1)),  # per-channel fraction positive
                      amean=np.clip(A,-1,1).mean(axis=(0,1)),
                      qstd=Q.std(axis=1).mean(0), qmean=Q.mean(axis=(0,1)),
                      C=np.mean([corrmat(np.clip(e["a"],-1,1)[:,6:]) for e in eps], 0),
                      Chalf=(np.mean([corrmat(np.clip(e["a"],-1,1)[:,6:]) for e in eps[0::2]],0),
                             np.mean([corrmat(np.clip(e["a"],-1,1)[:,6:]) for e in eps[1::2]],0)))
    res[arm] = r
    print(f"== {arm}")
    for lab, v in r.items():
        print(f"  {lab:10s} n{v['n']:3d} sat cam {v['sat_cam']:.2f} hip {v['sat_hip']:.2f} knee {v['sat_knee']:.2f} med|a| {v['med_abs_a']:.2f}")
    # handedness tests on clipped action means and coordination matrices
    for spd in ("022","025"):
        L, Rr = r.get(f"lat_l_{spd}"), r.get(f"lat_r_{spd}")
        if not (L and Rr): continue
        aL, aR = L["amean"], Rr["amean"]
        hand = Sa * aL[Pa]
        print(f"  [{spd}] clipped-mean dist: R vs L(agnostic) {np.linalg.norm(aR-aL):.2f} | R vs M_y L(handed) {np.linalg.norm(aR-hand):.2f} | |L| {np.linalg.norm(aL):.2f}")
        # coordination matrices over rod channels (6..17), M_y map restricted
        Pr = Pa[6:]-6; Sr = Sa[6:]
        CL, CR = L["C"], Rr["C"]
        CLm = (Sr[:,None]*Sr[None,:]) * CL[np.ix_(Pr,Pr)]
        h1, h2 = L["Chalf"]
        print(f"  [{spd}] rod coord-matrix dist: R vs L {np.linalg.norm(CR-CL):.2f} | R vs M_y L {np.linalg.norm(CR-CLm):.2f} | L split-half noise {np.linalg.norm(h1-h2):.2f}")
        qL, qR = L["qmean"], Rr["qmean"]
        print(f"  [{spd}] joint-mean dist: R vs L {np.linalg.norm(qR-qL):.3f} | R vs M_y L {np.linalg.norm(qR - Sq*qL[Pq]):.3f}")
