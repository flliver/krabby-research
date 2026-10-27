"""Independent recomputation of the draft's signature statistics straight from raw/episode_*.npz (CPU, read-only).
Own implementation: hold window [t0+200,t0+500) of each lat_* segment, alive through t0+500; rod actions = channels 6..17
(hip x6, knee x6, leg order FL FR ML MR RL RR); M_y: leg swap, hip +1, knee -1; clock shift pi (12 of 24 bins).
Also: per-episode odd RMS (median/IQR), split-half, a NO-shift control (M_0), and the same statistic in joint-position space."""
import json, sys, numpy as np, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("sc", "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/2026-09-28_side_bisect/scripts/side22_classify.py")
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
B="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-28_side_bisect"
S="/home/nickmagus/krabby/krabby-research/parkour/outputs/omni_plan/side_bisect"
STAMPS={
 "O": f"{B}/seed3_X01_s1_A15pB/side22_25__omni/seed001/2026-09-29_05-50-19",
 "C": f"{B}/seed3_X02_s1_A15pB/side22_25__omni/seed001/2026-09-29_07-00-57",
 "C_mo_s2": f"{B}/seed2_X08_s1_A15pB/side22_25__omni/seed001/2026-09-29_19-46-10",
 "LR4": f"{S}/lr/LR4/side22_25lr__omni/seed001/2026-09-30_01-44-42",
 "C_lr": f"{S}/lr/C_lr/side22_25lr__omni/seed001/2026-09-30_05-11-37",
 "canon_C": f"{S}/canon/C/wave/side22_25lr__omni/seed001/2026-09-30_12-27-21",
 "C_lr_rsi33": f"{S}/lr/C_lr_rsi33/side22_25lr__omni/seed001/2026-09-30_17-42-44",
}
LEGS=["FL","FR","ML","MR","RL","RR"]; SW=np.array([1,0,3,2,5,4]); NB=24
P=np.concatenate([SW,SW+6]); Sg=np.concatenate([np.ones(6),-np.ones(6)])   # rods: hip +1, knee -1
def mirror(A,shift):  # A (NB,12)
    return Sg[None,:]*np.roll(A,-shift,axis=0)[:,P]
def pat(X,ph):  # X (E,T,C) mean-removed per episode; ph (E,T)
    Xc=X-X.mean(1,keepdims=True); b=(np.floor((ph%(2*np.pi))/(2*np.pi)*NB).astype(int))%NB
    A=np.zeros((NB,X.shape[-1])); n=np.zeros(NB)
    np.add.at(A,b.reshape(-1),Xc.reshape(-1,X.shape[-1])); np.add.at(n,b.reshape(-1),1)
    return A/np.maximum(n,1)[:,None]
def rms(A): return float(np.sqrt(np.mean(np.sum(A**2,-1))))
def stats(X,ph):
    A=pat(X,ph); Mpi=mirror(A,NB//2); M0=mirror(A,0)
    odd=(A-Mpi)/2; even=(A+Mpi)/2; odd0=(A-M0)/2
    E=len(X); h1=np.arange(0,E,2); h2=np.arange(1,E,2)
    A1=pat(X[h1],ph[h1]); A2=pat(X[h2],ph[h2])
    o1=rms((A1-mirror(A1,NB//2))/2); o2=rms((A2-mirror(A2,NB//2))/2)
    # per-episode odd rms
    pe=np.array([rms((pat(X[i:i+1],ph[i:i+1])-mirror(pat(X[i:i+1],ph[i:i+1]),NB//2))/2) for i in range(E)])
    m=X.mean(1).mean(0); so=float(np.linalg.norm((m-Sg*m[P])/2)); se=float(np.linalg.norm((m+Sg*m[P])/2))
    return dict(odd=rms(odd),even=rms(even),odd_h=(o1,o2),odd0=rms(odd0),amp=rms(A),pe_med=float(np.median(pe)),pe_q=(float(np.percentile(pe,25)),float(np.percentile(pe,75))),static_odd=so,static_even=se,odd_frac=float(np.sum(odd**2)/np.sum(A**2)))
rows=[]
for name,d in STAMPS.items():
    d=Path(d); meta=json.loads((d/"run_meta.json").read_text()); labels=meta["hold_labels"]
    jn=meta["joint_names"]; jidx={n:i for i,n in enumerate(jn)}
    jsel=[jidx[f"{L}_{t}_RevoluteJoint"] for t in ("Hip_Femur","Femur_Tibia") for L in LEGS]
    files=[(d/"raw"/f"episode_{e['env_index']:02d}.npz",e["label"]) for e in meta["env_index_map"] if e["label"].startswith("lat_")]
    per={}
    for p,lab in files:
        z=np.load(p); done=z["done"].astype(bool); Sd=int(np.argmax(done)) if done.any() else len(done)
        seg=z["segment_id"]; sid=labels.index(lab); idx=np.where((seg==sid)&(np.arange(len(seg))<Sd))[0]
        if len(idx)==0: continue
        hs=sc.hold_stats(z,sid,Sd); t0=idx[0]
        if hs is None or idx[-1]<t0+500: continue
        sl=slice(t0+200,t0+500)
        r=per.setdefault(lab,{"act":[],"jp":[],"ph":[],"fs":[],"sw":[],"lean":[],"canon":[]})
        r["act"].append(np.clip(z["actions"][sl,6:18],-1,1)); r["jp"].append(z["joint_pos"][sl][:,jsel]); r["ph"].append(z["clock_phase"][sl])
        r["fs"].append(hs["feet_ss"]); r["sw"].append(hs["sweep"]); r["lean"].append(hs["lean_abs"]); r["canon"].append(float(z["canon_flag"][sl].mean()) if "canon_flag" in z.files else np.nan)
    for lab,r in sorted(per.items()):
        X=np.array(r["act"]); J=np.array(r["jp"]); ph=np.array(r["ph"])
        a=stats(X,ph); j=stats(J,ph)
        row=dict(stamp=name,label=lab,n=len(X),feet_ss=float(np.median(r["fs"])),sweep=float(np.nanmedian(r["sw"])),lean=float(np.median(r["lean"])),canon=float(np.nanmean(r["canon"])),act=a,jp=j)
        rows.append(row)
        print(f"{name:10s} {lab:10s} n{len(X):3d} feet {row['feet_ss']:.3f} sweep {row['sweep']:.3f} lean {row['lean']:+.3f} canon {row['canon']:.2f} | ACT odd {a['odd']:.2f} [h {a['odd_h'][0]:.2f}/{a['odd_h'][1]:.2f}] per-ep med {a['pe_med']:.2f} IQR {a['pe_q'][0]:.2f}-{a['pe_q'][1]:.2f} even {a['even']:.2f} odd0(noshift) {a['odd0']:.2f} static_odd {a['static_odd']:.2f} frac {a['odd_frac']:.2f} | JP(rad) odd {j['odd']:.3f} even {j['even']:.3f} static_odd {j['static_odd']:.3f} frac {j['odd_frac']:.2f}")
json.dump(rows,open("/tmp/claude-1000/-home-nickmagus-krabby-krabby-research/4b0248a2-13ef-464e-81a9-16e93543e283/scratchpad/check/recompute.json","w"),indent=1,default=float)
