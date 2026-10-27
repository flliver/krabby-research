import sys, json, numpy as np
from pathlib import Path
sys.argv=[sys.argv[0]]
exec(open("probe_hand.py").read().split("def windows")[0])
def yaw(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
for arm in ["O","R","C_band","C_rsi","LR4_dwell","LR4","LRfa","C_lr","C_lr_10","C","C_mo","C_mo_s2"]:
    d=Path(stamp(arm)); meta=json.loads((d/"run_meta.json").read_text()); labels=meta["hold_labels"]
    rows=[]
    for e in meta["env_index_map"]:
        lab=e["label"]
        if not lab.startswith("lat_"): continue
        z=np.load(d/"raw"/f"episode_{e['env_index']:02d}.npz")
        done=z["done"].astype(bool); S=int(np.argmax(done)) if done.any() else len(done)
        seg=z["segment_id"]; sid=labels.index(lab); idx=np.where((seg==sid)&(np.arange(len(seg))<S))[0]
        if len(idx)<500: continue
        t0=idx[0]; a4,hi=t0+200,t0+500
        rp=z["root_pos_w"][:,:2].astype(float); fc=z["foot_pos_w"][:,:,:2].astype(float).mean(1)
        psi=yaw(z["root_quat_w"].astype(float)); by=np.stack([-np.sin(psi),np.cos(psi)],1)
        sgn=1.0 if lab.startswith("lat_l") else -1.0   # travel along +body y for lat_l
        off0=sgn*((rp[t0-1]-fc[t0-1])*by[t0-1]).sum()   # + = body already offset toward travel
        u=sgn*by[t0]; v=((fc[hi]-fc[a4])@u)/((hi-a4)*0.02)
        rows.append((off0,v))
    R=np.array(rows); o,v=R[:,0],R[:,1]
    tow=o>0.05; opp=o<-0.05; cen=~(tow|opp)
    f=lambda m: f"n{m.sum():3d} feet {np.median(v[m]):.3f} stall {np.mean(v[m]<0.03):.2f}" if m.sum() else "n  0"
    print(f"{arm:9s} start leaning TOWARD travel: {f(tow)} | OPPOSITE: {f(opp)} | centred: {f(cen)} | corr(off0,v) {np.corrcoef(o,v)[0,1]:+.2f}")
