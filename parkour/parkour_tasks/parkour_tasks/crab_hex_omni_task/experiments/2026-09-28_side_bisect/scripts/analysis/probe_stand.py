import sys, json, glob, numpy as np
from pathlib import Path
sys.argv=[sys.argv[0]]
exec(open("probe_hand.py").read().split("def windows")[0])
def yaw(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]; return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
arms=["C","C_s2","C_mo_s2","C_mo_s1","C_mo","O","R","C_band","C_rsi","LR4","LRfa","C_lr","C_lr_10","LR4_dwell"]
for arm in arms:
    d=Path(stamp(arm)); meta=json.loads((d/"run_meta.json").read_text()); labels=meta["hold_labels"]
    offs=[]; sins=[]; offs_stand=[]
    for e in meta["env_index_map"]:
        z=np.load(d/"raw"/f"episode_{e['env_index']:02d}.npz")
        seg=z["segment_id"]; rp=z["root_pos_w"][:,:2].astype(float); fc=z["foot_pos_w"][:,:,:2].astype(float).mean(1)
        psi=yaw(z["root_quat_w"].astype(float)); by=np.stack([-np.sin(psi),np.cos(psi)],1)
        off=((rp-fc)*by).sum(1)
        lab=e["label"]; sid=labels.index(lab); idx=np.where(seg==sid)[0]
        if len(idx)==0: continue
        t0=idx[0]
        if lab=="stand":
            if len(idx)>=500: offs_stand.append(np.median(off[idx[-100:]]))
        else:
            offs.append(off[t0-1]); sins.append(np.sin(z["clock_phase"][t0-1]))
    offs=np.array(offs); sins=np.array(sins); st=np.array(offs_stand)
    r=np.corrcoef(offs,sins)[0,1] if offs.std()>0 and sins.std()>0 else np.nan
    print(f"{arm:9s} hold-start body-y offset: med {np.median(offs):+.3f} |med| {np.median(np.abs(offs)):.3f} frac>+0.05 {np.mean(offs>0.05):.2f} frac<-0.05 {np.mean(offs<-0.05):.2f} corr(off,sin clock) {r:+.2f} | stand-hold end: med {np.median(st):+.3f} |med| {np.median(np.abs(st)):.3f} frac>+.05 {np.mean(st>0.05):.2f} <-.05 {np.mean(st<-0.05):.2f}")
