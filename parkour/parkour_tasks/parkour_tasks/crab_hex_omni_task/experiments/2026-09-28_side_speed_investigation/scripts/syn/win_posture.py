# Within-hold time course: does W's side speed decay track the feet drifting to the workspace ends?
import sys, json, numpy as np
sys.path.insert(0,"/tmp/claude-1000/sidespeed")
from kin import STAMPS
def rot_inv(q, v):
    w,x,y,z=[q[...,i] for i in range(4)]
    psi=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)); c,s=np.cos(psi),np.sin(psi)
    return np.stack([c*v[...,0]+s*v[...,1], -s*v[...,0]+c*v[...,1]],-1)
WIN=100
def run(tag,labels_want,segs=None):
    d=STAMPS[tag]; meta=json.loads((d/"run_meta.json").read_text()); labels=meta["hold_labels"]
    bn=meta["body_names"]; fe=[bn.index(f"{l}_Femur") for l in ["FL","FR","ML","MR","RL","RR"]]
    if meta.get("suite"):
        files=[(d/"raw"/f"episode_{e['env_index']:02d}.npz",[labels.index(e['label'])]) for e in meta["env_index_map"] if e["label"] in labels_want]
    else:
        files=[(p,[labels.index(s) for s in segs]) for p in sorted((d/"raw").glob("episode_*.npz"))]
    res={}
    for p,ids in files:
        z=np.load(p); seg=z["segment_id"]; done=z["done"].astype(bool); S=int(np.argmax(done)) if done.any() else len(seg)-1
        m=np.isin(seg,ids)&(np.arange(len(seg))<S)
        idx=np.where(m)[0]
        if len(idx)<300: continue
        q=z["root_quat_w"].astype(float); rp=z["root_pos_w"].astype(float)
        fp=z["foot_pos_w"][:,:,:2].astype(float); bp=z["body_pos_w"][:,:,:2].astype(float)
        fb=rot_inv(q[:,None,:], fp-rp[:,None,:2]); hb=rot_inv(q[:,None,:], bp[:,fe]-rp[:,None,:2])
        vb=z["root_lin_vel_b"][:,:2].astype(float)
        ydir=np.sign(vb[idx,1].mean())
        rel=fb[...,1]-hb[...,1]; side=np.sign(np.median(hb[idx][:,:,1],0)); outb=rel*side; lead=side==ydir
        F=z["foot_force_norm"].astype(float)>50
        for w in range(len(idx)//WIN):
            ii=idx[w*WIN:(w+1)*WIN]; r=res.setdefault(w,{})
            r.setdefault("speed",[]).append(vb[ii,1].mean()*ydir)
            r.setdefault("trail_outb",[]).append(np.median(outb[ii][:,~lead]))
            r.setdefault("lead_outb",[]).append(np.median(outb[ii][:,lead]))
            r.setdefault("lean",[]).append(np.median(-fb[ii].mean(1)[:,1]*ydir))
            sw=[]
            for f in range(6):
                L=np.zeros(len(seg),bool); L[ii]=F[ii,f]; dd=np.diff(np.r_[0,L.astype(int),0]); st=np.where(dd==1)[0]; en=np.where(dd==-1)[0]
                sw+= [abs(outb[b-1,f]-outb[a,f]) for a,b in zip(st,en) if b-a>=5]
            r.setdefault("sweep",[]).append(np.median(sw) if sw else np.nan)
            nst=sum(1 for f in range(6) for _ in [0] if True)
    print(f"== {tag} {labels_want or segs}  (windows of {WIN} steps = {WIN*0.02:.0f} s from hold start; n episodes per window)")
    print(" w     n  speed  trail_outb  lead_outb  lean   stance_sweep")
    for w in sorted(res):
        r=res[w]; n=len(r["speed"])
        if n<5: continue
        f=lambda k: np.nanmedian(r[k])
        print(f"{w:2d} {n:5d}  {f('speed'):.3f}   {f('trail_outb'):+.3f}     {f('lead_outb'):+.3f}   {f('lean'):.3f}   {f('sweep'):.3f}")
    # per-episode correlation of window speed vs posture
    S=[];T=[];L=[];SW=[];LE=[]
    for w in res:
        S+=res[w]["speed"];T+=res[w]["trail_outb"];L+=res[w]["lead_outb"];SW+=res[w]["sweep"];LE+=res[w]["lean"]
    S,T,L,SW,LE=map(np.array,(S,T,L,SW,LE)); ok=~np.isnan(SW)
    print(" corr(speed, trail_outb) %.2f  corr(speed, lead_outb) %.2f  corr(speed, sweep) %.2f  corr(speed, lean) %.2f  N=%d"%(
        np.corrcoef(S,T)[0,1],np.corrcoef(S,L)[0,1],np.corrcoef(S[ok],SW[ok])[0,1],np.corrcoef(S,LE)[0,1],len(S)))
run("W",["lat_l","lat_r"])
run("t3",["lat_l","lat_r"])
run("W",["fwd"]) if False else None
run("SH",None,["creep"])
