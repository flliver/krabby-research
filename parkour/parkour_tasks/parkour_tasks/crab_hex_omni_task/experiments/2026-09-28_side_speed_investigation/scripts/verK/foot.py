import sys, json, math, numpy as np
sys.path.insert(0,"/tmp/claude-1000/sidespeed")
from kin import STAMPS
from pathlib import Path
def rot_inv(q, v):
    w,x,y,z=[q[...,i] for i in range(4)]
    psi=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)); c,s=np.cos(psi),np.sin(psi)
    return np.stack([c*v[...,0]+s*v[...,1], -s*v[...,0]+c*v[...,1]],-1)
def run(tag,label,segs=None):
    d=STAMPS[tag]; meta=json.loads((d/"run_meta.json").read_text()); labels=meta["hold_labels"]
    bn=meta["body_names"]; hips=[bn.index(f"{l}_Hip") for l in ["FL","FR","ML","MR","RL","RR"]]; fe=[bn.index(f"{l}_Femur") for l in ["FL","FR","ML","MR","RL","RR"]]
    out={}
    if meta.get("suite"):
        files=[(d/"raw"/f"episode_{e['env_index']:02d}.npz",[labels.index(label)]) for e in meta["env_index_map"] if e["label"]==label]
    else:
        files=[(p,[labels.index(s) for s in segs]) for p in sorted((d/"raw").glob("episode_*.npz"))]
    for p,ids in files:
        z=np.load(p); seg=z["segment_id"]; done=z["done"].astype(bool); S=int(np.argmax(done)) if done.any() else len(seg)-1
        m=np.isin(seg,ids)&z["steady_mask"].astype(bool)&(np.arange(len(seg))<S)
        if m.sum()<50: continue
        q=z["root_quat_w"].astype(float); rp=z["root_pos_w"].astype(float)
        fp=z["foot_pos_w"][:,:,:2].astype(float); bp=z["body_pos_w"][:,:,:2].astype(float)
        fb=rot_inv(q[:,None,:], fp-rp[:,None,:2]); hb=rot_inv(q[:,None,:], bp[:,fe]-rp[:,None,:2])
        # travel dir in body frame from velocity
        v=z["root_lin_vel_b"][m,:2].mean(0); ydir=np.sign(v[1])
        rel=(fb[...,1]-hb[...,1])  # foot minus femur pivot lateral
        side=np.sign(np.median(hb[m][:,:,1],0))  # each leg's side
        outb=rel*side  # outboard distance
        lead=side==ydir
        F=z["foot_force_norm"].astype(float)>50
        for f in range(6):
            r="lead" if lead[f] else "trail"
            out.setdefault(r+".outb_p50",[]).append(np.median(outb[m,f]))
            # stance sweep of foot relative to femur pivot (outb change over loaded runs)
            L=F[:,f]&m; dd=np.diff(np.r_[0,L.astype(int),0]); st=np.where(dd==1)[0]; en=np.where(dd==-1)[0]
            sw=[outb[b-1,f]-outb[a,f] for a,b in zip(st,en) if b-a>=5]
            if sw: out.setdefault(r+".stance_d_outb",[]).append(np.median(sw))
        out.setdefault("speed",[]).append(abs(v[1])); out.setdefault("vx",[]).append(v[0])
        out.setdefault("lean_yb",[]).append(np.median((0-fb[m].mean(1)[:,1])*ydir))
    return {k:round(float(np.median(v)),3) for k,v in out.items()}
for a in [("W","lat_l"),("W","lat_r"),("t3","lat_l"),("W","fwd")]: print(a,run(*a))
print("SH creep",run("SH",None,["creep"])); print("SH low",run("SH",None,["low"]))
