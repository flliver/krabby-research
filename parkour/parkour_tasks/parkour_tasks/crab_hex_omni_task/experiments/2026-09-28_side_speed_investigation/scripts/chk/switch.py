import sys, json, numpy as np, glob
sys.path.insert(0,"/tmp/claude-1000/sidespeed")
from kin import L, LEFT, jidx
from pathlib import Path
L1, L2 = L.FEMUR_LEN_M, 0.8255
def u_of(z,meta):
    jp=z["joint_pos"].astype(float); ih=jidx(meta,"Hip_Femur_RevoluteJoint"); ik=jidx(meta,"Femur_Tibia_RevoluteJoint")
    ks=np.where(LEFT,1.0,-1.0); th=jp[:,ih]; tk=jp[:,ik]*ks; phi=th+tk
    return L1*np.cos(th)-L2*np.sin(phi)
def rot_inv(q, v):
    w,x,y,z=[q[...,i] for i in range(4)]
    psi=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)); c,s=np.cos(psi),np.sin(psi)
    return np.stack([c*v[...,0]+s*v[...,1], -s*v[...,0]+c*v[...,1]],-1)
base="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/"
for p in sorted(glob.glob(base+"*/*/curriculum__omni/seed*/*/run_meta.json")):
    meta=json.load(open(p)); lab=meta["hold_labels"]
    if "lat_r" not in lab or lab.index("lat_r")!=lab.index("lat_l")+1: continue
    d=Path(p).parent; print("==",str(d).split("omni_phases/")[1], "speeds",meta.get("hold_speeds"))
    res={}
    for f in sorted((d/"raw").glob("episode_*.npz")):
        z=np.load(f); seg=z["segment_id"]; done=z["done"].astype(bool); S=int(np.argmax(done)) if done.any() else len(seg)-1
        u=u_of(z,meta); rp=z["root_pos_w"][:,:2].astype(float); fc=z["foot_pos_w"][:,:,:2].astype(float).mean(1)
        q=z["root_quat_w"].astype(float); fb=rot_inv(q[:,None,:], z["foot_pos_w"][:,:,:2].astype(float)-rp[:,None,:])
        vb=z["root_lin_vel_b"][:,:2].astype(float)
        for name in ["lat_l","lat_r"]:
            idx=np.where((seg==lab.index(name))&(np.arange(len(seg))<S))[0]
            if len(idx)<300: continue
            ydir=1 if name=="lat_l" else -1
            lead=(~LEFT) if name=="lat_l" else LEFT
            uw=rp[idx[-1]]-rp[idx[0]]; uw/=np.linalg.norm(uw)
            for w in range(min(5,len(idx)//100)):
                a,b=idx[w*100],idx[(w+1)*100-1]; T=(b-a)*0.02; ii=idx[w*100:(w+1)*100]
                r=res.setdefault((name,w),{})
                r.setdefault("body",[]).append((rp[b]-rp[a])@uw/T); r.setdefault("feet",[]).append((fc[b]-fc[a])@uw/T)
                r.setdefault("vy",[]).append(vb[ii,1].mean()*ydir)
                r.setdefault("lead_u",[]).append(np.median(u[ii][:,lead])); r.setdefault("trail_u",[]).append(np.median(u[ii][:,~lead]))
                r.setdefault("lean",[]).append(np.median(-fb[ii].mean(1)[:,1]*ydir))
            r=res.setdefault((name,"onset"),{}); ii=idx[:10]
            r.setdefault("lead_u",[]).append(np.median(u[ii][:,lead])); r.setdefault("trail_u",[]).append(np.median(u[ii][:,~lead])); r.setdefault("lean",[]).append(np.median(-fb[ii].mean(1)[:,1]*ydir))
    for k in sorted(res, key=lambda k:(k[0], -1 if k[1]=="onset" else k[1])):
        r=res[k]; n=len(r["lead_u"])
        s=" ".join(f"{kk} {np.median(v):+.3f}" for kk,v in r.items())
        print(f"  {k[0]} w{k[1]!s:6s} n={n:3d}  {s}")
