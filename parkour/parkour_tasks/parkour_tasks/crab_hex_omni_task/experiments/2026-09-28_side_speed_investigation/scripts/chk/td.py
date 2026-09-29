import sys, json, numpy as np, glob
sys.path.insert(0,"/tmp/claude-1000/sidespeed")
from kin import L, LEFT, jidx, STAMPS
from pathlib import Path
L1, L2 = L.FEMUR_LEN_M, 0.8255
FORM=Path("/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-16_1805_omni_formation")
def u_of(z,meta):
    jp=z["joint_pos"].astype(float); ih=jidx(meta,"Hip_Femur_RevoluteJoint"); ik=jidx(meta,"Femur_Tibia_RevoluteJoint")
    ks=np.where(LEFT,1.0,-1.0); th=jp[:,ih]; tk=jp[:,ik]*ks; phi=th+tk
    return L1*np.cos(th)-L2*np.sin(phi)
def stamp(tag):
    if tag in STAMPS: return STAMPS[tag]
    return Path(glob.glob(str(FORM/f"seed3_{tag}_A15pB/slow__omni/seed001/*"))[0])
def run(tag,label,skip=200):
    d=stamp(tag); meta=json.loads((d/"run_meta.json").read_text()); lab=meta["hold_labels"]
    if meta.get("suite"): files=[(d/"raw"/f"episode_{e['env_index']:02d}.npz",e['label']) for e in meta["env_index_map"] if e["label"]==label]
    else: files=[(p,label) for p in sorted((d/"raw").glob("episode_*.npz"))]
    acc={}
    def add(k,v): acc.setdefault(k,[]).append(v)
    # tracking income accounting
    for p,lb in files:
        z=np.load(p); seg=z["segment_id"]; done=z["done"].astype(bool); S=int(np.argmax(done)) if done.any() else len(seg)-1
        idx=np.where((seg==lab.index(lb))&(np.arange(len(seg))<S))[0]
        if len(idx)<skip+200: continue
        u=u_of(z,meta); F=z["foot_force_norm"].astype(float)>50
        lead=(~LEFT) if lb=="lat_l" else LEFT
        vb=z["root_lin_vel_b"][:,:2].astype(float); cmd=z["cmd_applied"][:,:2].astype(float)
        err=np.linalg.norm(cmd-vb,axis=1); inc=1.25*np.exp(-err**2/0.04)-2.0*err+0.6*np.clip((vb*cmd).sum(1)/np.maximum(np.linalg.norm(cmd,axis=1),1e-9),0,1.05)
        add("inc_w0",inc[idx[:100]].mean()); add("inc_w1",inc[idx[100:200]].mean()); add("inc_steady",inc[idx[200:]].mean())
        add("onset_lead_u",np.median(u[idx[:10]][:,lead])); add("onset_trail_u",np.median(u[idx[:10]][:,~lead]))
        ii=idx[skip:]
        for f in range(6):
            row="lead" if lead[f] else "trail"
            Lz=F[ii,f]; dd=np.diff(np.r_[0,Lz.astype(int),0]); st=np.where(dd==1)[0]; en=np.where(dd==-1)[0]
            td=[u[ii[a],f] for a,b in zip(st,en) if b-a>=5]; lo=[u[ii[b-1],f] for a,b in zip(st,en) if b-a>=5]
            sw=[]
            for k in range(len(st)-1):
                a=en[k]; b=st[k+1]
                if b>a: sw.append(np.max(u[ii[a:b],f]) if row=="lead" else np.min(u[ii[a:b],f]))
            if td: add(row+".u_touchdown",np.median(td)); add(row+".u_liftoff",np.median(lo))
            if sw: add(row+".u_swing_extreme",np.median(sw))
            add(row+".u_p5",np.percentile(u[ii,f],5)); add(row+".u_p95",np.percentile(u[ii,f],95))
    print(f"== {tag} {label} n={len(acc['inc_w0'])}")
    for k in sorted(acc): print(f"  {k:22s} {np.nanmedian(acc[k]):+.3f}")
run("W","lat_l"); run("t3","lat_l"); run("1b_002","lat_l"); run("1c_004","lat_l")
