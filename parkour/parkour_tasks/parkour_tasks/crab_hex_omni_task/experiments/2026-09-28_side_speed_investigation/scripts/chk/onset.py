import sys, json, numpy as np, glob
sys.path.insert(0,"/tmp/claude-1000/sidespeed")
from kin import L, LEGS, LEFT, heading, jidx, STAMPS
L1, L2 = L.FEMUR_LEN_M, 0.8255
def u_of(z,meta):
    jp=z["joint_pos"].astype(float); ih=jidx(meta,"Hip_Femur_RevoluteJoint"); ik=jidx(meta,"Femur_Tibia_RevoluteJoint")
    ks=np.where(LEFT,1.0,-1.0); th=jp[:,ih]; tk=jp[:,ik]*ks; phi=th+tk
    return L1*np.cos(th)-L2*np.sin(phi)
# 1) SH low (0.35) time course, 2-s windows
d=STAMPS["SH"]; meta=json.loads((d/"run_meta.json").read_text()); lab=meta["hold_labels"]
res={}; stand_u=[]; creep0=[]
for p in sorted((d/"raw").glob("episode_*.npz")):
    z=np.load(p); seg=z["segment_id"]; done=z["done"].astype(bool); S=int(np.argmax(done)) if done.any() else len(seg)-1
    u=u_of(z,meta)
    ist=np.where((seg==lab.index("stand"))&(np.arange(len(seg))<S))[0]
    if len(ist)>200: stand_u.append(np.median(u[ist[-100:]],0))
    for name in ["creep","low"]:
        idx=np.where((seg==lab.index(name))&(np.arange(len(seg))<S))[0]
        if len(idx)<300: continue
        vb=z["root_lin_vel_b"][:,1].astype(float)
        for w in range(len(idx)//100):
            ii=idx[w*100:(w+1)*100]; res.setdefault((name,w),[]).append(vb[ii].mean())
for k in sorted(res): print("SH",k,"n",len(res[k]),"vy med %.3f p25 %.3f p75 %.3f"%tuple(np.percentile(res[k],[50,25,75])))
su=np.array(stand_u); print("SH stand u (last 2 s of stand) per leg med", np.median(su,0).round(3), "L-row", np.median(su[:,LEFT]).round(3), "R-row", np.median(su[:,~LEFT]).round(3))
# 2) W stand posture and posture at lat hold onset
for tag in ["W","t3"]:
    d=STAMPS[tag]; meta=json.loads((d/"run_meta.json").read_text()); lab=meta["hold_labels"]
    st=[]; on={}; fw=[]
    for e in meta["env_index_map"]:
        z=np.load(d/"raw"/f"episode_{e['env_index']:02d}.npz"); seg=z["segment_id"]; done=z["done"].astype(bool); S=int(np.argmax(done)) if done.any() else len(seg)-1
        u=u_of(z,meta)
        if e["label"]=="stand":
            idx=np.where((seg==lab.index("stand"))&(np.arange(len(seg))<S))[0]
            if len(idx)>200: st.append(np.median(u[idx[-100:]],0))
        if e["label"] in ("lat_l","lat_r"):
            idx=np.where((seg==lab.index(e["label"]))&(np.arange(len(seg))<S))[0]
            if len(idx)<300: continue
            lead = LEFT if e["label"]=="lat_l" else ~LEFT
            for w,(a,b) in enumerate([(0,25),(25,50),(50,100),(100,200),(200,300),(300,500)]):
                ii=idx[a:b]; on.setdefault(w,[]).append((np.median(u[ii][:,lead]),np.median(u[ii][:,~lead])))
            # settle segment posture just before hold
            pre=np.where(seg==lab.index("settle"))[0]
            if len(pre)>20: on.setdefault("pre",[]).append((np.median(u[pre[-20:]][:,lead]),np.median(u[pre[-20:]][:,~lead])))
        if e["label"]=="fwd":
            idx=np.where((seg==lab.index("fwd"))&(np.arange(len(seg))<S))[0]
            if len(idx)>300: fw.append(np.median(u[idx[100:]],0))
    su=np.array(st); print(tag,"stand u per leg", np.median(su,0).round(3), "n",len(st))
    print(tag,"fwd u per leg", np.median(np.array(fw),0).round(3))
    for w in ["pre",0,1,2,3,4,5]:
        a=np.array(on[w]); print(tag,"lat onset window",w,"lead u %.3f trail u %.3f"%(np.median(a[:,0]),np.median(a[:,1])),"n",len(a))
