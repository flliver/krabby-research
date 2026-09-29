import sys, json, numpy as np
sys.path.insert(0,"/tmp/claude-1000/sidespeed")
from kin import LEFT, heading, STAMPS
def run(tag, labels_want, segs=None, skip=200):
    d=STAMPS[tag]; meta=json.loads((d/"run_meta.json").read_text()); lab=meta["hold_labels"]
    if meta.get("suite"):
        files=[(d/"raw"/f"episode_{e['env_index']:02d}.npz",[lab.index(e['label'])],e['label']) for e in meta["env_index_map"] if e["label"] in labels_want]
    else:
        files=[(p,[lab.index(s) for s in segs],"side") for p in sorted((d/"raw").glob("episode_*.npz"))]
    acc={}
    def add(k,v): acc.setdefault(k,[]).append(v)
    for p,ids,label in files:
        z=np.load(p); seg=z["segment_id"]; done=z["done"].astype(bool); S=int(np.argmax(done)) if done.any() else len(seg)-1
        idx=np.where(np.isin(seg,ids)&(np.arange(len(seg))<S))[0]
        if len(idx)<skip+200: continue
        idx=idx[skip:]  # steady part
        rp=z["root_pos_w"][:,:2].astype(float); fp=z["foot_pos_w"][:,:,:2].astype(float); F=z["foot_force_norm"].astype(float)
        u=rp[idx[-1]]-rp[idx[0]]; u/=np.linalg.norm(u); T=(idx[-1]-idx[0])*0.02
        vbody=(rp[idx[-1]]-rp[idx[0]])@u/T
        psi=heading(z["root_quat_w"].astype(float))
        # leading row = legs on the +travel side of the body. body-frame y of travel:
        c,s=np.cos(psi[idx]),np.sin(psi[idx]); ty=(-s*u[0]+c*u[1]).mean()  # travel dir y-component in body frame
        # LEFT legs at -y in this USD
        lead = (~LEFT) if ty>0 else LEFT
        loaded=F>50
        for f in range(6):
            row="lead" if lead[f] else "trail"
            L=loaded[idx,f]
            # foot world velocity along travel while loaded (slip), central diff
            fv=np.gradient(fp[idx,f,:]@u,0.02)
            add(row+".foot_v_loaded_along", fv[L].mean() if L.sum()>20 else np.nan)
            add(row+".load_frac", L.mean()); add(row+".mean_force_loaded", F[idx,f][L].mean() if L.sum()>20 else np.nan)
            # per-stance: foot displacement (slip) and body displacement
            dd=np.diff(np.r_[0,L.astype(int),0]); st=np.where(dd==1)[0]; en=np.where(dd==-1)[0]
            sl=[];bd=[];dur=[]
            for a,b in zip(st,en):
                if b-a<5: continue
                a2,b2=idx[a],idx[b-1]; sl.append((fp[b2,f]-fp[a2,f])@u); bd.append((rp[b2]-rp[a2])@u); dur.append((b2-a2)*0.02)
            if sl: add(row+".slip_per_stance",np.median(sl)); add(row+".body_adv_per_stance",np.median(bd)); add(row+".stance_dur",np.median(dur))
        add("vbody",vbody)
        add("load_share_lead", F[idx][:,lead].sum()/F[idx].sum())
    print(f"== {tag} {labels_want or segs} (steady part, skip {skip*0.02:.0f}s)  n={len(acc['vbody'])}")
    for k in sorted(acc): print(f"  {k:28s} {np.nanmedian(acc[k]):+.3f}")
run("W",["lat_l","lat_r"]); run("t3",["lat_l","lat_r"]); run("SH",None,["creep"],skip=250); run("SH",None,["low"],skip=100); run("W",["fwd"])
