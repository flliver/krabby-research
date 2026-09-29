# Split window speed into feet-centroid advance vs lean build-up (body over feet), per 2-s window.
import sys, json, numpy as np
sys.path.insert(0,"/tmp/claude-1000/sidespeed")
from kin import STAMPS
WIN=100; DT=0.02
def run(tag,labels_want,segs=None):
    d=STAMPS[tag]; meta=json.loads((d/"run_meta.json").read_text()); labels=meta["hold_labels"]
    if meta.get("suite"):
        files=[(d/"raw"/f"episode_{e['env_index']:02d}.npz",[labels.index(e['label'])]) for e in meta["env_index_map"] if e["label"] in labels_want]
    else:
        files=[(p,[labels.index(s) for s in segs]) for p in sorted((d/"raw").glob("episode_*.npz"))]
    res={}
    for p,ids in files:
        z=np.load(p); seg=z["segment_id"]; done=z["done"].astype(bool); S=int(np.argmax(done)) if done.any() else len(seg)-1
        idx=np.where(np.isin(seg,ids)&(np.arange(len(seg))<S))[0]
        if len(idx)<300: continue
        rp=z["root_pos_w"][:,:2].astype(float); fc=z["foot_pos_w"][:,:,:2].astype(float).mean(1)
        u=rp[idx[-1]]-rp[idx[0]]; u=u/np.linalg.norm(u)
        for w in range(len(idx)//WIN):
            a,b=idx[w*WIN],idx[(w+1)*WIN-1]; T=(b-a)*DT
            r=res.setdefault(w,{})
            r.setdefault("body",[]).append((rp[b]-rp[a])@u/T)
            r.setdefault("feet",[]).append((fc[b]-fc[a])@u/T)
    print(f"== {tag} {labels_want or segs}: 2-s windows from hold start, speed along travel (m/s)")
    for w in sorted(res):
        r=res[w]
        if len(r["body"])<5: continue
        print(f" w{w}  body {np.median(r['body']):.3f}  feet-centroid {np.median(r['feet']):.3f}  lean build {np.median(np.array(r['body'])-np.array(r['feet'])):+.3f}")
run("W",["lat_l","lat_r"]); run("t3",["lat_l","lat_r"]); run("W",["fwd"]); run("SH",None,["creep"])
