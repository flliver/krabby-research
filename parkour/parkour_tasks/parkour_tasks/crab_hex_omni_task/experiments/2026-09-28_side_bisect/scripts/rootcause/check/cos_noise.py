import json, numpy as np, sys
sys.path.insert(0,'.'); 
exec(open('recompute.py').read().split("rows=[]")[0])   # reuse loaders/maps
rng=np.random.default_rng(0)
def load_rows(name,labs):
    d=Path(STAMPS[name]); meta=json.loads((d/"run_meta.json").read_text()); labels=meta["hold_labels"]
    files=[(d/"raw"/f"episode_{e['env_index']:02d}.npz",e["label"]) for e in meta["env_index_map"] if e["label"] in labs]
    X=[];PH=[]
    for p,lab in files:
        z=np.load(p); done=z["done"].astype(bool); Sd=int(np.argmax(done)) if done.any() else len(done)
        seg=z["segment_id"]; sid=labels.index(lab); idx=np.where((seg==sid)&(np.arange(len(seg))<Sd))[0]
        if len(idx)==0: continue
        hs=sc.hold_stats(z,sid,Sd); t0=idx[0]
        if hs is None or idx[-1]<t0+500: continue
        sl=slice(t0+200,t0+500); X.append(np.clip(z["actions"][sl,6:18],-1,1)); PH.append(z["clock_phase"][sl])
    return np.array(X),np.array(PH)
def oddpat(X,ph):
    A=pat(X,ph); return (A-mirror(A,NB//2))/2
def cos(a,b): return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))
ref={}
for nm in ["C","C_mo_s2"]:
    X,ph=load_rows(nm,["lat_l_022"]); ref[nm]=oddpat(X,ph)
print("cos(odd C, odd C_mo_s2) [two FAST heads, same hand] =", round(cos(ref["C"],ref["C_mo_s2"]),3))
for nm,lab in [("O","lat_l_022"),("LR4","lat_l_022"),("LR4","lat_r_022"),("C_lr","lat_l_022"),("C_lr","lat_r_022"),("C_lr_rsi33","lat_l_022"),("C_lr_rsi33","lat_r_022"),("canon_C","lat_r_022")]:
    X,ph=load_rows(nm,[lab]); O=oddpat(X,ph)
    if lab.endswith("r_022"):   # bring the right hand into the left hand's frame: M_pi image
        O=mirror(O,NB//2)*(-1)  # odd part of the mirrored pattern = -M(odd)... use the left-frame image
    # per-episode odd rms and phase-shuffled floor
    pe=[]; fl=[]
    for i in range(len(X)):
        pe.append(rms(oddpat(X[i:i+1],ph[i:i+1])))
        phs=ph[i:i+1].copy(); phs=phs+rng.uniform(0,2*np.pi)  # random per-episode phase offset: destroys cross-episode consistency only
        fl.append(rms(oddpat(X[i:i+1],rng.permutation(ph[i].reshape(-1)).reshape(1,-1))))  # within-episode shuffle: pure noise floor
    print(f"{nm:10s} {lab}  cos(odd, C's odd) {cos(O,ref['C']):+.2f}  cos(odd, C_mo_s2's odd) {cos(O,ref['C_mo_s2']):+.2f} | per-ep odd rms med {np.median(pe):.2f}  shuffled-phase floor med {np.median(fl):.2f}")
X,ph=load_rows("C",["lat_l_022"]); pe=[rms(oddpat(X[i:i+1],ph[i:i+1])) for i in range(len(X))]; fl=[rms(oddpat(X[i:i+1],rng.permutation(ph[i]).reshape(1,-1))) for i in range(len(X))]
print(f"C          lat_l_022  per-ep odd rms med {np.median(pe):.2f}  shuffled floor {np.median(fl):.2f}")
