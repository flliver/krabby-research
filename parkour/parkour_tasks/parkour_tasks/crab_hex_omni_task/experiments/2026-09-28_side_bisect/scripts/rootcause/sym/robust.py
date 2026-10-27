"""Robustness of the rod-command odd/even split: bins 12/24/36, clock vs intrinsic phase, per-episode patterns."""
from symlib import *
import json
reg=json.load(open(CACHE/"registry.json"))
CL=json.load(open("sym_results.json"))
cls={(r["stamp"],r["label"]):r["cls"] for r in CL["rows"]}
out={}
print("stamp          lab        cls  | pooled odd/even RMS: clk nb12  nb24  nb36 | intr nb24 | per-episode(nb12) odd/even med | per-ep oddfrac")
for name,group,path in reg["stamps"]:
    for lab in ("lat_l_022","lat_l_025","lat_r_022","lat_r_025"):
        d=load(name,lab)
        if d is None: continue
        X=sub(d,"act_rod"); th,_,_=intrinsic_phase(d)
        has=np.isfinite(d["ph"]).all()
        res={}
        s=""
        for nb in (12,24,36):
            if has:
                A,_,_=pattern(X,d["ph"],nb=nb); O=(A-mir(A,"act_rod",nb//2))/2; E=A-O
                res[f"clk{nb}"]=(rms(O),rms(E)); s+=f"{rms(O):.2f}/{rms(E):.2f} "
            else: s+="  -  /  -  "
        A,_,_=pattern(X,th,nb=24); O=(A-mir(A,"act_rod",12))/2; E=A-O; res["int24"]=(rms(O),rms(E))
        s+=f"| {rms(O):.2f}/{rms(E):.2f} "
        ph=d["ph"] if has else th
        oe=[];ee=[];ff=[]
        for e in range(len(X)):
            A,_,_=pattern(X[e:e+1],ph[e:e+1],nb=12); O=(A-mir(A,"act_rod",6))/2; E=A-O
            oe.append(rms(O)); ee.append(rms(E)); ff.append(rms(O)**2/(rms(O)**2+rms(E)**2))
        res["perep"]=(float(np.median(oe)),float(np.median(ee)),float(np.median(ff)))
        s+=f"| {np.median(oe):.2f}/{np.median(ee):.2f} | {np.median(ff):.2f}"
        out[f"{name}|{lab}"]=res
        print(f"{name:14s} {lab} {cls[(name,lab)]:4s} | {s}")
json.dump(out,open("robust.json","w"),indent=1)
