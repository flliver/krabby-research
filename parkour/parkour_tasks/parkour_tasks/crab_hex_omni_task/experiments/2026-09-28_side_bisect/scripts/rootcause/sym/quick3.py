from symlib import *
import json
reg=json.load(open(CACHE/"registry.json"))
for name,group,path in reg["stamps"]:
    for lab in ("lat_l_022","lat_r_022"):
        d=load(name,lab)
        if d is None: continue
        th,f0,ev=intrinsic_phase(d)
        cl = d["ph"]
        s=""
        if np.isfinite(cl).all():
            dd=np.angle(np.exp(1j*(th-cl)))
            R=np.abs(np.mean(np.exp(1j*dd),1))   # per-episode phase-locking value
            glob_=np.abs(np.mean(np.exp(1j*dd)))
            cf=np.median(np.diff(np.unwrap(cl,axis=1),axis=1).mean(1))/DT/(2*np.pi)
            s=f"clock f {cf:.3f}  PLV(int-clock) per-ep {np.median(R):.2f} pooled {glob_:.2f}"
        rs=[]
        for rep in ("act","jp","foot"):
            A,r2,_=pattern(d[rep],th); rs.append(r2)
        print(f"{name:14s} {lab} f0_int {f0:.3f} PC1ev {ev:.2f} lockR2_int act/jp/foot {rs[0]:.2f}/{rs[1]:.2f}/{rs[2]:.2f} {s}")
