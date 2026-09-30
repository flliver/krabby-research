from symlib import *
import json
reg=json.load(open(CACHE/"registry.json"))
for name,group,path in reg["stamps"]:
    for lab in ("lat_l_022","lat_r_022"):
        d=load(name,lab)
        if d is None: continue
        ch=None
        if np.isfinite(d["ph"]).all():
            cf=np.median(np.diff(np.unwrap(d["ph"],axis=1),axis=1).mean(1))/(2*np.pi)  # cycles per step
            ch=0.5/cf
        s=[]
        for rep in ("act","jp","foot"):
            o=lagged(d[rep].astype(float),rep,clock_half=ch)
            s.append(f"{rep} T*{o['T_star_s']:.2f} rpk{o['r_peak']:.2f} oddT/2 {o['oddhalf_lag']:.2f} odd0 {o['odd0_lag']:.2f} oddclk {o.get('oddclock_lag',np.nan):.2f}")
        print(f"{name:13s} {lab} "+" | ".join(s))
