from symlib import *
import json
reg=json.load(open(CACHE/"registry.json"))
print(f"{'stamp':14s} {'hand':9s} {'n':>3s} {'feet':>5s} {'swp':>5s} {'lean':>5s} | rep  R2   oddpi  (h1,h2)     corr  odd0  dmin@tau  | halfnoise")
for name,group,path in reg["stamps"]:
    for lab in ("lat_l_022","lat_l_025","lat_r_022","lat_r_025"):
        d=load(name,lab)
        if d is None: continue
        if not np.isfinite(d["ph"]).all(): continue
        fs=np.median(d["feet_ss"]); sw=np.nanmedian(d["sweep"]); le=np.median(d["lean_abs"])
        for rep in ("act","jp","foot"):
            o,A=selfmirror(d[rep],d["ph"],rep)
            print(f"{name:14s} {lab:9s} {len(d['act']):3d} {fs:5.3f} {sw:5.3f} {le:5.2f} | {rep:4s} {o['lockR2']:.2f} {o['odd_pi']:.3f} ({o['odd_pi_h1']:.2f},{o['odd_pi_h2']:.2f}) {o['odd_pi_corr']:.3f} {o['odd_0']:.3f} {o['d_min']:.2f}@{o['tau_min_deg']:.0f} | {o['d_half']:.2f}")
