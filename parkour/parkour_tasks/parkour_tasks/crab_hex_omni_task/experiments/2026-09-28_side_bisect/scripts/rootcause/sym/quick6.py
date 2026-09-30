from symlib import *
import json
reg=json.load(open(CACHE/"registry.json"))
for name,group,path in reg["stamps"]:
    for lab in ("lat_l_022","lat_l_025","lat_r_022","lat_r_025"):
        d=load(name,lab)
        if d is None: continue
        th,f0,ev=intrinsic_phase(d)
        s=f"{name:13s} {lab} f0 {f0:.3f} "
        for rep in ("act","jp","foot"):
            oi,_=selfmirror(d[rep].astype(float),th,rep)
            s+=f"| {rep} R2i {oi['lockR2']:.2f} oddpi_i {oi['odd_pi']:.2f} odd0_i {oi['odd_0']:.2f} "
            if np.isfinite(d["ph"]).all():
                oc,_=selfmirror(d[rep].astype(float),d["ph"],rep)
                s+=f"[clk R2 {oc['lockR2']:.2f} pi {oc['odd_pi']:.2f}] "
        print(s)
