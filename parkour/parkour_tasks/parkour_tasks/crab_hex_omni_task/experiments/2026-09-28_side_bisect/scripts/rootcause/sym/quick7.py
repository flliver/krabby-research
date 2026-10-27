from symlib import *
np.set_printoptions(precision=2, suppress=True, linewidth=200)
for name in ["R","O","C","LR4","W","side_s3_4999"]:
    d=load(name,"lat_l_022"); th,f0,_=intrinsic_phase(d)
    print("==",name,"f0",f0, "camv mean per leg", d["camv"].mean(axis=(0,1)), "camv std", d["camv"].std(axis=1).mean(0))
    print("   act static mean", d["act"].mean(axis=(0,1)))
    for rep in ("act","jp","foot"):
        o,A=selfmirror(d[rep].astype(float),th,rep)
        po=pair_odd(A,rep)
        print(f"  {rep} R2 {o['lockR2']:.2f} odd_pi {o['odd_pi']:.2f} blocks:", {k:round(v,2) for k,v in po.items()}, "leg share", leg_amp(A))
