import sys, json, numpy as np
from pathlib import Path
sys.argv=[sys.argv[0]]
exec(open("probe_hand.py").read().split("res = {}")[0])
arms=["C","C_s2","C_mo_s2","C_mo_s1","C_mo","O","R","C_band","C_rsi","LR4","LRfa","C_lr","C_lr_10"]
for arm in arms:
    meta,W=windows(stamp(arm)); jn=meta["joint_names"]
    yi=[i for i,n in enumerate(jn) if "_Body_Hip_" in n]; pi_=[i for i,n in enumerate(jn) if "Hip_Femur" in n]; ki=[i for i,n in enumerate(jn) if "Femur_Tibia" in n]
    for lab in ("lat_l_022","lat_r_022"):
        if lab not in W: continue
        Q=np.stack([e["q"] for e in W[lab]]); A=np.clip(np.stack([e["a"] for e in W[lab]]),-1,1)
        qs=Q.std(axis=1).mean(0); cam=A[:,:,:6].mean(axis=(0,1))
        # leading row = left legs for lat_l (FL,ML,RL) ; legs order in joint names per group: FL,FR,ML,MR,RL,RR
        lead=[0,2,4] if lab.startswith("lat_l") else [1,3,5]; trail=[1,3,5] if lab.startswith("lat_l") else [0,2,4]
        pq=qs[pi_]; kq=qs[ki]; yq=qs[yi]
        print(f"{arm:8s} {lab:9s} yaw std {yq.mean():.3f} | pitch std lead {pq[lead].mean():.3f} trail {pq[trail].mean():.3f} | knee std lead {kq[lead].mean():.3f} trail {kq[trail].mean():.3f} | |cam mean| max {np.abs(cam).max():.2f}")
