import glob, numpy as np, pickle
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
d=R+"2026-09-30_12-27-45_seed3_X19_s1_A15pB"
ea=EventAccumulator(d,size_guidance={"scalars":0}); ea.Reload()
tags=ea.Tags()["scalars"]
tr=[t for t in tags if "track_ratio" in t.lower()]
print(tr)
t=tr[0]
s=np.array([(e.step,e.value) for e in ea.Scalars(t)])
print("last step",s[-1,0])
for a,b in [(0,100),(100,200),(200,300),(300,400),(400,500),(500,600),(600,700),(700,800),(800,900),(900,1000),(1000,1100),(1100,1200),(1200,1300),(1300,1400),(1400,1500),(1500,1600)]:
    m=(s[:,0]>=a)&(s[:,0]<b)
    if m.any(): print(a,b,round(s[m,1].mean(),3))
# compare to C (X02) and C_lr (X14), C_lr_rsi (X16) from dyn pkl cache
for x in ["X02","X14","X16","X17","X04"]:
    try:
        o=pickle.load(open(f"dyn/tb/{x}.pkl","rb"))
        k=[k for k in o if "track_ratio" in k.lower()][0]
        a=np.array(o[k])
        row=[round(a[(a[:,0]>=lo)&(a[:,0]<lo+100),1].mean(),3) for lo in (500,800,900,1100)]
        print(x,k,row)
    except Exception as e: print(x,e)
