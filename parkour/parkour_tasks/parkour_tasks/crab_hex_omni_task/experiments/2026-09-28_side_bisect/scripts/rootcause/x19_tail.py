import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
d="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-30_12-27-45_seed3_X19_s1_A15pB"
ea=EventAccumulator(d,size_guidance={"scalars":0}); ea.Reload()
s=np.array([(e.step,e.value) for e in ea.Scalars('Metrics/base_parkour/track_ratio_mean')])
for lo in range(1600,3000,200):
    m=(s[:,0]>=lo)&(s[:,0]<lo+200); print(lo,round(s[m,1].mean(),3))
print("final 50:",round(s[-50:,1].mean(),3))
tags=ea.Tags()["scalars"]
for t in tags:
    if "rsi" in t.lower(): 
        v=np.array([e.value for e in ea.Scalars(t)]); print(t, round(v[-300:].mean(),3))
