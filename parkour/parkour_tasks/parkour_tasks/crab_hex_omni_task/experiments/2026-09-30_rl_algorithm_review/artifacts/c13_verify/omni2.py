import numpy as np,sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
B='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
for r,p in [('2026-09-16_23-11-00',999),('2026-09-16_23-44-00',1998),('2026-09-17_00-13-40',2248),('2026-09-17_00-13-40',6997)]:
    ea=EventAccumulator(B+r,size_guidance={'scalars':0}); ea.Reload()
    er=[t for t in ea.Tags()['scalars'] if t.startswith('Episode_Reward/')]
    d={}
    for t in er:
        ss=ea.Scalars(t); a=np.array([e.value for e in ss]); stp=np.array([e.step for e in ss]); m=(stp<=p)&(stp>p-50); d[t.split('/')[-1]]=a[m].mean()
    print(r,p,'sum*1',round(sum(d.values()),4),'x40',round(40*sum(d.values()),3), {k:round(v*40,3) for k,v in d.items() if abs(v)>1e-5})
