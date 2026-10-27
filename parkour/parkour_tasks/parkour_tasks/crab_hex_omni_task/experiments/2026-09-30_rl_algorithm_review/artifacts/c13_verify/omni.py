import numpy as np,sys,glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
B='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
for r in sys.argv[1:]:
    ea=EventAccumulator(B+r,size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']
    tr=[t for t in tags if 'track_ratio' in t]
    er=[t for t in tags if t.startswith('Episode_Reward/')]
    s=ea.Scalars('Train/mean_reward'); st=np.array([e.step for e in s]); v=np.array([e.value for e in s])
    print(r,'steps',st[0],st[-1],'n',len(st),'tr tags',tr)
    def win(tag,endi):
        ss=ea.Scalars(tag); a=np.array([e.value for e in ss]); stp=np.array([e.step for e in ss])
        m=(stp<=endi)&(stp>endi-50); return a[m].mean() if m.any() else float('nan')
    pts=sorted(set([st[0]+49,st[0]+300,st[0]+1000,2248,2998,3998,4998,5998,st[-1]]))
    for p in pts:
        if p<st[0] or p>st[-1]: continue
        tot=sum(win(t,p) for t in er)/40.0
        print(f"  it{p}: mean_reward {win('Train/mean_reward',p):.2f}  ", ' '.join(f"{t.split('/')[-1]} {win(t,p):.3f}" for t in tr), f" sumEpRew/40 {tot:.3f}")
