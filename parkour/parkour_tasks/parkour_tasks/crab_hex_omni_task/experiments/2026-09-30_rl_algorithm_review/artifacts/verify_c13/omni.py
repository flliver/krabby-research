import numpy as np
from multiprocessing import Pool
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
runs={'1a':'2026-09-16_23-11-00','1b':'2026-09-16_23-44-00','1c':'2026-09-17_00-13-40'}
def f(k):
    ea=EventAccumulator(R+runs[k],size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']; out=[k+' tags with track: '+str([t for t in tags if 'track_ratio' in t or 'ratio' in t])]
    for t in ['Train/mean_reward']+[t for t in tags if 'track_ratio' in t]:
        ev=ea.Scalars(t); s=np.array([e.step for e in ev]); v=np.array([e.value for e in ev])
        out.append(f"{k} {t} steps {s.min()}-{s.max()} n={len(v)} first50={v[:50].mean():.4g} first={v[0]:.4g} last50={v[-50:].mean():.4g} last200={v[-200:].mean():.4g} last={v[-1]:.4g} max={v.max():.4g}")
        if k=='1c' and t=='Train/mean_reward':
            for a in range(s.min(), s.max()+1, 500):
                m=(s>=a)&(s<a+50); out.append(f"   {a}: {v[m].mean():.4g}")
    er=[t for t in tags if t.startswith('Episode_Reward/')]
    tot=sum(np.mean([e.value for e in ea.Scalars(t)][-50:]) for t in er)
    tot0=sum(np.mean([e.value for e in ea.Scalars(t)][:50]) for t in er)
    out.append(f"{k} sum Episode_Reward first50={tot0:.4g} last50={tot:.4g} (n terms {len(er)})")
    return '\n'.join(out)
with Pool(3) as p:
    for s in p.map(f,list(runs)): print(s); print()
