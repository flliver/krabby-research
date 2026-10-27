import numpy as np
from multiprocessing import Pool
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
L='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
runs={'1a':'2026-09-16_23-11-00','1b':'2026-09-16_23-44-00','1c':'2026-09-17_00-13-40'}
def f(k):
    ea=EventAccumulator(L+runs[k],size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']; out=[f'== {k} {runs[k]}']
    mr=ea.Scalars('Train/mean_reward'); s=np.array([e.step for e in mr]); v=np.array([e.value for e in mr])
    out.append(f'  mean_reward n={len(v)} steps {s[0]}..{s[-1]} first50={v[:50].mean():.2f} last50={v[-50:].mean():.2f} last200={v[-200:].mean():.2f}')
    for p in [999,2248,2498,2700,6997]:
        i=np.where(s==p)[0]
        if len(i): out.append(f'   @{p}: {v[i[0]]:.2f} (w50 {v[max(0,i[0]-49):i[0]+1].mean():.2f})')
    tr=[t for t in tags if 'track_ratio' in t]
    for t in tr:
        ev=ea.Scalars(t); ss=np.array([e.step for e in ev]); vv=np.array([e.value for e in ev])
        line=f'  {t} first50={vv[:50].mean():.3f} last50={vv[-50:].mean():.3f} last200={vv[-200:].mean():.3f}'
        for p in [2248,2498,2700]:
            i=np.where(ss==p)[0]
            if len(i): line+=f' @{p}={vv[i[0]]:.3f}(w50 {vv[max(0,i[0]-49):i[0]+1].mean():.3f})'
        out.append(line)
    ep=[t for t in tags if t.startswith('Episode_Reward/')]
    tot_last=0; tot_first=0; rows=[]
    for t in ep:
        ev=ea.Scalars(t); vv=np.array([e.value for e in ev]); ss=np.array([e.step for e in ev])
        tot_last+=vv[-50:].mean(); tot_first+=vv[:50].mean()
        if abs(vv[-50:].mean())>0.005: rows.append(f'{t.split("/")[1]}={vv[-50:].mean():.3f}')
    el=ea.Scalars('Train/mean_episode_length'); ev=np.array([e.value for e in el])
    out.append(f'  summed Episode_Reward last50={tot_last:.3f} first50={tot_first:.3f}; eplen last50={ev[-50:].mean():.0f} first50={ev[:50].mean():.0f}')
    out.append('  terms(last50): '+' '.join(rows))
    return '\n'.join(out)
with Pool(3) as p:
    for s in p.map(f,list(runs)): print(s)
