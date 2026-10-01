import numpy as np, glob, os
from multiprocessing import Pool
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R='/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/'
runs={'H':'2026-08-09_1526_gait_tuned/logs/rsl_rl/crab_hex_flat_walk/2026-08-09_15-27-14',
'v5':'2026-08-10_0058_tripod_stability/fromscratch_tripod_v5_crossing_short/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_00-44-45',
'sym':'2026-08-13_0035_mirror_symmetry/fromscratch_sym_20k/logs/rsl_rl/crab_hex_flat_walk/2026-08-13_00-35-14'}
for s in range(2,9):
    g=glob.glob(R+f'2026-08-12_1550_seed_basin_search/seed{s}/logs/rsl_rl/crab_hex_flat_walk/*')
    if g: runs[f'seed{s}']=g[0].replace(R,'')
for g in sorted(glob.glob(R+'2026-08-10_0058_tripod_stability/*/logs/rsl_rl/crab_hex_flat_walk/*')):
    runs['ts:'+g.split('tripod_stability/')[1].split('/')[0]+'/'+os.path.basename(g)]=g.replace(R,'')
def f(k):
    ea=EventAccumulator(R+runs[k],size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']; out=[f'== {k} {runs[k]}']
    for t in ['Train/mean_reward','Train/mean_episode_length','Policy/mean_noise_std','Episode_Reward/reward_tripod_schedule']:
        if t not in tags: out.append(f'  {t}: ABSENT'); continue
        ev=ea.Scalars(t); s=np.array([e.step for e in ev]); v=np.array([e.value for e in ev])
        line=f'  {t:45s} n={len(v)} last={s[-1]}'
        for p in [2999,4999,9999,19999]:
            i=np.where(s==p)[0]
            if len(i): line+=f' @{p}={v[i[0]]:.3f}(w51={v[max(0,i[0]-50):i[0]+1].mean():.3f})'
        if s[-1]>=19000: line+=f' last500={v[-500:].mean():.3f}'
        if 'tripod' in t: line+=f' sum={v.sum():.5f} max={v.max():.3g}'
        out.append(line)
    return '\n'.join(out)
with Pool(4) as p:
    for s in p.map(f,list(runs)): print(s)
