import numpy as np
from multiprocessing import Pool
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R='/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/'
runs={'H':'2026-08-09_1526_gait_tuned/logs/rsl_rl/crab_hex_flat_walk/2026-08-09_15-27-14',
'v5':'2026-08-10_0058_tripod_stability/fromscratch_tripod_v5_crossing_short/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_00-44-45',
'seed6':'2026-08-12_1550_seed_basin_search/seed6/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_21-27-30',
'seed7':'2026-08-12_1550_seed_basin_search/seed7/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_22-25-57',
'sym':'2026-08-13_0035_mirror_symmetry/fromscratch_sym_20k/logs/rsl_rl/crab_hex_flat_walk/2026-08-13_00-35-14'}
def f(k):
    ea=EventAccumulator(R+runs[k],size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']; out=[]
    for t in tags:
        ev=ea.Scalars(t); s=np.array([e.step for e in ev]); v=np.array([e.value for e in ev])
        if t=='Train/mean_reward' or 'Episode_Reward' in t or t=='Train/mean_episode_length':
            i=np.where(s==2999)[0]
            at=v[i[0]] if len(i) else float('nan')
            w=v[(s>=2900)&(s<=2999)].mean()
            out.append(f"{k:6s} {t:55s} n={len(v)} maxstep={s.max()} @2999={at:.4g} mean2900-2999={w:.4g} sum0-2999={v[s<=2999].sum():.4g} max={v[s<=2999].max():.4g}")
    return '\n'.join(out)
with Pool(4) as p:
    for s in p.map(f,list(runs)): print(s); print()
