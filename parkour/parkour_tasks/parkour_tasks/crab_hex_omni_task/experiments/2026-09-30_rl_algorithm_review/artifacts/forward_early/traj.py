import numpy as np,sys
from multiprocessing import Pool
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R='/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/'
runs={'H(gait_tuned)':'2026-08-09_1526_gait_tuned/logs/rsl_rl/crab_hex_flat_walk/2026-08-09_15-27-14',
'v1_20k':'2026-08-10_0058_tripod_stability/fromscratch_tripod_reward_w0.15/logs/rsl_rl/crab_hex_flat_walk/2026-08-10_20-08-53',
'v2':'2026-08-10_0058_tripod_stability/fromscratch_tripod_v2_w0.15/logs/rsl_rl/crab_hex_flat_walk/2026-08-11_10-55-07',
'v3':'2026-08-10_0058_tripod_stability/fromscratch_tripod_v3_w0.15_short/logs/rsl_rl/crab_hex_flat_walk/2026-08-11_16-12-47',
'v3b':'2026-08-10_0058_tripod_stability/fromscratch_tripod_v3b_slide_short/logs/rsl_rl/crab_hex_flat_walk/2026-08-11_20-09-33',
'v4':'2026-08-10_0058_tripod_stability/fromscratch_tripod_v4_swap_short/logs/rsl_rl/crab_hex_flat_walk/2026-08-11_22-23-46',
'v5':'2026-08-10_0058_tripod_stability/fromscratch_tripod_v5_crossing_short/logs/rsl_rl/crab_hex_flat_walk/2026-08-12_00-44-45',
'baseline_0807':'2026-08-07_1441_baseline/logs/rsl_rl/crab_hex_flat_walk/2026-08-07_15-31-12',
'mrev_0807':'2026-08-07_2303_motor_reversal_on/logs/rsl_rl/crab_hex_flat_walk/2026-08-07_23-03-37',
'v2stride':'2026-08-08_0950_stride_length_on/logs/rsl_rl/crab_hex_flat_walk/2026-08-08_09-51-01',
'v3stride':'2026-08-08_1701_stride_length_v3/logs/rsl_rl/crab_hex_flat_walk/2026-08-08_17-01-57',
'v4stride':'2026-08-09_0106_stride_length_v4/logs/rsl_rl/crab_hex_flat_walk/2026-08-09_01-06-25'}
T=['Train/mean_reward','Metrics/base_velocity/error_vel_xy','Episode_Reward/reward_tripod_schedule','Policy/mean_noise_std','Loss/learning_rate','Episode_Termination/crab_failure','Train/mean_episode_length']
def f(k):
    ea=EventAccumulator(R+runs[k],size_guidance={'scalars':0}); ea.Reload(); out=[]
    for t in T:
        if t not in ea.Tags()['scalars']: continue
        v=np.array([e.value for e in ea.Scalars(t)])
        pts=[50,250,500,1000,2000,2999,5000,10000,19999]
        vals=[]
        for p in pts:
            if p<len(v): vals.append(f"{v[max(0,p-50):p+1].mean():.4g}")
        extra=''
        if 'tripod' in t: extra=f' sum={v.sum():.4g} max={v.max():.4g}'
        if 'learning_rate' in t: extra=f' mean_first1000={v[:1000].mean():.2e} mean_first100={v[:100].mean():.2e} floor%first1000={np.mean(v[:1000]<=1.0001e-5)*100:.0f}'
        out.append(f"{k:14s} {t:40s} "+' '.join(vals)+extra)
    return '\n'.join(out)
with Pool(4) as p:
    for s in p.map(f,list(runs)): print(s)
