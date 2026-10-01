import sys,glob,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
for r in ['2026-09-23_04-25-38_seed3_X03_c6ft_A15pB','2026-09-23_02-38-36_seed3_X01_c6ft_A15pB','2026-09-23_03-32-07_seed2_X02_c6ft_A15pB','2026-09-22_11-58-07_seed3_X08_c6_A15pB']:
    ea=EventAccumulator(base+r,size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']
    lr=np.array([e.value for e in ea.Scalars('Loss/learning_rate')]); st=[e.step for e in ea.Scalars('Loss/learning_rate')]
    sd=np.array([e.value for e in ea.Scalars('Policy/mean_noise_std')])
    floor=np.isclose(lr,1e-5,rtol=1e-3)
    print(r, 'n',len(lr),'steps',st[0],st[-1],'onfloor%',floor.mean()*100,'offfloor%',(~floor).mean()*100,'lr med',np.median(lr),'max',lr.max(), 'min', lr.min())
    print('  lr>1e-5*1.01 frac', (lr>1.01e-5).mean(), 'lr>=1e-4 frac',(lr>=1e-4).mean())
    print('  std first/q/last', sd[:3], np.percentile(sd,[10,50,90]), sd[-3:])
    for t in ['Loss/entropy','Loss/surrogate','Train/mean_reward','Train/mean_episode_length']:
        if t in tags:
            v=np.array([e.value for e in ea.Scalars(t)]); print('  ',t, v[:3].round(4), 'last100 mean', v[-100:].mean().round(4))
    kl=[t for t in tags if 'kl' in t.lower()]; print('  kl tags',kl)
