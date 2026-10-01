import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
for r in ['2026-09-23_04-25-38_seed3_X03_c6ft_A15pB','2026-09-23_02-38-36_seed3_X01_c6ft_A15pB','2026-09-23_03-32-07_seed2_X02_c6ft_A15pB']:
    ea=EventAccumulator(base+r,size_guidance={'scalars':0}); ea.Reload()
    g=lambda t: np.array([e.value for e in ea.Scalars(t)])
    lr=g('Loss/learning_rate'); sd=g('Policy/mean_noise_std')
    vals,c=np.unique(np.round(lr/1e-5,3),return_counts=True)
    print(r,'meanLR',lr.mean(),'ratio to floor',lr.mean()/1e-5, dict(zip(vals,c)))
    print('  first 20 lr/1e-5', np.round(lr[:20]/1e-5,2))
    # windows
    for a,b in [(0,500),(500,1000),(1000,1500),(1500,2000)]:
        print('  win',a,b,'offfloor',(lr[a:b]>1.01e-5).mean().round(3),'std',sd[a:b].mean().round(3))
    idx=np.argmax(sd>=0.4999); print('  first iter std>=0.4999:', idx)
    for t in ['Loss/value_function','Loss/entropy','Train/mean_reward','Train/mean_episode_length']:
        v=g(t); print('  ',t,[round(float(v[a:b].mean()),3) for a,b in [(0,500),(500,1000),(1000,1500),(1500,2000)]])
