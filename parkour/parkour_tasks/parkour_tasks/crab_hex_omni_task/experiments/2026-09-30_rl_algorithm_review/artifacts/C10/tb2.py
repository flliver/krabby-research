import sys,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
L='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
for r in sys.argv[1:]:
    ea=EventAccumulator(L+r,size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']
    if r==sys.argv[1]: print([t for t in tags if 'Loss' in t or 'Policy' in t or 'kl' in t.lower()])
    v=np.array([x.value for x in ea.Scalars('Loss/learning_rate')])
    print(r, 'meanLR %.3g'%v.mean(), 'first500 off %.3f'%(v[:500]>1.0001e-5).mean(), 'last500 off %.3f'%(v[-500:]>1.0001e-5).mean())
    for t in ['Train/mean_reward','Train/mean_episode_length']:
        w=np.array([x.value for x in ea.Scalars(t)])
        print('  ',t,[round(w[i:i+250].mean(),1) for i in range(0,2000,250)])
