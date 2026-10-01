import glob,sys,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
L='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
for r in sys.argv[1:]:
    fs=sorted(glob.glob(L+r+'/events*'))
    print('==',r,len(fs))
    ea=EventAccumulator(L+r,size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']
    for t in ['Loss/learning_rate','Policy/mean_noise_std','Loss/entropy','Train/mean_reward','Train/mean_episode_length','Loss/surrogate','Loss/value_function']:
        if t not in tags: print(t,'MISSING'); continue
        s=ea.Scalars(t); st=np.array([x.step for x in s]); v=np.array([x.value for x in s])
        extra=''
        if t=='Loss/learning_rate':
            off=(v>1.0001e-5).mean(); extra=f' offfloor={off:.4f} n_off={(v>1.0001e-5).sum()} max={v.max():.3g} median={np.median(v):.3g} first5={v[:5]}'
        print(t,'n',len(v),'steps',st[0],st[-1],'first',v[:3],'last',v[-3:],'mean_last200',v[-200:].mean(),extra)
