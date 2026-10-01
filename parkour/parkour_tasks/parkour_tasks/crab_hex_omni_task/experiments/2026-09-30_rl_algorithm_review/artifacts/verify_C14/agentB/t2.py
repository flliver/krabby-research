import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/'
for line in open('lowlr.txt'):
    lrcfg,d=line.split()
    ea=EventAccumulator(base+d,size_guidance={'scalars':0}); ea.Reload(); tags=ea.Tags()['scalars']
    if 'Loss/learning_rate' not in tags: print(d,'no lr'); continue
    lr=[e.value for e in ea.Scalars('Loss/learning_rate')]; sd=[e.value for e in ea.Scalars('Policy/mean_noise_std')] if 'Policy/mean_noise_std' in tags else [0]
    print(lrcfg,d,len(lr),'floor%.3f'%(sum(v<=1.01e-5 for v in lr)/len(lr)),'max%.2e'%max(lr),'sd %.2f->%.2f min %.2f'%(sd[0],sd[-1],min(sd)))
