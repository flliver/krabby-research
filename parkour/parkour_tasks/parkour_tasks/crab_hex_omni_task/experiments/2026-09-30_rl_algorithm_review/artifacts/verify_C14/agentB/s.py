import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
for d in sorted(glob.glob(base+'2026-09-2*')):
    fs=glob.glob(d+'/events*')
    if not fs: continue
    ea=EventAccumulator(d,size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']
    def g(t): return [e.value for e in ea.Scalars(t)] if t in tags else []
    lr=g('Loss/learning_rate'); sd=g('Policy/mean_noise_std'); pr=g('Loss/priv_reg_coef')
    st=[e.step for e in ea.Scalars('Loss/learning_rate')] if lr else [0]
    if not lr: continue
    fl=sum(1 for v in lr if v<=1.01e-5)/len(lr)
    print(d.split('/')[-1], 'step0',st[0],'n',len(lr),'floor%.3f'%fl,'lrmax%.2e'%max(lr),'sd %.2f->%.2f'%(sd[0],sd[-1]) if sd else '', 'priv %.4f->%.4f'%(pr[0],pr[-1]) if pr else '')
