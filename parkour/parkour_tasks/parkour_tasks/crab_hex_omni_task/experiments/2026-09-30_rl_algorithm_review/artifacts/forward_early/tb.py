import sys, glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/'
tags=['Loss/learning_rate','Policy/mean_noise_std','Loss/value_function','Loss/entropy','Loss/symmetry','Loss/priv_reg_coef']
for run in sys.argv[1:]:
    d=base+run
    ea=EventAccumulator(d,size_guidance={'scalars':0}); ea.Reload()
    avail=ea.Tags()['scalars']
    print('==',run,'ntags',len(avail))
    for t in tags:
        if t not in avail: print('  missing',t); continue
        ev=ea.Scalars(t); vals=[e.value for e in ev]; steps=[e.step for e in ev]
        n=len(vals)
        idx=[0,1,2,5,10,50,n//4,n//2,n-1]
        s=' '.join(f'{steps[i]}:{vals[i]:.3g}' for i in idx if i<n)
        frac_floor = sum(1 for v in vals if v<=1.01e-5)/n if 'learning_rate' in t else None
        mx=max(vals); mn=min(vals)
        print(f'  {t}: n={n} min={mn:.3g} max={mx:.3g} '+(f'frac_at_floor={frac_floor:.3f} ' if frac_floor is not None else '')+s)
