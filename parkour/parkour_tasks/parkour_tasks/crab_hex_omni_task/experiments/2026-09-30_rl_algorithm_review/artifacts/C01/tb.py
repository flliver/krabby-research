import sys,glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    f=sorted(glob.glob(d+'/events.out.tfevents.*'))
    for ff in f:
        ea=EventAccumulator(ff,size_guidance={'scalars':0}); ea.Reload()
        tags=ea.Tags()['scalars']
        print(ff, len(tags))
        for t in ['Loss/learning_rate','Policy/mean_noise_std','Loss/surrogate','Loss/entropy']:
            if t in tags:
                s=ea.Scalars(t); v=[x.value for x in s]; st=[x.step for x in s]
                import numpy as np; v=np.array(v)
                print(' ',t,'n',len(v),'first5',[(st[i],float(v[i])) for i in range(min(5,len(v)))],'min',v.min(),'max',v.max(),'frac==1e-5',float(np.mean(np.isclose(v,1e-5,rtol=1e-3))), 'frac>1e-5*1.01', float(np.mean(v>1.01e-5)))
