import sys, glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    f=glob.glob(d+'/events.out.tfevents.*')[0]
    ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
    lr=np.array([e.value for e in ea.Scalars('Loss/learning_rate')])
    r=lr[1:]/lr[:-1]
    for a,b in [(0,100),(100,500),(500,1500),(1500,3000)]:
        s=r[a:b]; l=lr[a+1:b+1]
        print(d[-30:],a,b,'max jump',s.max().round(1),'n jumps>1.6',(s>1.6).sum(),'max lr',l.max(),'p90 lr',np.percentile(l,90))
    tags=ea.Tags()['scalars']; print([t for t in tags if 'kl' in t.lower()])
