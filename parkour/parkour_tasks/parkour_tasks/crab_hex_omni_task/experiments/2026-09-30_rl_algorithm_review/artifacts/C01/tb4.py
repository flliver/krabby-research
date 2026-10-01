import sys,glob,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
d=sys.argv[1]; ff=glob.glob(d+'/events.out.tfevents.*')[0]
ea=EventAccumulator(ff,size_guidance={'scalars':0}); ea.Reload()
tags=ea.Tags()['scalars']; print([t for t in tags if t.startswith('Loss/')])
for t in ['Loss/estimator','Loss/priv_reg','Loss/hist_latent_loss','Loss/hist_latent']:
    if t in tags:
        v=np.array([x.value for x in ea.Scalars(t)]); print(t,'median',np.median(v),'last100 median',np.median(v[-100:]))
