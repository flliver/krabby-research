import sys,glob,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    for ff in sorted(glob.glob(d+'/events.out.tfevents.*')):
        ea=EventAccumulator(ff,size_guidance={'scalars':0}); ea.Reload()
        if 'Loss/learning_rate' not in ea.Tags()['scalars']: continue
        s=ea.Scalars('Loss/learning_rate'); st=np.array([x.step for x in s]); v=np.array([x.value for x in s])
        floor=np.isclose(v,1e-5,rtol=1e-3)
        d20=(st%20==0)
        print(ff.split('/')[-2], 'n',len(v),'steps',st.min(),st.max(),'median',np.median(v),
              'floor frac dagger-its',floor[d20].mean().round(3),'(n',d20.sum(),') floor frac other',floor[~d20].mean().round(3),
              'median other',np.median(v[~d20]), 'p10/p90 other',np.percentile(v[~d20],[10,90]))
