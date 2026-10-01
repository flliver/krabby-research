import numpy as np,sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    ea=EventAccumulator(d,size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']
    for t in ['Train/mean_reward','Episode_Reward/reward_tripod_schedule','Train/mean_episode_length']:
        if t not in tags: continue
        s=ea.Scalars(t); st=np.array([e.step for e in s]); v=np.array([e.value for e in s])
        out=[]
        for p in [999,1999,2999,4999,9999,19999]:
            i=np.where(st==p)[0]
            if len(i): out.append(f"{p}:{v[i[0]]:.3f}(w51 {v[max(0,i[0]-50):i[0]+1].mean():.3f})")
        print(d.split('experiments/')[-1],t,'n=',len(v),'last',st[-1],f'{v[-1]:.3f}',' '.join(out), 'sum=%.4g'%v.sum() if 'tripod' in t else '')
