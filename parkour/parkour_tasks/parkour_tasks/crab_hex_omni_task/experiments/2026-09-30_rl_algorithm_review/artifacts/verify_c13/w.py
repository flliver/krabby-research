import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
exec(open('tb.py').read().split('def f')[0])
for k in runs:
    ea=EventAccumulator(R+runs[k],size_guidance={'scalars':0}); ea.Reload()
    ev=ea.Scalars('Train/mean_reward'); v=np.array([e.value for e in ev]); s=np.array([e.step for e in ev])
    assert (s==np.arange(len(s))).all()
    print(k, 'w2949-2999=%.4g'%v[2949:3000].mean(), 'w2500-2999=%.4g'%v[2500:3000].mean(), 'final(last)=%.4g'%v[-1], 'last500=%.4g'%v[-500:].mean(), 'max all=%.4g'%v.max())
