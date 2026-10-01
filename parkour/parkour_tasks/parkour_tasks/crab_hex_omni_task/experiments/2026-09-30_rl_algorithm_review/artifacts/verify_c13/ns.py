import numpy as np
from multiprocessing import Pool
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
exec(open('tb.py').read().split('def f')[0])
def f(k):
    ea=EventAccumulator(R+runs[k],size_guidance={'scalars':0}); ea.Reload()
    v=np.array([e.value for e in ea.Scalars('Policy/mean_noise_std')])
    return f"{k} noise_std @0={v[0]:.3f} @2999={v[2999]:.3f} last={v[-1]:.3f}"
with Pool(4) as p: print('\n'.join(p.map(f,list(runs))))
