import sys,glob,numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    for ff in sorted(glob.glob(d+'/events.out.tfevents.*')):
        ea=EventAccumulator(ff,size_guidance={'scalars':0}); ea.Reload()
        s=ea.Scalars('Loss/learning_rate'); v=np.array([x.value for x in s])
        k=np.log(v/1e-5)/np.log(1.5)
        print(d,'k int frac',np.mean(np.abs(k-np.round(k))<1e-3),'k range',np.round(k).min(),np.round(k).max(),'max lr',v.max())
        n=ea.Scalars('Policy/mean_noise_std'); nv=np.array([x.value for x in n]); ns=np.array([x.step for x in n])
        i=np.argmax(nv>1.999); print('  noise_std first >=1.999 at step',ns[i] if (nv>1.999).any() else None,'frac >=1.999',np.mean(nv>1.999).round(3))
