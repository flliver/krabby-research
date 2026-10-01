import sys, glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    f=glob.glob(d+'/events.out.tfevents.*')[0]
    ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
    lr=np.array([e.value for e in ea.Scalars('Loss/learning_rate')]); st=np.array([e.step for e in ea.Scalars('Loss/learning_rate')])
    print(d.split('/')[-1], 'n',len(lr))
    print(' first 30:', ' '.join(f'{x:.2e}' for x in lr[:30]))
    fl=np.isclose(lr,1e-5)
    print(' frac at floor', fl.mean(), 'frac>1e-4',(lr>1e-4).mean(), 'median',np.median(lr))
    for a,b in [(0,100),(100,500),(500,1500),(1500,len(lr))]:
        s=lr[a:b]
        if len(s): print(f'  iters {a}-{b}: floor {np.isclose(s,1e-5).mean():.2f} >1e-4 {(s>1e-4).mean():.3f} mean {s.mean():.2e} median {np.median(s):.2e}')
    r=lr[1:]/lr[:-1]
    idx=np.where(r>1.6)[0]
    print(' jumps>1.6 count',len(idx),' first iters',st[idx[:15]+1], 'ratios',np.round(r[idx[:15]],1))
    # after a jump what's next value
    nxt=[lr[i+2] if i+2<len(lr) else np.nan for i in idx[:15]]
    print(' value after jump iter', ' '.join(f'{x:.1e}' for x in lr[idx[:15]+1]), '| next', ' '.join(f'{x:.1e}' for x in nxt))
    down=r[r<1/1.6]
    print(' big drops count',len(down),'min ratio', down.min() if len(down) else None)
