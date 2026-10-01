import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def load(run):
    f=glob.glob(run+'/events.out.tfevents.*')[0]
    ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
    return {t:[(e.step,e.value) for e in ea.Scalars(t)] for t in ea.Tags()['scalars']}
a,b=load(sys.argv[1]),load(sys.argv[2])
tags=sorted(set(a)&set(b)); same=[];diff=[]
for t in tags:
    (same if a[t]==b[t] else diff).append(t)
print(f"tags common {len(tags)} identical {len(same)} differ {len(diff)}")
print("DIFFER:", [t for t in diff if not t.startswith('Perf')][:30])
for t in ['Train/mean_reward','Loss/hist_latent','Loss/priv_reg','Loss/value_function']:
    if t in a and t in b:
        xa=dict(a[t]); xb=dict(b[t]); first=None
        for s in sorted(xa):
            if s in xb and xa[s]!=xb[s]: first=(s,xa[s],xb[s]); break
        last=max(s for s in xa if s in xb)
        print(t,'first differing step:',first,'| last',last,xa[last],xb[last])
        for s in [0,10,19,20,25,50,100,500]:
            if s in xa and s in xb: print('   step',s,xa[s],xb[s])
