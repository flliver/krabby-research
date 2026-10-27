import sys, glob, collections
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
for d in sys.argv[1:]:
    f=sorted(glob.glob(d+'/events*'))
    if not f: print(d,'NO EVENTS'); continue
    ea=EventAccumulator(f[0], size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']
    v=[x.value for x in ea.Scalars('Loss/learning_rate')]; st=[x.step for x in ea.Scalars('Loss/learning_rate')]
    s=[x.value for x in ea.Scalars('Policy/mean_noise_std')]
    c=collections.Counter(round(x,7) for x in v)
    n=len(v)
    top=sorted(c.items(), key=lambda kv:-kv[1])[:5]
    firstfloor=next((i for i,x in enumerate(v) if x<=1.0001e-5),-1)
    pr=[x.value for x in ea.Scalars('Loss/priv_reg_coef')] if 'Loss/priv_reg_coef' in tags else None
    print(d.split('/')[-3] if 'experiments' in d else d.split('/')[-1], 'n',n,'steps',st[0],st[-1],'floor%',round(100*sum(x<=1.0001e-5 for x in v)/n,1),'mean',f"{sum(v)/n:.2e}",'max',f"{max(v):.2e}",'first5',[f"{x:.2e}" for x in v[:5]],'firstfloor',firstfloor,'top',[(f"{k:.2e}",round(m/n,3)) for k,m in top],'| sig',round(s[0],3),round(min(s),3),round(s[-1],3),'| priv',None if pr is None else (pr[0],pr[-1]))
