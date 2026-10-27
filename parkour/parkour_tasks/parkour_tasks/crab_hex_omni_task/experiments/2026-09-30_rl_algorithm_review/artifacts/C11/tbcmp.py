import sys, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def load(d):
    f=glob.glob(d+'/events.out.tfevents.*')[0]
    ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload(); return ea
L='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/'
a,b=load(L+sys.argv[1]),load(L+sys.argv[2])
tags=sorted(set(a.Tags()['scalars'])&set(b.Tags()['scalars']))
print(len(tags),'common tags')
first={}
for t in tags:
    sa={e.step:e.value for e in a.Scalars(t)}; sb={e.step:e.value for e in b.Scalars(t)}
    for s in sorted(set(sa)&set(sb)):
        if sa[s]!=sb[s]:
            first[t]=(s,sa[s],sb[s]); break
    else: first[t]=None
div=[(v[0],t,v[1],v[2]) for t,v in first.items() if v]
div.sort()
print('identical tags:',sum(1 for v in first.values() if v is None))
for x in div[:15]: print(x)
for t in ['Train/mean_reward','Policy/mean_noise_std','Loss/learning_rate']:
    if t in tags:
        sa={e.step:e.value for e in a.Scalars(t)}; sb={e.step:e.value for e in b.Scalars(t)}
        for s in [0,10,19,20,21,22,25,30,50,100,500,1000,2000,2999]:
            if s in sa and s in sb: print(t,s,round(sa[s],5),round(sb[s],5))
