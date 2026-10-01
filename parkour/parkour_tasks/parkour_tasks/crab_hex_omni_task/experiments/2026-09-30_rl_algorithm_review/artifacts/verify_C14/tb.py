import sys, glob, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/'
out={}
for d in sys.argv[1:]:
    f=glob.glob(base+d+'/events*')
    ea=EventAccumulator(f[0], size_guidance={'scalars':0}); ea.Reload()
    tags=ea.Tags()['scalars']
    r={}
    for t in ['Loss/learning_rate','Policy/mean_noise_std','Loss/priv_reg_coef','Loss/entropy']:
        if t not in tags: r[t]='absent'; continue
        s=ea.Scalars(t); v=[x.value for x in s]; st=[x.step for x in s]
        n=len(v)
        r[t]=dict(n=n, first_step=st[0], last_step=st[-1], first=v[:3], last=v[-1], min=min(v), max=max(v),
          frac_le_1p01e5=sum(1 for x in v if x<=1.01e-5)/n,
          frac_ge_1p99=sum(1 for x in v if x>=1.99)/n,
          frac_first_half_floor=sum(1 for x in v[:n//2] if x<=1.01e-5)/max(1,n//2),
          quartiles=[v[0],v[n//4],v[n//2],v[3*n//4],v[-1]],
          mean=sum(v)/n)
    out[d]=r
print(json.dumps(out,indent=1))
