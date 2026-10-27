import sys; sys.path.insert(0,'.')
from common import *
SL=[x for x in SLOW if x not in ('X03',)]  # R has a different reward economy
tags=sorted(t for t in load('X10') if t.startswith(('Episode_Reward','Metrics','Loss','Policy','Train','Episode_Term')) and 'pool_' not in t)
def S(x,t,k=50):
    s,v=ser(x,t)
    if s is None or s.max()>=3100: return None
    a=np.full(3100,np.nan); a[s.astype(int)]=v; return smooth(np.nan_to_num(a,nan=np.nanmean(v)),k)
res=[]
for t in tags:
    F=[S(x,t) for x in FAST]; Sl=[S(x,t) for x in SL]
    if any(f is None for f in F+Sl): continue
    m=min(1500,min(len(np.where(~np.isnan(z))[0]) for z in Sl))  # X19 partial
    F=np.stack([f[:1500] for f in F]); Sl=np.stack([z[:1500] for z in Sl])
    for sign in (+1,-1):
        gap=sign*(F.min(0) if sign>0 else F.max(0)) - sign*(Sl.max(0) if sign>0 else Sl.min(0))
        ok=gap>0
        ok[:49]=False
        # earliest t0 such that ok holds for all t in [t0,1500)
        bad=np.where(~ok)[0]
        t0=bad[-1]+1 if len(bad) else 0
        if t0<1500:
            res.append((t0,t,'F>S' if sign>0 else 'F<S', float(np.nanmean(F[:,t0:t0+50])), float(np.nanmean(Sl[:,t0:t0+50]))))
res.sort()
for r in res[:25]: print('%5d %-60s %s F=%.4f S=%.4f'%r)
