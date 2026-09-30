import pickle, numpy as np
F=['X02','X07','X08','X09']; S=['X01','X03','X05','X06','X10','X11','X12','X14','X15']; I=['X04']
D={x:pickle.load(open(f'tb/{x}.pkl','rb')) for x in F+S+I}
tags=set.intersection(*[set(d) for d in D.values()])
def w(d,t,a,b):
    s=np.array(d[t]); m=(s[:,0]>=a)&(s[:,0]<b); return float(np.nanmean(s[m,1])) if m.any() else np.nan
res=[]
for win in [(800,1000),(2900,3000)]:
    for t in tags:
        f=np.array([w(D[x],t,*win) for x in F]); s=np.array([w(D[x],t,*win) for x in S])
        if np.isnan(f).any() or np.isnan(s).any(): continue
        rng=max(f.max(),s.max())-min(f.min(),s.min())
        if rng<=0: continue
        gap=max(f.min()-s.max(), s.min()-f.max())/rng
        res.append((gap,win,t,f,s,w(D['X04'],t,*win)))
res.sort(key=lambda r:-r[0])
for g,win,t,f,s,i in res[:40]:
    print(f"{g:+.2f} {win} {t:60s} F[{f.min():.3f},{f.max():.3f}] S[{s.min():.3f},{s.max():.3f}] C_mo {i:.3f}")
