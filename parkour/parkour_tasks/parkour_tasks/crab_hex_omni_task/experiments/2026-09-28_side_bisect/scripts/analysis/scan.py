from common import *
import numpy as np
F=['X02','X07','X04','X08','X09']; S=['X01','X03','X05','X06','X10','X11','X12','X14','X15']
S_y=['X01','X03','X05','X06']  # +y-only slow
tags=sorted(set(D['X01'])&set(D['X10']))
for lo,hi in [(100,200),(200,300),(300,400),(400,500),(500,600),(600,800),(800,1000),(2500,3000)]:
    out=[]
    for t in tags:
        f=np.array([win(ser(x,t),lo,hi) for x in F]); s=np.array([win(ser(x,t),lo,hi) for x in S]); sy=np.array([win(ser(x,t),lo,hi) for x in S_y])
        if np.any(np.isnan(f)) or np.any(np.isnan(s)): continue
        rng=max(np.ptp(np.r_[f,s]),1e-12)
        gap=max(f.min()-s.max(), s.min()-f.max())
        gapy=max(f.min()-sy.max(), sy.min()-f.max())
        if gap>0 or gapy>0: out.append((gap/rng,gapy/rng,t,f.mean(),s.mean(),sy.mean()))
    out.sort(reverse=True)
    print(f'--- window {lo}-{hi}: tags separating FAST vs all SLOW (gap>0) or vs +y SLOW (gapy>0)')
    for g,gy,t,fm,sm,sym in out[:12]: print('  %+.2f %+.2f %-55s F %.4g  S %.4g  Sy %.4g'%(g,gy,t,fm,sm,sym))
