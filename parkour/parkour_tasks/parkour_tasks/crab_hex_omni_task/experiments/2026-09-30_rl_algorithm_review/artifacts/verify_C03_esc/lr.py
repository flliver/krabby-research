import glob, numpy as np, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
L="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/"
runs={"X17":L+"crab_hex_omni_flat_walk/2026-09-30_10-35-34_seed3_X17_s1_A15pB",
      "X19":None,"W_c6x":None}
import os
for d in sorted(glob.glob(L+"crab_hex_omni_flat_walk/*")):
    b=os.path.basename(d)
    if "12-27-45" in b: runs["X19"]=d
    if "2026-09-25_14-53-27" in b: runs["W_c6x"]=d
print(runs)
for name,d in runs.items():
    if d is None: continue
    ea=EventAccumulator(d,size_guidance={"scalars":0}); ea.Reload()
    ev=ea.Scalars("Loss/learning_rate"); it=np.array([e.step for e in ev]); v=np.array([e.value for e in ev])
    print("==",name,"n=",len(v),"iters",it.min(),it.max())
    print(" first 8:",[(int(i),float(f"{x:.4g}")) for i,x in zip(it[:8],v[:8])])
    r=v[1:]/v[:-1]; j=np.argmax(r); print(" max jump x%.1f at it %d (%.3g -> %.3g); log1.5 = %.2f"%(r[j],it[j+1],v[j],v[j+1],np.log(r[j])/np.log(1.5)))
    print(" jumps >= x11.4 (1.5^6): n=%d; >= x38.4: n=%d"%((r>=11.3).sum(),(r>=38.3).sum()))
    # what follows a big jump?
    big=np.where(r>=11.3)[0]+1
    if len(big):
        nxt=v[np.minimum(big+1,len(v)-1)]; nxt2=v[np.minimum(big+2,len(v)-1)]
        print(" after big jump: next==floor frac %.2f, next2==floor frac %.2f"%((nxt<=1.0001e-5).mean(),(nxt2<=1.0001e-5).mean()))
    for lo,hi in [(0,100),(0,250),(0,500),(100,500),(250,500),(500,1000),(500,1500),(1000,2000),(1500,3000),(2750,3000)]:
        m=(it>=lo)&(it<hi)
        if m.sum()==0: continue
        w=v[m]; rr=w[1:]/w[:-1] if len(w)>1 else np.array([1.0])
        print("  it[%d,%d): n=%d mean=%.3g median=%.3g max=%.3g floor%%=%.0f >1e-4%%=%.1f maxjump=x%.1f"%(lo,hi,m.sum(),w.mean(),np.median(w),w.max(),100*(w<=1.0001e-5).mean(),100*(w>1e-4).mean(),rr.max()))
    # last iteration above 1e-4
    ab=it[v>1e-4]; print(" last iter with LR>1e-4:", int(ab.max()) if len(ab) else None)
