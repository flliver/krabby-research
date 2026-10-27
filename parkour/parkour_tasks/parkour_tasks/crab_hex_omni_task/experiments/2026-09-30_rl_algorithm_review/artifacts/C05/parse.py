import sys, glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
tags=["Policy/mean_noise_std","Loss/entropy","Loss/learning_rate"]
for run in sys.argv[1:]:
    f=glob.glob(run+"/events.out.tfevents.*")[0]
    ea=EventAccumulator(f,size_guidance={"scalars":0,"histograms":1,"images":1,"tensors":1,"compressedHistograms":1}); ea.Reload()
    print("==",run)
    for t in tags:
        if t not in ea.Tags()["scalars"]: print(" missing",t); continue
        ev=ea.Scalars(t); s=np.array([e.step for e in ev]); v=np.array([e.value for e in ev])
        idx=[0,1,2,5,10,20,50,100,200,500,1000,len(v)-1]
        pts=", ".join(f"{s[i]}:{v[i]:.4f}" for i in idx if i<len(v))
        print(f" {t} n={len(v)} min={v.min():.4f} max={v.max():.4f} last={v[-1]:.4f}")
        print("   ",pts)
        if t=="Policy/mean_noise_std":
            first=np.argmax(v>=1.999) if (v>=1.999).any() else -1
            print(f"   first step >=1.999: {s[first] if first>=0 else None}; frac steps >2.0: {(v>2.0).mean():.3f}; frac >=1.999: {(v>=1.999).mean():.3f}")
