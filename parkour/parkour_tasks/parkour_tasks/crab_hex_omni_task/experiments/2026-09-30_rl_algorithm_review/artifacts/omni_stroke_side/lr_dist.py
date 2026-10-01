import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
R = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
for ts in ["2026-09-29_00-40-18","2026-09-29_01-52-56","2026-09-29_20-33-42","2026-09-30_12-27-45"]:
    d=glob.glob(R+ts+"*")[0]
    ea=EventAccumulator(d,size_guidance={"scalars":0}); ea.Reload()
    v=np.array([e.value for e in ea.Scalars("Loss/learning_rate")])
    print(ts, "median %.2e mean %.2e p10 %.2e p90 %.2e max %.2e"%(np.median(v),v.mean(),np.percentile(v,10),np.percentile(v,90),v.max()),
      "frac<=1e-5 %.2f frac>=1e-4 %.2f frac>=3e-4 %.2f"%((v<=1.0001e-5).mean(),(v>=1e-4).mean(),(v>=3e-4).mean()))
    # geometric mean
    print("  geomean %.2e"%np.exp(np.log(v).mean()), "first 20:", " ".join("%.1e"%x for x in v[:20]))
