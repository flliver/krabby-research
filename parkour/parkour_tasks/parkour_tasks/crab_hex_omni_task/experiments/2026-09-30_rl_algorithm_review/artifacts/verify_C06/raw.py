import sys, glob, os, numpy as np
from tensorboard.backend.event_processing.event_file_loader import RawEventFileLoader
from tensorboard.compat.proto import event_pb2
R="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/"
for rel in sys.argv[1:]:
    fs=sorted(glob.glob(R+rel+"/events.out.tfevents*"))
    for p in fs:
        s={}; 
        for raw in RawEventFileLoader(p).Load():
            ev=event_pb2.Event.FromString(raw)
            for v in ev.summary.value:
                if v.tag in ("Policy/mean_noise_std","Loss/entropy"):
                    val=v.simple_value if v.HasField("simple_value") else (v.tensor.float_val[0] if v.tensor.float_val else None)
                    s.setdefault(v.tag,[]).append((ev.step,val))
        sd=np.array([x[1] for x in sorted(s["Policy/mean_noise_std"])]); st=np.array([x[0] for x in sorted(s["Policy/mean_noise_std"])])
        en=np.array([x[1] for x in sorted(s["Loss/entropy"])])
        idx=np.where(sd>=1.99)[0]
        dd=np.max(np.maximum.accumulate(sd)-sd)
        print(rel, os.path.basename(p)[-30:], "n",len(sd),"first",sd[0],"last",sd[-1],"min",sd.min(),"max",sd.max(),"frac>=1.99",(sd>=1.99).mean(),"first>=1.99 step",st[idx[0]] if len(idx) else None,"maxdrawdown",dd,"ent first/last/min/max",en[0],en[-1],en.min(),en.max())
