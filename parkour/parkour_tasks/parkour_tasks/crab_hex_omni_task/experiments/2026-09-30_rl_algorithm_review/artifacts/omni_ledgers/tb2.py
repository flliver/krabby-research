import sys, glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
for run in sys.argv[1:]:
    e=glob.glob(base+run+"/events.out.tfevents.*")[0]
    ea=EventAccumulator(e,size_guidance={"scalars":0}); ea.Reload()
    tags=ea.Tags()["scalars"]
    print("==",run); print("  kl-ish tags:",[t for t in tags if 'kl' in t.lower() or 'Loss/' in t])
    lr=[x.value for x in ea.Scalars("Loss/learning_rate")]
    print(f"  LR: frac==1e-5 {sum(1 for v in lr if v<=1.0001e-5)/len(lr):.3f}  frac>1e-4 {sum(1 for v in lr if v>1e-4)/len(lr):.3f} median {sorted(lr)[len(lr)//2]:.3g}")
    sd=ea.Scalars("Policy/mean_noise_std")
    first2=[x.step for x in sd if x.value>=1.999]
    print("  first step std>=1.999:", first2[0] if first2 else None, " std at +50/+100/+200/+500:", [round(sd[i].value,3) for i in (50,100,200,500) if i<len(sd)])
