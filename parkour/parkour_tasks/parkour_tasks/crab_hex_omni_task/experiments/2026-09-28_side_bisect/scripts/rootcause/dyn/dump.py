import glob, pickle, sys, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
for d in sorted(glob.glob(R+"2026-09-29_*_s1_A15pB")+glob.glob(R+"2026-09-30_*_s1_A15pB")):
    x=d.split("_")[-3]
    ea=EventAccumulator(d,size_guidance={"scalars":0}); ea.Reload()
    out={t:[(e.step,e.value) for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}
    pickle.dump(out,open(f"tb/{x}.pkl","wb"))
    print(x, os.path.basename(d), len(out), max(s for s,_ in out.get("Train/mean_reward",[(0,0)])))
