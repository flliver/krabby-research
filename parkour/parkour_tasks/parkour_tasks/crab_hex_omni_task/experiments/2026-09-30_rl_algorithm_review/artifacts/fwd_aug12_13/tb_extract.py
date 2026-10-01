import sys, glob, os, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root = "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/"
files = sorted(glob.glob(root+"2026-08-1[2-3]*/**/events.out.tfevents*", recursive=True))
keys = ["Loss/learning_rate","Policy/mean_noise_std","Loss/entropy","Loss/value_function","Loss/surrogate","Loss/symmetry","Loss/priv_reg_coef","Perf/total_fps","Train/mean_reward","Train/mean_episode_length","Loss/depth_actor_loss"]
out = {}
for f in files:
    rel = f.replace(root,"")
    ea = EventAccumulator(f, size_guidance={"scalars":0}); ea.Reload()
    tags = ea.Tags()["scalars"]
    r = {"tags_n": len(tags)}
    for k in keys:
        if k not in tags: continue
        ev = ea.Scalars(k)
        steps = [e.step for e in ev]; vals=[e.value for e in ev]
        n=len(vals)
        pick = {}
        for q in [0,1,5,10,50,100,500,1000,2000,2999,5000,10000,19999]:
            # find first index with step >= steps[0]+q
            tgt = steps[0]+q
            idx = next((i for i,s in enumerate(steps) if s>=tgt), None)
            if idx is not None: pick[str(steps[idx])] = round(vals[idx],5)
        pick["last:"+str(steps[-1])] = round(vals[-1],5)
        r[k] = {"n":n,"min":round(min(vals),5),"max":round(max(vals),5),"pts":pick}
    r["other_tags"] = [t for t in tags if t.startswith(("Loss","Policy","Perf"))]
    out[rel]=r
    print(rel, file=sys.stderr)
json.dump(out, open(sys.argv[1],"w"), indent=1)
