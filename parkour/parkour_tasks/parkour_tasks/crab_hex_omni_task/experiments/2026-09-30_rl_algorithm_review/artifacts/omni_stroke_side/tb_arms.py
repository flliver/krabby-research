import glob, sys, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
R = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
TAG = {"O":"2026-09-29_00-40-18","C":"2026-09-29_01-52-56","R":"2026-09-29_08-28-30","C_mo":"2026-09-29_09-39-22",
       "C_band":"2026-09-29_10-52-23","C_rsi":"2026-09-29_12-09-37","C_s2":"2026-09-29_13-23-08","C_mo_s2":"2026-09-29_14-36-09",
       "C_mo_s1":"2026-09-29_18-09-45","LRfa":"2026-09-29_19-18-51","LR4":"2026-09-29_20-33-42","LR4_dwell":"2026-09-29_22-15-30",
       "C_lr":"2026-09-30_00-00-38","C_lr_10":"2026-09-30_01-15-33","C_lr_rsi":"2026-09-30_09-05-43","C_lr_10_rsi":"2026-09-30_10-35-34","C_lr_rsi33":"2026-09-30_12-27-45"}
want = ["Loss/learning_rate","Policy/mean_noise_std","Loss/entropy","Loss/value_function","Loss/surrogate","Loss/symmetry","Perf/total_fps","Train/mean_reward","Metrics/base_parkour/track_ratio_mean","Perf/iteration_time","Perf/collection time","Perf/learning_time"]
its = [0, 5, 50, 300, 600, 1000, 2000, 2990]
out = {}
for arm, ts in TAG.items():
    ds = glob.glob(R + ts + "*")
    if not ds: print(arm, "missing"); continue
    ea = EventAccumulator(ds[0], size_guidance={"scalars": 0}); ea.Reload()
    tags = ea.Tags()["scalars"]
    if arm == "O": print([t for t in tags if t.startswith(("Loss","Policy","Perf","Train"))])
    row = {}
    for w in want:
        if w not in tags: row[w] = None; continue
        ev = ea.Scalars(w); s = np.array([e.step for e in ev]); v = np.array([e.value for e in ev])
        vals = {}
        for i in its:
            m = (s >= i - 5) & (s <= i + 5) if i < 60 else (s >= i - 50) & (s <= i + 50)
            vals[i] = float(np.mean(v[m])) if m.any() else None
        vals["max"] = float(v.max()); vals["min"]=float(v.min()); vals["n"]=int(len(v))
        # first step std hits >=1.99
        if w == "Policy/mean_noise_std":
            hit = s[v >= 1.99]
            vals["first_ge_1.99"] = int(hit[0]) if len(hit) else None
            vals["frac_ge_1.99"] = float((v>=1.99).mean())
        if w == "Loss/learning_rate":
            vals["frac_at_floor"] = float((v <= 1.0001e-5).mean())
            vals["first5"] = [float(x) for x in v[:5]]
        row[w] = vals
    out[arm] = row
json.dump(out, open("tb_arms.json","w"), indent=1)
for arm,row in out.items():
    def g(t,k): 
        r=row.get(t); 
        return None if r is None else r.get(k)
    print(arm, "lr0", g("Loss/learning_rate","first5"), "lr_floorfrac", g("Loss/learning_rate","frac_at_floor"),
      "| std 300/600/1000/2990", [None if g("Policy/mean_noise_std",k) is None else round(g("Policy/mean_noise_std",k),3) for k in (0,300,600,1000,2990)], "first>=1.99", g("Policy/mean_noise_std","first_ge_1.99"),
      "| ent", [None if g("Loss/entropy",k) is None else round(g("Loss/entropy",k),2) for k in (0,600,2990)],
      "| vf", [None if g("Loss/value_function",k) is None else round(g("Loss/value_function",k),3) for k in (50,600,2990)],
      "| sym", [None if g("Loss/symmetry",k) is None else round(g("Loss/symmetry",k),4) for k in (50,600,2990)],
      "| fps", None if g("Perf/total_fps",1000) is None else round(g("Perf/total_fps",1000)),
      "| rew", [None if g("Train/mean_reward",k) is None else round(g("Train/mean_reward",k),1) for k in (600,2990)])
