import os, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
R="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/"
runs={"L_w0":"crab_hex_flat_walk/2026-09-06_21-32-46","L_w1":"crab_hex_flat_walk/2026-09-06_23-59-54","L_w3":"crab_hex_flat_walk/2026-09-07_04-38-50","L_w4":"crab_hex_flat_walk/2026-09-07_06-57-52","L_w5":"crab_hex_flat_walk/2026-09-07_09-16-39","SD_s2_1a":"crab_hex_side_flat_walk/2026-09-12_00-31-09","SD_s2_2c":"crab_hex_side_teacher/2026-09-12_07-45-40","S_3a":"crab_hex_student/2026-09-08_05-54-01"}
os.nice(10)
for k,rel in runs.items():
    ea=EventAccumulator(R+rel,size_guidance={"scalars":0}); ea.Reload()
    tags=ea.Tags()["scalars"]
    lr=np.array([e.value for e in ea.Scalars("Loss/learning_rate")])
    st=np.array([e.step for e in ea.Scalars("Loss/learning_rate")])
    segs=[(0,100),(100,500),(500,1000),(1000,2500),(2500,5001)]
    s0=st[0]
    out=[]
    for a,b in segs:
        m=(st-s0>=a)&(st-s0<b)
        if m.any(): out.append(f"[{a}-{b}) med {np.median(lr[m]):.2e} p90 {np.percentile(lr[m],90):.2e}")
    print(k, "first10", " ".join(f"{x:.1e}" for x in lr[:10]))
    print("   ", " | ".join(out))
    if "Train/mean_reward" in tags:
        mr=np.array([e.value for e in ea.Scalars("Train/mean_reward")])
        print("    mean_reward first50 %.1f  at1000 %.1f last50 %.1f max %.1f"%(mr[:50].mean(), mr[min(1000,len(mr)-1)], mr[-50:].mean(), mr.max()))
    if "Train/mean_episode_length" in tags:
        el=np.array([e.value for e in ea.Scalars("Train/mean_episode_length")]); print("    ep_len first50 %.0f last50 %.0f"%(el[:50].mean(), el[-50:].mean()))
    print("    tags:", [t for t in tags if not t.startswith("Episode") and not t.startswith("Metrics") and not t.startswith("Curriculum")][:40])
