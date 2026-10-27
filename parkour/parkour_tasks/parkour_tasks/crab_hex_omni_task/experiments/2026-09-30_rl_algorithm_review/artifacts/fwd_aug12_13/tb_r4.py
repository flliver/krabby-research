import glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root = "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/"
for r in ["2026-08-13_2000_onedir_spin/screen_r4_synthesis","2026-08-13_0035_mirror_symmetry/fromscratch_sym_20k"]:
    f = glob.glob(root+r+"/**/events.out.tfevents*", recursive=True)[0]
    ea = EventAccumulator(f, size_guidance={"scalars":0}); ea.Reload()
    g = lambda k: (np.array([e.step for e in ea.Scalars(k)]),np.array([e.value for e in ea.Scalars(k)]))
    s,sym=g("Loss/symmetry"); _,vf=g("Loss/value_function"); _,lr=g("Loss/learning_rate")
    print(r)
    idx=np.where(sym>1)[0]; print(" iters sym>1:", len(idx), s[idx][:10], sym[idx][:10].round(2), "... last10", sym[-10:].round(1))
    print(" VF>1 iters:", s[np.where(vf>1)[0]][:10], vf[vf>1][:10].round(2))
    print(" LR first 30:", [f"{x:.0e}" for x in lr[:30]])
