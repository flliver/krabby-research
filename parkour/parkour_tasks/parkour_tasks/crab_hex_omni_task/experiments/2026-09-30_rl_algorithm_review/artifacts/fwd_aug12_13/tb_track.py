import glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root = "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/"
for r in ["2026-08-13_0035_mirror_symmetry/fromscratch_sym_20k","2026-08-13_2000_onedir_spin/fromscratch_w0.3_20k","2026-08-13_2000_onedir_spin/screen_w0.3","2026-08-13_2000_onedir_spin/screen_sigma_fix","2026-08-12_1550_seed_basin_search/seed6"]:
    f = glob.glob(root+r+"/**/events.out.tfevents*", recursive=True)[0]
    ea = EventAccumulator(f, size_guidance={"scalars":0}); ea.Reload()
    tags=ea.Tags()["scalars"]
    k=[t for t in tags if "track_lin_vel_xy" in t][0]
    s=np.array([e.step for e in ea.Scalars(k)]); v=np.array([e.value for e in ea.Scalars(k)])
    pts=[0,500,1000,2500,2999,5000,10000,19999]
    print(r, k, {p: round(float(v[np.argmin(abs(s-p))]),3) for p in pts if p<=s[-1]})
