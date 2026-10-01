import glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
root = "/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments/"
runs = ["2026-08-13_0035_mirror_symmetry/fromscratch_sym_20k","2026-08-12_1550_seed_basin_search/seed2","2026-08-13_2000_onedir_spin/fromscratch_w0.3_20k","2026-08-13_2000_onedir_spin/screen_r4_synthesis","2026-08-13_1900_geometry_velocity_actions/canary_velact_ft","2026-08-12_1017_lean_reduction/L1_orientation_w-2.0"]
for r in runs:
    f = glob.glob(root+r+"/**/events.out.tfevents*", recursive=True)[0]
    ea = EventAccumulator(f, size_guidance={"scalars":0}); ea.Reload()
    tags = ea.Tags()["scalars"]
    g = lambda k: np.array([e.value for e in ea.Scalars(k)]) if k in tags else None
    lr=g("Loss/learning_rate"); sg=g("Policy/mean_noise_std"); sur=g("Loss/surrogate"); ent=g("Loss/entropy"); sym=g("Loss/symmetry"); vf=g("Loss/value_function"); rw=g("Train/mean_reward"); pr=g("Loss/priv_reg_coef")
    print("==", r, "tags:", [t for t in tags if t.startswith(("Loss","Policy"))])
    print(" LR pct 10/50/90/99:", np.percentile(lr,[10,50,90,99]).round(7), "mean", lr.mean().round(7), "frac>1e-4", (lr>1e-4).mean().round(3))
    # LR per window
    n=len(lr); w=max(n//10,1)
    print(" LR mean per decile:", [f"{lr[i:i+w].mean():.1e}" for i in range(0,n,w)])
    print(" sigma per decile:", [f"{sg[i:i+w].mean():.3f}" for i in range(0,n,w)], "n decreases>1e-4:", int((np.diff(sg)<-1e-4).sum()))
    print(" |surrogate| median:", np.median(np.abs(sur)).round(5), " entropy term 0.01*ent median:", (0.01*np.median(ent)).round(4), " VF median", np.median(vf).round(5))
    if sym is not None: print(" sym per decile:", [f"{sym[i:i+w].mean():.4f}" for i in range(0,n,w)], "max", sym.max().round(3), "argmax", int(sym.argmax()))
    if pr is not None: print(" priv_reg_coef first/last:", pr[0].round(4), pr[-1].round(4))
    print(" reward per decile:", [f"{rw[i:i+w].mean():.1f}" for i in range(0,n,w)])
