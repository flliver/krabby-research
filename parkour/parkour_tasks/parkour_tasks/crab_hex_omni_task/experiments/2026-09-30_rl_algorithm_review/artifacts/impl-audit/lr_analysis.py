import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tb_read import read_scalars
TAGS = ["Loss/learning_rate","Policy/mean_noise_std","Loss/entropy","Loss/surrogate","Loss/value_function",
        "Loss/symmetry","Loss/priv_reg","Loss/priv_reg_coef","Loss/estimator","Loss/hist_latent","Train/mean_reward",
        "Train/mean_episode_length","Perf/total_fps"]
run = sys.argv[1]
freq = int(sys.argv[2]) if len(sys.argv) > 2 else 20
d = read_scalars(run, set(TAGS))
lr = dict(d["Loss/learning_rate"])
its = np.array(sorted(lr)); v = np.array([lr[i] for i in its])
print("run", run, "n_it", len(its), "first", its[0], "last", its[-1])
print("LR first 5:", [f"{x:.3g}" for x in v[:5]])
floor = np.isclose(v, 1e-5, rtol=1e-3)
print(f"frac at floor 1e-5: {floor.mean():.3f}; median LR {np.median(v):.3g}; mean {v.mean():.3g}; max {v.max():.3g}")
dag = (its % freq == 0)
print(f"dagger its (it%{freq}==0): n={dag.sum()} frac_floor={floor[dag].mean():.3f} median={np.median(v[dag]):.3g}")
print(f"non-dagger its: n={(~dag).sum()} frac_floor={floor[~dag].mean():.3f} median={np.median(v[~dag]):.3g}")
# LR the iteration after dagger vs before
for lag in range(0,4):
    m = (its % freq == lag)
    print(f"  it%{freq}=={lag}: median LR {np.median(v[m]):.3g}  frac_floor {floor[m].mean():.3f}")
# ratio of LR at dagger iteration vs previous iteration
prev = {i: lr.get(i-1) for i in its[dag] if (i-1) in lr}
r = [lr[i]/p for i,p in prev.items() if p]
if r: print(f"LR(dagger it)/LR(prev it): median {np.median(r):.3g}, min {np.min(r):.3g}, max {np.max(r):.3g}; implied net /1.5 steps median {np.median(np.log(r)/np.log(1.5)):.2f}")
# per-window
for a,b in [(0,100),(100,500),(500,1000),(1000,2000),(2000,3000),(3000,6000),(6000,10000),(10000,20000),(20000,40000)]:
    m = (its>=a)&(its<b)
    if m.sum(): print(f"  its [{a},{b}): median LR {np.median(v[m]):.3g} frac_floor {floor[m].mean():.3f} (dag median {np.median(v[m&dag]) if (m&dag).sum() else float('nan'):.3g}, nondag median {np.median(v[m&~dag]):.3g})")
for t in TAGS[1:]:
    if t in d:
        x = np.array([y for _,y in d[t]])
        print(f"{t}: first {x[0]:.4g} p50 {np.median(x):.4g} last {x[-1]:.4g} max {x.max():.4g} min {x.min():.4g}")
std = np.array([y for _,y in d["Policy/mean_noise_std"]]); si = np.array([s for s,_ in d["Policy/mean_noise_std"]])
above = std >= 2.0
if above.any(): print("first it with mean std>=2.0:", si[above.argmax()], " frac its >=2.0:", above.mean().round(3), "max std", std.max())
