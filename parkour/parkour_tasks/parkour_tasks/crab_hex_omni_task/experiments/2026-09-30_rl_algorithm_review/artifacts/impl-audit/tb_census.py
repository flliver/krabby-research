import sys, os, glob, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from tb_read import read_scalars
ROOT = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl"
TAGS = {"Loss/learning_rate","Policy/mean_noise_std","Loss/entropy","Loss/surrogate","Loss/value_function",
        "Loss/symmetry","Loss/priv_reg","Loss/priv_reg_coef","Loss/estimator","Loss/hist_latent","Train/mean_reward",
        "Train/mean_episode_length","Perf/total_fps","Perf/collection time","Perf/learning_time",
        "Loss_depth/depth_actor_loss","Loss_depth/yaw_loss"}
def one(run):
    try:
        d = read_scalars(run, TAGS)
    except Exception as e:
        return {"run": run, "err": str(e)}
    out = {"run": run.replace(ROOT + "/", "")}
    def arr(t):
        if t not in d: return None, None
        s = sorted(d[t]); return np.array([a for a, _ in s]), np.array([b for _, b in s])
    it, lr = arr("Loss/learning_rate")
    if it is None: out["n"] = 0; return out
    out.update(n=int(len(it)), it0=int(it[0]), itN=int(it[-1]), lr_first=float(lr[0]), lr_med=float(np.median(lr)),
               lr_mean=float(lr.mean()), lr_max=float(lr.max()), lr_floor_frac=float(np.isclose(lr, 1e-5, rtol=1e-3).mean()),
               lr_le_3p4e5=float((lr <= 3.4e-5).mean()), lr_last=float(lr[-1]),
               lr_med_after100=float(np.median(lr[it >= it[0] + 100])) if (it >= it[0] + 100).any() else None)
    for t, k in [("Policy/mean_noise_std", "std"), ("Loss/entropy", "ent"), ("Loss/priv_reg_coef", "prc"), ("Loss/priv_reg", "pr"),
                 ("Loss/value_function", "vf"), ("Loss/surrogate", "surr"), ("Loss/symmetry", "sym"), ("Perf/total_fps", "fps"),
                 ("Train/mean_episode_length", "eplen"), ("Train/mean_reward", "rew"), ("Loss/estimator", "est"), ("Loss/hist_latent", "hl"),
                 ("Perf/collection time", "tcol"), ("Perf/learning_time", "tlearn")]:
        i, v = arr(t)
        if i is None: continue
        out[k + "_first"] = float(v[0]); out[k + "_last"] = float(v[-1]); out[k + "_med"] = float(np.median(v)); out[k + "_max"] = float(v.max())
        if k == "std":
            hit = np.nonzero(v >= 1.999)[0]
            out["std_first_it_2"] = int(i[hit[0]]) if len(hit) else None
            out["std_frac_ge2"] = float((v >= 1.999).mean())
    return out
if __name__ == "__main__":
    runs = sorted(set(os.path.dirname(f) for f in glob.glob(f"{ROOT}/*/*/events.out.tfevents*") if "/gait_eval/" not in f))
    if len(sys.argv) > 1: runs = [r for r in runs if sys.argv[1] in r]
    with Pool(4) as p:
        res = p.map(one, runs, chunksize=2)
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "tb_census.json"), "w"), indent=0)
    print("done", len(res))
