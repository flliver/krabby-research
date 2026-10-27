"""CPU census: saturation of RECORDED pre-clip deterministic actions (eval_crab_hex_gait.py:1043
buf['actions'] = policy(obs)) across every gait_eval run dir. No .pt reads."""
import glob, json, os, sys, re
import numpy as np
from multiprocessing import Pool
ROOT = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval"
MAXEP = 40

def one(meta_path):
    d = os.path.dirname(meta_path)
    try:
        m = json.load(open(meta_path))
    except Exception as e:
        return None
    files = sorted(glob.glob(os.path.join(d, "raw", "episode_*.npz")))
    if not files: return None
    rng = np.random.default_rng(0)
    if len(files) > MAXEP: files = list(rng.choice(files, MAXEP, replace=False))
    A = []
    for f in files:
        try:
            z = np.load(f); a = z["actions"]; done = z["done"]
        except Exception:
            continue
        alive = np.concatenate([[True], np.cumsum(done)[:-1] == 0]) if len(done) else np.zeros(0, bool)
        A.append(a[alive])
    if not A: return None
    A = np.abs(np.concatenate(A)).astype(np.float64)
    if A.size == 0: return None
    return dict(dir=os.path.relpath(d, ROOT), ckpt=m.get("checkpoint"), role=m.get("role"), task=m.get("task"),
                sample=m.get("sample_actions"), zero=m.get("zero_actions"), canon=m.get("canon"),
                n=int(A.shape[0]), f1=float((A > 1).mean()), f2=float((A > 2).mean()),
                med=float(np.median(A)), p90=float(np.quantile(A, 0.9)),
                perdim_min=float((A > 1).mean(0).min()), perdim_max=float((A > 1).mean(0).max()))

if __name__ == "__main__":
    metas = sorted(glob.glob(ROOT + "/**/run_meta.json", recursive=True))
    with Pool(4) as p:
        res = [r for r in p.map(one, metas) if r]
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "npz_sat.json"), "w"), indent=0)
    print(len(res))
