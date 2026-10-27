import sys, glob, os, json, numpy as np
from scipy.stats import norm
def stats(ev, sigma=None, sub=False, n_ep=None, seed=0):
    files = sorted(glob.glob(os.path.join(ev, "raw", "episode_*.npz")))
    if n_ep:
        rng = np.random.default_rng(seed); rng.shuffle(files); files = files[:n_ep]
    A = []
    for f in files:
        d = np.load(f)
        a = d["actions"]
        alive = ~np.cumsum(d["done"]).astype(bool)
        idx = np.arange(12 if sub else 0, len(a)); idx = idx[alive[idx]]
        if sub: idx = idx[::3]
        A.append(a[idx])
    A = np.abs(np.concatenate(A)).astype(np.float64)
    out = dict(n=A.shape[0], gt1=(A > 1).mean(), gt2=(A > 2).mean(), med=np.median(A), p90=np.quantile(A, .9),
               perdim_gt1_min=(A > 1).mean(0).min(), perdim_gt1_max=(A > 1).mean(0).max())
    if sigma:
        out["P_in"] = (norm.cdf((1 - A) / sigma) - norm.cdf((-1 - A) / sigma)).mean()
    return out
for ev in sys.argv[1:]:
    m = json.load(open(os.path.join(ev, "run_meta.json")))
    print(m["checkpoint"])
    for sub in (False, True):
        s = stats(ev, sigma=2.0, sub=sub, n_ep=(120 if sub else None))
        print("  sub" if sub else "  all", {k: (round(float(v), 4) if not isinstance(v, int) else v) for k, v in s.items()})
