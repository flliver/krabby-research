import sys, glob, os, json, numpy as np
from scipy.stats import norm
G="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/"
EVALS={
 "W":"omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29",
 "X08c6":"omni_phases/2026-09-22_gait_motion_quality/reeval_X08-c6/suite__omni/seed001/2026-09-23_06-23-30",
 "X01":"omni_phases/2026-09-22_gait_motion_quality/seed3_X01_c6ft_A15pB/suite__omni/seed001/2026-09-23_07-24-36",
 "M1":"omni_phases/2026-09-22_gait_motion_quality/seed3_X03_c6ft_A15pB/suite__omni/seed001/2026-09-23_09-11-38",
 "X19side":None, "fwd199":None, "omni1a999":None, "smoke19":None,
}
# resolve wildcards
def find(pat):
    r=sorted(glob.glob(G+pat)); return os.path.relpath(r[-1],G) if r else None
EVALS["X19side"]=find("omni_phases/2026-09-28_side_bisect/seed3_X19_s1_A15pB/side22_25__omni/seed001/*")
EVALS["fwd199"]=find("morph_x_exposure/smoke_001_A15*/*/seed001/*")
EVALS["omni1a999"]=find("omni_phases/2026-09-16_1805_omni_formation/seed3_1a_001_A15pB/slow__omni/seed001/*")
EVALS["smoke19"]=find("omni_phases/v6_drill/seed3_X01_1a_A15pB/suite__omni/seed001/*")
def stats(ev, sub=False):
    files=sorted(glob.glob(os.path.join(ev,"raw","episode_*.npz")))
    nf=len(files)
    if sub:
        rng=np.random.default_rng(0); rng.shuffle(files); files=files[:120]
    A=[]
    for f in files:
        d=np.load(f); a=d["actions"]; done=d["done"]
        alive=np.concatenate([[True],np.cumsum(done)[:-1]==0])  # include the done step
        idx=np.arange(12 if sub else 0,len(a)); idx=idx[alive[idx]]
        if sub: idx=idx[::3]
        A.append(a[idx])
    A=np.abs(np.concatenate(A)).astype(np.float64)
    pin=(norm.cdf((1-A)/2.0)-norm.cdf((-1-A)/2.0)).mean()
    pd=(A>1).mean(0)
    return dict(n_ep=nf, n=int(A.shape[0]), gt1=(A>1).mean(), gt2=(A>2).mean(), med=np.median(A), p90=np.quantile(A,.9), P_in=pin, pd_min=pd.min(), pd_max=pd.max(), ndim=A.shape[1])
for k,rel in EVALS.items():
    if rel is None: print(k,"NOT FOUND"); continue
    ev=G+rel; m=json.load(open(os.path.join(ev,"run_meta.json")))
    print(f"== {k}  ckpt={m.get('checkpoint')}  sample_actions={m.get('sample_actions')} zero={m.get('zero_actions')} canon={m.get('canon')}  dir={rel}")
    for sub in ((False,True) if k=="W" else (False,)):
        s=stats(ev,sub)
        print("   "+("sub120" if sub else "all   "), {a:(round(float(b),4) if not isinstance(b,(int,np.integer)) else int(b)) for a,b in s.items()})
