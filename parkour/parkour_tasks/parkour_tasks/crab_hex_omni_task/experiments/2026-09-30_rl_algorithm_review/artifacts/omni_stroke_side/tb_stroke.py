import glob, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
R = "/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
RUNS = {"Bf_s3":"2026-09-26_17-42-44_seed3_X01_t3","Bf_s2":"2026-09-26_22-54-02_seed2_X02_t3","Sf_s3":"2026-09-27_00-15-39_seed3_X03_t3",
"Sf_s2":"2026-09-27_01-35-20_seed2_X04_t3","Pf_s3":"2026-09-27_02-56-58_seed3_X05_t3","Pf_s2":"2026-09-27_04-16-39_seed2_X06_t3",
"Scatf_s3":"2026-09-28_16-40-58_seed3_X01_t3","Scatf_s2":"2026-09-28_18-05-55_seed2_X02_t3",
"B_c6y_s3":"2026-09-26_03-01-22_seed3_X01_c6y","B_c6y_s2":"2026-09-26_04-47-00_seed2_X02_c6y","S_c6y_s3":"2026-09-26_06-32-38_seed3_X03_c6y","P_c6y_s3":"2026-09-26_08-21-34_seed3_X04_c6y",
"Scat_W_s3":"2026-09-28_08-11-28_seed3_X01_c6y","Scat_W_s2":"2026-09-28_14-37-59_seed2_X02_c6y"}
tags=["Loss/learning_rate","Policy/mean_noise_std","Loss/entropy","Loss/value_function","Loss/priv_reg_coef","Loss/priv_reg","Train/mean_reward","Metrics/base_parkour/track_ratio_mean","Perf/total_fps","Train/mean_episode_length"]
for name, ts in RUNS.items():
    ds=glob.glob(R+ts+"*")
    if not ds: print(name,"MISSING"); continue
    ea=EventAccumulator(ds[0],size_guidance={"scalars":0}); ea.Reload()
    have=ea.Tags()["scalars"]; out=[]
    for t in tags:
        if t not in have: out.append(t.split('/')[-1]+" n/a"); continue
        ev=ea.Scalars(t); s=np.array([e.step for e in ev]); v=np.array([e.value for e in ev])
        s0=s.min(); L=s.max()-s0
        def win(a,b):
            m=(s>=s0+a)&(s<s0+b); return v[m].mean() if m.any() else np.nan
        if t=="Loss/learning_rate":
            out.append("lr med %.1e floor%% %.0f"%(np.median(v),100*(v<=1.0001e-5).mean()))
        elif t=="Policy/mean_noise_std":
            out.append("std first %.2f @300 %.2f @1000 %.2f last %.2f"%(v[0],win(250,350),win(950,1050),win(L-100,L+1)))
        elif t=="Loss/priv_reg_coef":
            out.append("prc first %.3g last %.3g"%(v[0],v[-1]))
        elif t=="Perf/total_fps":
            out.append("fps %.0f"%np.median(v))
        else:
            out.append(t.split('/')[-1]+" %.3g->%.3g"%(win(0,100),win(L-100,L+1)))
    print(name, "steps %d-%d"%(s0,s.max()), " | ".join(out))
