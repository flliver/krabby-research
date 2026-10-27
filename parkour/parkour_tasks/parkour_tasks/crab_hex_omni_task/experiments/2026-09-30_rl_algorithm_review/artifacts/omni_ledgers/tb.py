import sys, glob, os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
base="/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/"
tags=["Loss/learning_rate","Policy/mean_noise_std","Loss/entropy","Loss/value_function","Loss/surrogate","Loss/symmetry","Loss/priv_reg_coef","Perf/total_fps","Train/mean_reward","Train/mean_episode_length"]
for run in sys.argv[1:]:
    ev=glob.glob(base+run+"/events.out.tfevents.*")
    for e in ev:
        ea=EventAccumulator(e,size_guidance={"scalars":0}); ea.Reload()
        avail=ea.Tags()["scalars"]
        print("==",run, os.path.basename(e))
        for t in tags:
            if t not in avail: print(" ",t,"MISSING"); continue
            s=ea.Scalars(t); v=[x.value for x in s]; st=[x.step for x in s]
            n=len(v)
            def m(a,b): 
                seg=v[a:b]; return sum(seg)/max(1,len(seg))
            print(f"  {t}: n={n} steps {st[0]}..{st[-1]} first={v[0]:.4g} v[1]={v[1] if n>1 else float('nan'):.4g} mean_first100={m(0,100):.4g} mean_last100={m(n-100,n):.4g} min={min(v):.4g} max={max(v):.4g}")
