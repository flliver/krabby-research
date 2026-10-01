import sys, os, glob, json, numpy as np
from multiprocessing import Pool
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
ROOT='/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_forward_task/experiments'
TAGS=['Loss/learning_rate','Policy/mean_noise_std','Loss/entropy','Loss/value_function','Loss/surrogate','Perf/total_fps','Train/mean_reward','Train/mean_episode_length','Metrics/base_velocity/error_vel_xy','Episode_Termination/crab_failure','Loss/priv_reg_coef','Perf/collection time','Perf/learning_time']
def summ(d):
    try:
        ea=EventAccumulator(d,size_guidance={'scalars':0}); ea.Reload()
        out={'run':os.path.relpath(d,ROOT)}
        tags=ea.Tags()['scalars']
        for t in TAGS:
            if t not in tags: continue
            ev=ea.Scalars(t); s=np.array([e.step for e in ev]); v=np.array([e.value for e in ev])
            n=len(v)
            r={'n':n,'step0':int(s[0]),'stepN':int(s[-1]),'first':float(v[0]),'v1':float(v[min(1,n-1)]),'v10':float(v[min(10,n-1)]),'v100':float(v[min(100,n-1)]),'last':float(v[-1]),'min':float(v.min()),'max':float(v.max()),'median':float(np.median(v))}
            if t=='Loss/learning_rate':
                r['frac_at_floor']=float(np.mean(v<=1.0001e-5)); r['frac_ge_1e-4']=float(np.mean(v>=1e-4))
                r['first_floor_idx']=int(np.argmax(v<=1.0001e-5)) if (v<=1.0001e-5).any() else -1
                r['mean_last1000']=float(v[-1000:].mean())
            if t=='Policy/mean_noise_std':
                r['frac_ge_1.99']=float(np.mean(v>=1.99)); r['first_ge_1.99_idx']=int(np.argmax(v>=1.99)) if (v>=1.99).any() else -1
            out[t]=r
        wt=[e.wall_time for e in ea.Scalars('Train/mean_reward')] if 'Train/mean_reward' in tags else []
        if wt: out['wall_h']=(wt[-1]-wt[0])/3600
        return out
    except Exception as e:
        return {'run':d,'err':str(e)}
if __name__=='__main__':
    dirs=sorted({os.path.dirname(p) for p in glob.glob(ROOT+'/2026-08-0*/**/events.out*',recursive=True)+glob.glob(ROOT+'/2026-08-10_0058_tripod_stability/**/events.out*',recursive=True)})
    with Pool(4) as p: res=p.map(summ,dirs)
    json.dump(res,open(sys.argv[1],'w'),indent=1)
    print(len(res))
