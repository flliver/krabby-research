import numpy as np, glob, json
S='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39'
W='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29'
def stats(files, seg_ok):
    frac=[];maxrun=[]
    for f in files:
        z=np.load(f); seg=z['segment_id']; m=seg_ok(seg)&z['steady_mask']
        if m.sum()<50: continue
        c=z['foot_force_norm'][m]>0.1
        frac.append(c.mean(0))
        mr=[]
        for j in range(6):
            run=0;best=0
            for x in c[:,j]:
                run=0 if x else run+1; best=max(best,run)
            mr.append(best)
        maxrun.append(mr)
    frac=np.array(frac);maxrun=np.array(maxrun)
    print(' contact frac per foot FL FR ML MR RL RR:',np.round(frac.mean(0),3))
    print(' median max airborne run (steps):',np.median(maxrun,0),' frac eps with any foot run>90:',(maxrun.max(1)>90).mean())
print('side head creep'); stats(sorted(glob.glob(S+'/raw/*.npz')), lambda s: s==1)
m=json.load(open(W+'/run_meta.json')); lab=m['hold_labels']
for cat,labs in (('side',('lat_l','lat_r')),('fwd',('fwd',))):
    files=[W+f"/raw/episode_{e['env_index']:02d}.npz" for e in m['env_index_map'] if e['category']==cat]
    ids=[lab.index(l) for l in labs]
    print('W',cat); stats(files, lambda s: np.isin(s,ids))
