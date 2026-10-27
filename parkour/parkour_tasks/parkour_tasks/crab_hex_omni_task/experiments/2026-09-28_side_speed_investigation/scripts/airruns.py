import numpy as np, glob, json
S='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39'
W='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29'
T3='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation/step0_E3_X08-t3_visible/suite__omni/seed001/2026-09-23_21-38-11'
def runs(files):
    air=[];con=[];secs=0
    for f,segs in files:
        z=np.load(f); m=np.isin(z['segment_id'],segs)&z['steady_mask']
        if z['done'].any(): m[np.argmax(z['done']):]=False
        idx=np.where(m)[0]
        if len(idx)<50: continue
        c=z['foot_force_norm'][idx[0]:idx[-1]+1]>0.1; secs+=len(c)*0.02
        for j in range(6):
            x=c[:,j]; b=np.flatnonzero(np.diff(x.astype(int)))+1; seg=np.split(x,b)
            for s in seg[1:-1]:
                (con if s[0] else air).append(len(s))
    air=np.array(air); con=np.array(con)
    return air,con,secs
def omni(stamp,cat,labs):
    m=json.load(open(stamp+'/run_meta.json')); lab=m['hold_labels']; ids=[lab.index(l) for l in labs]
    return [(stamp+f"/raw/episode_{e['env_index']:02d}.npz",ids) for e in m['env_index_map'] if e['category']==cat]
sets={'W fwd':omni(W,'fwd',['fwd']),'W side':omni(W,'side',['lat_l','lat_r']),'W back':omni(W,'back',['back']),
      't3 fwd':omni(T3,'fwd',['fwd']),'t3 side':omni(T3,'side',['lat_l','lat_r']),'t3 back':omni(T3,'back',['back']),
      'sidehead':[(f,[1]) for f in sorted(glob.glob(S+'/raw/*.npz'))]}
bins=[1,5,20,40,60,70,80,90,100,150,10000]
print('airborne-run (force<=0.1N) length histogram, steps; share of runs per bin; + contact runs <5 steps per foot-s')
print(f"{'':10s}"+''.join(f"{f'{bins[i]}-{bins[i+1]-1}':>9s}" for i in range(len(bins)-1))+'   n/foot-s  shortcon/foot-s  p90air')
for k,fs in sets.items():
    a,c,secs=runs(fs); h=np.histogram(a,bins)[0]/len(a)
    print(f"{k:10s}"+''.join(f"{x:9.3f}" for x in h)+f"   {len(a)/(secs*6):.3f}   {(c<5).sum()/(secs*6):.3f}   {np.percentile(a,90):.0f}")
