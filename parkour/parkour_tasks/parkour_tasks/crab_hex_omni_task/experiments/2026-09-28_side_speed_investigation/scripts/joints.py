import numpy as np, glob, json, sys, importlib.util
sys.path.insert(0,'/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools')
spec=importlib.util.spec_from_file_location('rmt','/home/nickmagus/krabby/krabby-research/parkour/parkour_tasks/parkour_tasks/crab_hex_omni_task/experiments/tools/replay_motion_terms.py'); RMT=importlib.util.module_from_spec(spec); sys.modules['rmt']=RMT; spec.loader.exec_module(RMT)
S='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/side_phases/2026-09-11_0414_sideways_formation/seed2_2a_006_A15pB/slow__side/seed001/2026-09-12_09-18-39'
W='/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/gait_eval/omni_phases/2026-09-23_clock_deprecation_w2/seed3_X03_c6x_A15pB/suite__omni/seed001/2026-09-25_20-31-29'
d=RMT._yaml_load_tolerant('/home/nickmagus/krabby/krabby-research/parkour/logs/rsl_rl/crab_hex_omni_flat_walk/2026-09-25_14-53-27_seed3_X03_c6x_A15pB/params/env.yaml')
dj=d['scene']['robot']['init_state']['joint_pos']
jn=json.load(open(W+'/run_meta.json'))['joint_names']
q0=RMT.resolve_default_pos(dj,jn)
LEGS=('FL','FR','ML','MR','RL','RR')
types=['Body_Hip','Body_CamShaft','Hip_Femur','Femur_Tibia']
def run(files,segfn):
    acc={t:{'st_dev':[],'st_range':[],'sw_dev':[]} for t in types}
    for f,segs in files:
        z=np.load(f); m=np.isin(z['segment_id'],segs)&z['steady_mask']
        if m.sum()<50: continue
        q=z['joint_pos'][m].astype(float); F=z['foot_force_norm'][m]
        for t in types:
            for li,leg in enumerate(LEGS):
                n=f"{leg}_{t}_RevoluteJoint"
                if n not in jn: continue
                j=jn.index(n); st=F[:,li]>0.1
                dq=q[:,j]-q0[j]
                if st.sum()>5:
                    acc[t]['st_dev'].append(np.sqrt((dq[st]**2).mean())); acc[t]['st_range'].append(np.percentile(q[st,j],95)-np.percentile(q[st,j],5))
                if (~st).sum()>5: acc[t]['sw_dev'].append(np.sqrt((dq[~st]**2).mean()))
    return {t:{k:float(np.median(v)) if v else float('nan') for k,v in a.items()} for t,a in acc.items()}
m=json.load(open(W+'/run_meta.json')); lab=m['hold_labels']
def wfiles(cat,labs): return [(W+f"/raw/episode_{e['env_index']:02d}.npz",[lab.index(l) for l in labs]) for e in m['env_index_map'] if e['category']==cat]
res={'W fwd':run(wfiles('fwd',['fwd']),None),'W side':run(wfiles('side',['lat_l','lat_r']),None),'sidehead':run([(f,[1]) for f in sorted(glob.glob(S+'/raw/*.npz'))],None)}
print('q0:',{t:round(float(q0[jn.index('ML_'+t+'_RevoluteJoint')]),3) for t in types})
for t in types:
    for k in ('st_dev','st_range','sw_dev'):
        print(f"{t:14s} {k:9s}"+''.join(f"  {n}={r[t][k]:.3f}" for n,r in res.items()))
