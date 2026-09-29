import sys, json, os, glob, ast, numpy as np
DEF={'FL':0.31020612922871676,'ML':0.31020612922871676,'RL':0.31020612922871676,'FR':-0.31020612922871676,'MR':-0.31020612922871676,'RR':-0.31020612922871676}
def run(name, stamp, keep):
    meta=json.load(open(os.path.join(stamp,'run_meta.json')))
    jn=meta['joint_names']; jn=ast.literal_eval(jn) if isinstance(jn,str) else jn
    feet=[n.split('_')[0] for n in meta['foot_body_names']]
    kidx=[jn.index(f'{f}_Femur_Tibia_RevoluteJoint') for f in feet]; kdef=np.array([DEF[f] for f in feet])
    labs=meta['hold_labels']; dt=0.02; res={}
    for f in sorted(glob.glob(os.path.join(stamp,'raw','episode_*.npz'))):
        a=np.load(f); seg=a['segment_id']; st=a['steady_mask'].astype(bool)
        F=a['foot_force_norm']; c=F>0.1
        q=a['joint_pos'][:,kidx]; tib=((q-kdef)**2*c).sum(1)
        # air time at touchdown
        T=len(F); air=np.zeros(6); pay=np.zeros(T)
        for t in range(T):
            td=c[t]&(air>0)
            pay[t]=np.sum(np.maximum(air-0.38,0)*td)/dt  # per-second rate like manager (value*dt summed)/s
            air=np.where(c[t],0,air+dt)
        for sid,lab in enumerate(labs):
            if lab not in keep: continue
            m=(seg==sid)&st
            if m.sum()<50: continue
            res.setdefault(lab,[]).append((-0.28*tib[m].mean(), 0.8*pay[m].mean()*dt))
    for lab,v in res.items():
        v=np.array(v); print(f"{name:9s} {lab:6s} n={len(v)} tibia_dev={np.median(v[:,0]):+.3f}/s airtime={np.median(v[:,1]):+.3f}/s")
for x in sys.argv[1:]:
    n,s,l=x.split('|'); run(n,s,l.split(','))
